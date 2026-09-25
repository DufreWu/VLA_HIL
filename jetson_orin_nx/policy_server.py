"""Stage 4: single-client HTTP SmolVLA inference server (also runs on host GPU)."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common.contract import contract, check_contract, vector, action_chunk, decode_image, IMAGE_KEY


class Engine:
    def __init__(self, model, device):
        from importlib.metadata import version
        if version('lerobot') != '0.4.3':
            raise RuntimeError('Use LeRobot 0.4.3 on both training host and NX.')
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.factory import make_pre_post_processors
        check_contract(json.loads((Path(model)/'scene_contract.json').read_text()))
        self.torch, self.device = torch, device
        config = SmolVLAConfig.from_pretrained(model)
        config.device = device
        self.policy = SmolVLAPolicy.from_pretrained(model, config=config).to(device).eval()
        self.policy.config.device = device
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(model),
            preprocessor_overrides={'device_processor': {'device': device}})
        if self.policy.config.output_features['action'].shape != (9,):
            # Some serializers deserialize tuples as lists.
            if list(self.policy.config.output_features['action'].shape) != [9]:
                raise ValueError('Expected nine-dimensional action output.')
        if set(self.policy.config.image_features) != {IMAGE_KEY}:
            raise ValueError('Checkpoint cameras do not match the one-camera scene.')

    def infer(self, request):
        torch = self.torch
        state = vector(request['state'])
        rgb = decode_image(request['rgb_png'])
        observation = {'observation.state': torch.from_numpy(state.copy()),
                       IMAGE_KEY: torch.from_numpy(rgb).permute(2, 0, 1).float()/255.,
                       'task': request['task']}
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        self.policy.reset()  # One-observation policy; do not retain select_action queues.
        batch = self.pre(observation)
        with torch.inference_mode():
            normalized = self.policy.predict_action_chunk(batch)
            # Process each [B,D] action through the saved unnormalizer.
            actions = torch.stack([self.post(normalized[:, i, :])[0]
                                   for i in range(normalized.shape[1])]).cpu().numpy()
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        return action_chunk(actions), (time.perf_counter()-start)*1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--device', default='cuda')
    p.add_argument('--mock', action='store_true', help='Hold current joints; transport testing ONLY')
    a = p.parse_args()
    if not a.mock and a.model is None:
        p.error('--model is required unless --mock is used')
    engine = None if a.mock else Engine(str(a.model.resolve()), a.device)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                self.reply(404, {'error': 'unknown endpoint'}); return
            self.reply(200, {'contract': contract(), 'mock': a.mock, 'model': str(a.model)})

        def do_POST(self):
            if self.path != '/infer':
                self.reply(404, {'error': 'unknown endpoint'}); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 2_000_000:
                    raise ValueError('Invalid request size')
                self.connection.settimeout(30)
                request = json.loads(self.rfile.read(length))
                check_contract(request['contract'])
                state = vector(request['state'])
                if a.mock:
                    decode_image(request['rgb_png'])
                    actions, inference_ms = np.tile(state, (50, 1)), 0.
                else:
                    actions, inference_ms = engine.infer(request)
                self.reply(200, {'episode_id': request['episode_id'], 'obs_id': request['obs_id'],
                                'actions': actions.tolist(), 'inference_ms': inference_ms, 'mock': a.mock})
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {'error': str(exc)})
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.reply(500, {'error': str(exc)})

    server = HTTPServer((a.host, a.port), Handler)
    print(f'Listening on {a.host}:{a.port}; mock={a.mock}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
