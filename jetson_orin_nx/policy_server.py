"""Stage 4: single-client HTTP SmolVLA inference server (also runs on host GPU)."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common.contract import (contract, check_contract, vector, action_chunk, decode_image,
                             encode_image, IMAGE_KEY)


class Engine:
    def __init__(self, model, device):
        from importlib.metadata import version
        if version('lerobot') != '0.4.3':
            raise RuntimeError('Use LeRobot 0.4.3 on both training host and NX.')
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        check_contract(json.loads((Path(model)/'scene_contract.json').read_text()))
        self.torch, self.device = torch, device
        config = PreTrainedConfig.from_pretrained(model)
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
        camera_keys = set(self.policy.config.image_features)

        if IMAGE_KEY not in camera_keys:
            raise ValueError(
                f"Scene camera {IMAGE_KEY!r} is absent from "
                f"checkpoint camera keys: {sorted(camera_keys)}"
            )

        unused_cameras = camera_keys - {IMAGE_KEY}
        if unused_cameras:
            print(
                f"Using scene camera: {IMAGE_KEY}; "
                f"checkpoint cameras not supplied: {sorted(unused_cameras)}"
            )

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
        start_mono = time.monotonic()
        if request.get('noise_seed') is not None:
            # Reproducible flow-matching noise: identical chunk index -> identical noise.
            torch.manual_seed(int(request['noise_seed']))
        self.policy.reset()  # One-observation policy; do not retain select_action queues.
        batch = self.pre(observation)
        with torch.inference_mode():
            normalized = self.policy.predict_action_chunk(batch)
            # Process each [B,D] action through the saved unnormalizer.
            actions = torch.stack([self.post(normalized[:, i, :])[0]
                                   for i in range(normalized.shape[1])]).cpu().numpy()
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter()-start)*1000
        # Same timed region on the monotonic clock used by the power monitor.
        self.last_window = (start_mono, time.monotonic())
        return action_chunk(actions), elapsed_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--device', default='cuda')
    p.add_argument('--mock', action='store_true', help='Hold current joints; transport testing ONLY')
    p.add_argument('--energy', action='store_true',
                   help='Sample VDD_IN and expose /episode/start and /episode/stop energy windows')
    p.add_argument('--dvfs-control', action='store_true',
                   help='Expose POST /dvfs to pin CPU/GPU frequencies (needs sudo; implies --energy). '
                        'Original DVFS settings are restored on exit.')
    p.add_argument('--power-interval-s', type=float, default=0.005)
    p.add_argument('--dvfs-warmup', type=int, default=3, help='Inferences after each DVFS switch')
    a = p.parse_args()
    if not a.mock and a.model is None:
        p.error('--model is required unless --mock is used')
    engine = None if a.mock else Engine(str(a.model.resolve()), a.device)
    dvfs = monitor = None
    if a.dvfs_control:
        from jetson_orin_nx.benchmark.dvfs_control import DvfsController
        dvfs = DvfsController()
        dvfs.check_writable()
        dvfs.take_snapshot()
    if a.energy or a.dvfs_control:
        from jetson_orin_nx.benchmark.dvfs_control import PowerMonitor, DvfsController as _D
        board = dvfs or _D()
        monitor = PowerMonitor(board.ina_path, a.power_interval_s, freq_reader=board.read_freqs).start()

    def warmup(count):
        if engine is None:
            return []
        request = dict(state=[0.0]*9, task='Grasp the red cube and lift it above the table.',
                       rgb_png=encode_image(np.zeros((256, 256, 3), dtype=np.uint8)))
        return [engine.infer(request)[1] for _ in range(max(0, int(count)))]

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/dvfs' and dvfs is not None:
                self.reply(200, dvfs.status()); return
            if self.path == '/dvfs/capabilities' and dvfs is not None:
                self.reply(200, dvfs.capabilities()); return
            if self.path != '/health':
                self.reply(404, {'error': 'unknown endpoint'}); return
            self.reply(200, {'contract': contract(), 'mock': a.mock, 'model': str(a.model),
                             'energy': monitor is not None, 'dvfs_control': dvfs is not None,
                             'dvfs_modes': ['fixed', 'governor'] if dvfs is not None else [],
                             'features': ['request_timing', 'inference_energy', 'dvfs_capabilities',
                                          'symbolic_frequencies']})

        def do_POST(self):
            control = {'/dvfs': dvfs, '/episode/start': monitor, '/episode/stop': monitor}
            if self.path in control:
                if control[self.path] is None:
                    self.reply(404, {'error': f'{self.path} disabled; start server with --dvfs-control/--energy'})
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0')) or 0) or b'{}')
                    if self.path == '/dvfs':
                        settle = float(body.get('settle_s', 1.0))
                        if body.get('resolve_only'):  # Validate/resolve without touching sysfs.
                            cpu, gpu = dvfs.resolve(body['cpu_khz'], body['gpu_hz'])
                            self.reply(200, dict(cpu_khz=cpu, gpu_hz=gpu)); return
                        if body.get('cpu_governor') or body.get('gpu_governor'):
                            status = dvfs.apply_governor(body['cpu_governor'], body['gpu_governor'], settle)
                        else:
                            status = dvfs.apply(body['cpu_khz'], body['gpu_hz'], settle)
                        status['name'] = body.get('name')
                        status['warmup_ms'] = warmup(body.get('warmup', a.dvfs_warmup))
                        print(f"DVFS -> {body.get('name')}: {status['cpu_khz'][0]} kHz / {status['gpu_hz']} Hz "
                              f"governors {status['cpu_governors'][0]}/{status['gpu_governor']} "
                              f"valid={status['frequency_valid']}", flush=True)
                        self.reply(200, status)
                    elif self.path == '/episode/start':
                        self.reply(200, monitor.begin_episode(body['tag']))
                    else:
                        self.reply(200, monitor.end_episode(body['tag'], active_s=body.get('active_s'),
                                                            include_samples=bool(body.get('include_samples'))))
                except (ValueError, KeyError, TypeError) as exc:
                    self.reply(400, {'error': str(exc)})
                except OSError as exc:
                    self.reply(500, {'error': f'sysfs write failed: {exc}'})
                return
            if self.path != '/infer':
                self.reply(404, {'error': 'unknown endpoint'}); return
            t_recv = time.monotonic()  # Headers parsed; body not yet read.
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 2_000_000:
                    raise ValueError('Invalid request size')
                self.connection.settimeout(30)
                request = json.loads(self.rfile.read(length))
                check_contract(request['contract'])
                state = vector(request['state'])
                t_parsed = time.monotonic()
                if a.mock:
                    decode_image(request['rgb_png'])
                    actions, inference_ms = np.tile(state, (50, 1)), 0.
                    t0 = t1 = time.monotonic()
                else:
                    actions, inference_ms = engine.infer(request)
                    t0, t1 = engine.last_window  # Model forward pass only (after image decode).
                    if monitor is not None:
                        monitor.record_inference(t0, t1, dict(episode_id=request['episode_id'],
                                                              obs_id=request['obs_id']))
                body = {'episode_id': request['episode_id'], 'obs_id': request['obs_id'],
                        'actions': actions.tolist(), 'inference_ms': inference_ms, 'mock': a.mock}
                # Server-clock durations only; the client never subtracts across hosts.
                t_reply = time.monotonic()
                body['server_timing_ms'] = dict(receive_to_parsed=(t_parsed-t_recv)*1000,
                                                parsed_to_inference_start=(t0-t_parsed)*1000,
                                                inference=(t1-t0)*1000,
                                                inference_end_to_reply=(t_reply-t1)*1000,
                                                total=(t_reply-t_recv)*1000)
                self.reply(200, body)
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {'error': str(exc)})
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.reply(500, {'error': str(exc)})

    import signal

    def _terminate(signum, frame):
        raise KeyboardInterrupt  # Route SIGTERM through the DVFS-restoring finally block.
    signal.signal(signal.SIGTERM, _terminate)
    server = HTTPServer((a.host, a.port), Handler)
    print(f'Listening on {a.host}:{a.port}; mock={a.mock}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if monitor is not None:
            monitor.close()
        if dvfs is not None:
            dvfs.restore()


if __name__ == '__main__':
    main()
