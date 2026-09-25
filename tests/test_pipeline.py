import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from common.contract import (contract, check_contract, action_chunk, trim_chunk,
                             encode_image, decode_image, success_metrics, TARGET, FPS)
from training.convert_dataset import validate_episode


class PipelineTests(unittest.TestCase):
    def test_lossless_camera_transport(self):
        rgb = np.random.default_rng(2).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        np.testing.assert_array_equal(rgb, decode_image(encode_image(rgb)))

    def test_reject_bad_model_contract_and_nan_actions(self):
        c = contract(); c['joints'] = list(reversed(c['joints']))
        with self.assertRaises(ValueError): check_contract(c)
        with self.assertRaises(ValueError): action_chunk([[float('nan')]*9])

    def test_late_chunk_uses_current_slot(self):
        actions = np.arange(50*9).reshape(50, 9)
        out, age = trim_chunk(actions, 100, 107, 10)
        self.assertEqual(age, 7)
        np.testing.assert_array_equal(out, actions[7:17])
        out, _ = trim_chunk(actions, 100, 151, 10)
        self.assertEqual(len(out), 0)

    def test_success_requires_lift_release_and_settling(self):
        poses = np.tile(TARGET, (FPS, 1))
        fingers = np.full((FPS, 2), .04)
        self.assertTrue(success_metrics(poses, fingers, True)['success'])
        self.assertFalse(success_metrics(poses, fingers, False)['success'])
        self.assertFalse(success_metrics(poses, np.zeros_like(fingers), True)['success'])
        poses[-1, 0] += .10
        self.assertFalse(success_metrics(poses, fingers, True)['success'])

    def test_raw_episode_and_bad_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            ep = Path(directory); (ep/'rgb').mkdir()
            (ep/'metadata.json').write_text(json.dumps({'success': True, 'fps': FPS}))
            for i in range(3):
                Image.new('RGB', (256, 256)).save(ep/'rgb'/f'{i:06d}.png')
            def save(ts):
                np.savez(ep/'trajectory.npz', state=np.zeros((3, 9)), action=np.ones((3, 9)), timestamp=ts)
            save([0, .05, .1]); validate_episode(ep)
            save([0, .05, .2])
            with self.assertRaises(ValueError): validate_episode(ep)

    def test_mock_server_full_request_and_error(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
        process = subprocess.Popen([sys.executable, str(ROOT/'jetson_orin_nx/policy_server.py'),
                    '--mock', '--host', '127.0.0.1', '--port', str(port)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f'http://127.0.0.1:{port}'
        try:
            for _ in range(100):
                try:
                    with urllib.request.urlopen(base+'/health', timeout=.2) as r:
                        self.assertTrue(json.load(r)['mock'])
                    break
                except (OSError, urllib.error.URLError):
                    if process.poll() is not None: self.fail('Server exited')
                    time.sleep(.02)
            else: self.fail('Server did not start')
            payload = dict(contract=contract(), episode_id='unit-test', obs_id=12,
                           state=[0.]*9, rgb_png=encode_image(np.zeros((256,256,3), dtype=np.uint8)), task='test')
            def post(body):
                req = urllib.request.Request(base+'/infer', data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=5) as r: return json.load(r)
            out = post(payload)
            self.assertEqual(out['obs_id'], 12)
            self.assertEqual(out['episode_id'], 'unit-test')
            self.assertEqual(action_chunk(out['actions']).shape, (50, 9))
            payload['state'] = [0]
            with self.assertRaises(urllib.error.HTTPError) as caught: post(payload)
            self.assertEqual(caught.exception.code, 400)
        finally:
            process.terminate(); process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
