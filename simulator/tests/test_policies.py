import unittest
from simulator.evaluation.policies import validate_response


class PolicyTests(unittest.TestCase):
    def test_response_identity_shape_and_latency(self):
        response = dict(episode_id='ep', obs_id=3, actions=[[0.]*9]*5, inference_ms=12.)
        actions, latency = validate_response(response, 'ep', 3)
        self.assertEqual(actions.shape, (5, 9))
        self.assertEqual(latency, 12.)
        with self.assertRaises(ValueError): validate_response(response, 'other', 3)
        with self.assertRaises(ValueError): validate_response(response, 'ep', 4)
        for value in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                validate_response(dict(response, inference_ms=value), 'ep', 3)
        with self.assertRaises(ValueError):
            validate_response(dict(response, actions=[[0.]*8]), 'ep', 3)
