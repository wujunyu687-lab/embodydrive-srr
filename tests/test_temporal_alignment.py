import unittest

import torch

from embodydrive.infer_full_episode import aggregate_causal_raw, causal_conditions
from embodydrive.train_g0 import downsample_temporal


class TemporalAlignmentTest(unittest.TestCase):
    def test_first_frame_plus_four_grouping(self):
        values = torch.zeros(9, 7)
        values[:, 0] = torch.arange(9)
        grouped = aggregate_causal_raw(values, "action")
        self.assertEqual(tuple(grouped.shape), (3, 7))
        self.assertEqual(grouped[:, 0].tolist(), [0.0, 2.5, 6.5])

    def test_batch_downsample_matches_causal_clock(self):
        values = torch.zeros(1, 9, 7)
        values[0, :, 0] = torch.arange(9)
        grouped = downsample_temporal(values, latent_frames=3)
        self.assertEqual(grouped[0, :, 0].tolist(), [0.0, 2.5, 6.5])

    def test_normalized_condition_window(self):
        raw = torch.zeros(9, 7)
        raw[:, 0] = torch.arange(9)
        stats = {
            "action": {"p01": [0.0] * 7, "p99": [8.0] + [1.0] * 6},
        }
        result = causal_conditions(raw, 1, 2, stats, "action")
        self.assertEqual(tuple(result.shape), (1, 2, 7))
        self.assertAlmostEqual(float(result[0, 0, 0]), -0.375, places=6)


if __name__ == "__main__":
    unittest.main()
