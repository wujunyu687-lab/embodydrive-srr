import unittest

import torch

from embodydrive.rollout import select_continuous_candidate


class RolloutUtilityTest(unittest.TestCase):
    def test_continuous_candidate_selection(self):
        history = torch.tensor([0.0, 1.0]).view(1, 1, 2, 1, 1)
        candidates = torch.tensor([2.0, 9.0]).view(2, 1, 1, 1, 1, 1)
        selected = select_continuous_candidate(candidates, history)
        self.assertEqual(tuple(selected.shape), (1, 1, 1, 1, 1))
        self.assertEqual(float(selected.item()), 2.0)


if __name__ == "__main__":
    unittest.main()
