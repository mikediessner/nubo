import unittest
import torch

from nubo.optimisation import *


class Test(unittest.TestCase):

    def test_bounds_type(self):
        """
        Test that bounds are a 2 x dims torch.Tenosr.
        """

        dims = 4
        func = Ackley(dims=dims)
        self.assertIsInstance(func.bounds, torch.Tensor)
        self.assertEqual(func.bounds.size(), torch.Size([2, dims]))