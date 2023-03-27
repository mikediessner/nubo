import unittest
import torch

from nubo.utils import *


class TestGenInputs(unittest.TestCase):

    def test_without_bounds(self):
        """
        Test that generated inputs are a n x d torch.Tensor and in the range
        [0, 1]^d.
        """

        # inputs
        n = 20
        d = 6

        # run code
        X = gen_inputs(n, d)

        # test
        self.assertIsInstance(X, torch.Tensor)
        self.assertEqual(X.size(), torch.Size([n, d]))
        self.assertTrue(torch.min(X) >= 0. and torch.max(X <= 1.))


    def test_with_bounds(self):
        """
        Test that generated inputs are a n x d torch.Tensor and in the
        specified range.
        """

        # inputs
        n = 20
        d = 3
        lb = torch.Tensor([0.25, -10., 10.])
        ub = torch.Tensor([0.75, 0., 100.])
        bounds = torch.stack([lb, ub])

        # run code
        X = gen_inputs(num_points=n, num_dims=d, bounds=bounds)

        # test
        self.assertIsInstance(X, torch.Tensor)
        self.assertEqual(X.size(), torch.Size([n, d]))
        self.assertTrue(torch.min(X[:, 0]) >= lb[0] and torch.max(X[:, 0] <= ub[0]))
        self.assertTrue(torch.min(X[:, 1]) >= lb[1] and torch.max(X[:, 1] <= ub[1]))
        self.assertTrue(torch.min(X[:, 2]) >= lb[2] and torch.max(X[:, 2] <= ub[2]))


class TestLatinHypercube(unittest.TestCase):
     
    def test_random_LHS(self):
        """
        Test that random Latin hypercube design is a n x d torch.Tensor.
        """

        # inputs
        n = 20
        d = 6

        # run code
        lhs = LatinHypercubeSampling(dims = d)
        X = lhs.random(n)

        # test
        self.assertIsInstance(X, torch.Tensor)
        self.assertEqual(X.size(), torch.Size([n, d]))

    def test_maximin_LHS(self):
        """
        Test that maximin Latin hypercube design is a n x d torch.Tensor.
        """

        # inputs
        n = 20
        d = 6

        # run code
        lhs = LatinHypercubeSampling(dims = d)
        X = lhs.maximin(n)

        # test
        self.assertIsInstance(X, torch.Tensor)
        self.assertEqual(X.size(), torch.Size([n, d]))


class TestStandardise(unittest.TestCase):
     
     def test_shape(self):
        """
        Test that standardised output is a n torch.Tensor.
        """

        # inputs
        n = 20
        y = torch.rand(n)

        # run code
        y_stand = standardise(y)

        # test
        self.assertIsInstance(y_stand, torch.Tensor)
        self.assertEqual(y_stand.size(), torch.Size([n]))


class TestNormalise(unittest.TestCase):

    def test_shape(self):
        """
        Test that normalised output is a n x d torch.Tensor and in the range
        [0, 1]^d.
        """

        # inputs
        n = 20
        d = 3
        
        lb = torch.Tensor([-3., 0., -5.])
        ub = torch.Tensor([10., 5., 15.])
        bounds = torch.stack([lb, ub])
        X = torch.rand((n, d)) * (ub - lb) + lb

        # run code
        X_norm = normalise(X, bounds=bounds)

        # test
        self.assertIsInstance(X_norm, torch.Tensor)
        self.assertEqual(X_norm.size(), torch.Size([n, d]))
        self.assertTrue(torch.min(X_norm) >= 0. and torch.max(X_norm<= 1.))


class TestUnnormalise(unittest.TestCase):

    def test_shape(self):
        """
        Test that unnormalised output is a n x d torch.Tensor and in the range
        [lb, ub]^d.
        """

        # inputs
        n = 20
        d = 3
        X = torch.rand((n, d))
        lb = torch.Tensor([-3., 0., -5.])
        ub = torch.Tensor([10., 5., 15.])
        bounds = torch.stack([lb, ub])

        # run code
        X_norm = normalise(X, bounds=bounds)

        # test
        self.assertIsInstance(X_norm, torch.Tensor)
        self.assertEqual(X_norm.size(), torch.Size([n, d]))
        self.assertTrue(torch.min(X[:, 0]) >= lb[0] and torch.max(X[:, 0] <= ub[0]))
        self.assertTrue(torch.min(X[:, 1]) >= lb[1] and torch.max(X[:, 1] <= ub[1]))
        self.assertTrue(torch.min(X[:, 2]) >= lb[2] and torch.max(X[:, 2] <= ub[2]))


if __name__ == "__main__":
    unittest.main()
