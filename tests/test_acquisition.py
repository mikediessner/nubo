import unittest
import torch
import numpy as np

from nubo.acquisition import *
from nubo.models import *
from gpytorch.likelihoods import GaussianLikelihood


class TestExpectedImprovement(unittest.TestCase):

    def test_EI_with_torch_input(self):
        """
        Test that the expected improvement acquisition function returns a
        torch.Tensor of size 1 with values in [0., 1.] if input is a
        torch.Tensor.
        """

        # inputs
        n = 40
        d = 4
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = ExpectedImprovement(gp=gp, y_best=torch.max(y))
        X_test = torch.rand((1, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, torch.Tensor)
        self.assertEqual(ei.size(), torch.Size([1]))
        self.assertTrue(torch.min(-ei) >= 0.)
        self.assertTrue(torch.max(-ei) <= 1.)

    def test_EI_with_numpy_input(self):
        """
        Test that the expected improvement acquisition function returns a
        np.ndarray of shape () with values in [0., 1.] if input is a np.ndarray.
        """

        # inputs
        n = 40
        d = 4
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = ExpectedImprovement(gp=gp, y_best=torch.max(y))
        X_test = np.random.rand(1, d)
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, np.ndarray)
        self.assertEqual(ei.shape, ())
        self.assertTrue(np.min(-ei) >= 0)
        self.assertTrue(np.max(-ei) <= 1.)


class TestUpperConfidenceBound(unittest.TestCase):

    def test_UCB_with_torch_input(self):
        """
        Test that the upper confidence bound acquisition function returns a
        torch.Tensor of size 1 if input is a torch.Tensor.
        """

        # inputs
        n = 40
        d = 4
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = UpperConfidenceBound(gp=gp)
        X_test = torch.rand((1, d))
        ucb = acq(X_test)

        # test
        self.assertIsInstance(ucb, torch.Tensor)
        self.assertEqual(ucb.size(), torch.Size([1]))


    def test_UCB_with_numpy_input(self):
        """
        Test that the upper confidence bound acquisition acquisition function
        returns a np.ndarray of shape () if input is a np.ndarray.
        """

        # inputs
        n = 40
        d = 4
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = UpperConfidenceBound(gp=gp)
        X_test = np.random.rand(1, d)
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, np.ndarray)
        self.assertEqual(ei.shape, ())


class TestMCExpectedImprovement(unittest.TestCase):

    def test_MC_EI_with_torch_input(self):
        """
        Test that the Monte Carlo expected improvement acquisition function
        returns a torch.Tensor of size 1 with values in [0., 1.] if input is a
        torch.Tensor.
        """

        # inputs
        n = 40
        d = 4
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCExpectedImprovement(gp=gp, y_best=torch.max(y), samples=32)
        X_test = torch.rand((1, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, torch.Tensor)
        self.assertEqual(ei.size(), torch.Size())
        self.assertTrue(torch.min(-ei) >= 0.)
        self.assertTrue(torch.max(-ei) <= 1.)

    def test_EI_with_numpy_input(self):
        """
        Test that the expected improvement acquisition function returns a 
        np.ndarray of shape 1 with values in [0., 1.] if input is a np.ndarray.
        """

        # inputs
        n = 40
        d = 4
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCExpectedImprovement(gp=gp, y_best=torch.max(y), samples=32)
        X_test = np.random.rand(1, d)
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, np.ndarray)
        self.assertEqual(ei.shape, ())
        self.assertTrue(np.min(-ei) >= 0)
        self.assertTrue(np.max(-ei) <= 1.)
    
    def test_MC_EI_with_pending_points(self):
        """
        Test that the Monte Carlo expected improvement acquisition function
        returns a torch.Tensor of size 1 with values in [0., 1.] if input is a
        torch.Tensor.
        """

        # inputs
        n = 40
        n_pending = 5
        d = 4
        X = torch.rand((n, d))
        X_pending = torch.rand((n_pending, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCExpectedImprovement(gp=gp, y_best=torch.max(y), x_pending=X_pending, samples=32)
        X_test = torch.rand((1, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, torch.Tensor)
        self.assertEqual(ei.size(), torch.Size())
        self.assertTrue(torch.min(-ei) >= 0.)
        self.assertTrue(torch.max(-ei) <= 1.)


if __name__ == "__main__":
    unittest.main()
