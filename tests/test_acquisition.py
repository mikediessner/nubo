import unittest
import torch
import numpy as np

from nubo.acquisition import *
from nubo.models import *
from gpytorch.likelihoods import GaussianLikelihood


class TestExpectedImprovement(unittest.TestCase):

    def test_EI_with_torch_input(self):
        """
        Test that the expected improvement acquisition function returns a m
        torch.Tensor with values in [0., 1.] if input is a torch.Tensor.
        """

        # inputs
        n = 40
        d = 4
        m = 10
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = ExpectedImprovement(gp=gp, y_best=torch.max(y))
        X_test = torch.rand((m, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, torch.Tensor)
        self.assertEqual(ei.size(), torch.Size([m]))
        self.assertTrue(torch.min(-ei) >= 0.)
        self.assertTrue(torch.max(-ei) <= 1.)

    def test_EI_with_numpy_input(self):
        """
        Test that the expected improvement acquisition function returns a m
        np.ndarray with values in [0., 1.] if input is a np.ndarray.
        """

        # inputs
        n = 40
        d = 4
        m = 10
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = ExpectedImprovement(gp=gp, y_best=torch.max(y))
        X_test = np.random.rand(m, d)
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, np.ndarray)
        self.assertEqual(ei.shape, (m,))
        self.assertTrue(np.min(-ei) >= 0)
        self.assertTrue(np.max(-ei) <= 1.)


class TestUpperConfidenceBound(unittest.TestCase):

    def test_UCB_with_torch_input(self):
        """
        Test that the upper confidence bound acquisition function returns a m
        torch.Tensor if input is a torch.Tensor.
        """

        # inputs
        n = 40
        d = 4
        m = 10
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = UpperConfidenceBound(gp=gp)
        X_test = torch.rand((m, d))
        ucb = acq(X_test)

        # test
        self.assertIsInstance(ucb, torch.Tensor)
        self.assertEqual(ucb.size(), torch.Size([m]))


    def test_UCB_with_numpy_input(self):
        """
        Test that the upper confidence bound acquisition acquisition function returns a m
        np.ndarray if input is a np.ndarray.
        """

        # inputs
        n = 40
        d = 4
        m = 10
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = UpperConfidenceBound(gp=gp)
        X_test = np.random.rand(m, d)
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, np.ndarray)
        self.assertEqual(ei.shape, (m,))


class TestMCExpectedImprovement(unittest.TestCase):

    def test_MC_EI_with_torch_input(self):
        """
        Test that the expected improvement acquisition function returns a m
        torch.Tensor with values in [0., 1.] if input is a torch.Tensor.
        """

        # inputs
        n = 40
        d = 4
        m = 10
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = ExpectedImprovement(gp=gp, y_best=torch.max(y))
        X_test = torch.rand((m, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, torch.Tensor)
        self.assertEqual(ei.size(), torch.Size([m]))
        self.assertTrue(torch.min(-ei) >= 0.)
        self.assertTrue(torch.max(-ei) <= 1.)

    def test_EI_with_numpy_input(self):
        """
        Test that the expected improvement acquisition function returns a m
        np.ndarray with values in [0., 1.] if input is a np.ndarray.
        """

        # inputs
        n = 40
        d = 4
        m = 10
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = ExpectedImprovement(gp=gp, y_best=torch.max(y))
        X_test = np.random.rand(m, d)
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, np.ndarray)
        self.assertEqual(ei.shape, (m,))
        self.assertTrue(np.min(-ei) >= 0)
        self.assertTrue(np.max(-ei) <= 1.)


if __name__ == "__main__":
    unittest.main()
