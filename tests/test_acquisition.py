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
        torch.Tensor of size [1] with values in [0, 1] if input is a
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
        self.assertTrue(-ei >= 0.)
        self.assertTrue(-ei <= 1.)

    def test_EI_with_numpy_input(self):
        """
        Test that the expected improvement acquisition function returns a
        float between [0, 1] if input is a np.ndarray.
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
        X_test = np.float32(np.random.rand(1, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, float)
        self.assertTrue(-ei >= 0)
        self.assertTrue(-ei <= 1.)


class TestUpperConfidenceBound(unittest.TestCase):

    def test_UCB_with_torch_input(self):
        """
        Test that the upper confidence bound acquisition function returns a
        torch.Tensor of size [1] if input is a torch.Tensor.
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
        returns a float if input is a np.ndarray.
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
        X_test = np.float32(np.random.rand(1, d))
        ucb = acq(X_test)

        # test
        self.assertIsInstance(ucb, float)


class TestMCExpectedImprovement(unittest.TestCase):

    def test_MC_EI_with_single_torch_input(self):
        """
        Test that the Monte Carlo expected improvement acquisition function
        returns a torch.Tensor of size [1] between [0, 1] if input is a
        torch.Tensor of size [1, d].
        """

        # inputs
        n = 40
        d = 4
        q = 1
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCExpectedImprovement(gp=gp, y_best=torch.max(y), samples=32)
        X_test = torch.rand((q, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, torch.Tensor)
        self.assertEqual(ei.size(), torch.Size([1]))
        self.assertTrue(-ei >= 0.)
        self.assertTrue(-ei <= 1.)

    def test_MC_EI_with_multi_torch_inputs(self):
        """
        Test that the Monte Carlo expected improvement acquisition function
        returns a torch.Tensor of size [1] between [0, 1] if input is a
        torch.Tensor of size [4, d].
        """

        # inputs
        n = 40
        d = 4
        q = 4
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCExpectedImprovement(gp=gp, y_best=torch.max(y), samples=32)
        X_test = torch.rand((q, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, torch.Tensor)
        self.assertEqual(ei.size(), torch.Size([1]))
        self.assertTrue(-ei >= 0.)
        self.assertTrue(-ei <= 1.)

    def test_EI_with_single_numpy_input(self):
        """
        Test that the expected improvement acquisition function returns a 
        float between [0., 1.] if input is a np.ndarray of shape (1, d).
        """

        # inputs
        n = 40
        d = 4
        q = 1
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCExpectedImprovement(gp=gp, y_best=torch.max(y), samples=32)
        X_test = np.float32(np.random.rand(q, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, float)
        self.assertTrue(-ei >= 0.)
        self.assertTrue(-ei <= 1.)
    
    def test_EI_with_multi_numpy_inputs(self):
        """
        Test that the expected improvement acquisition function returns a 
        float between [0., 1.] if input is a np.ndarray of shape (4, d).
        """

        # inputs
        n = 40
        d = 4
        q = 4
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCExpectedImprovement(gp=gp, y_best=torch.max(y), samples=32)
        X_test = np.float32(np.random.rand(q, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, float)
        self.assertTrue(-ei >= 0.)
        self.assertTrue(-ei <= 1.)
    
    def test_MC_EI_with_pending_points_and_torch_inputs(self):
        """
        Test that the Monte Carlo expected improvement acquisition function
        returns a torch.Tensor of size [1] between [0., 1.] when pending points
        is a torch.tensor of size [5, d] if input is a torch.Tensor.
        """

        # inputs
        n = 40
        n_pending = 5
        d = 4
        q = 2
        X = torch.rand((n, d))
        X_pending = torch.rand((n_pending, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCExpectedImprovement(gp=gp, y_best=torch.max(y), x_pending=X_pending, samples=32)
        X_test = torch.rand((q, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, torch.Tensor)
        self.assertEqual(ei.size(), torch.Size([1]))
        self.assertTrue(-ei >= 0.)
        self.assertTrue(-ei <= 1.)

    def test_MC_EI_with_pending_points_and_numpy_inputs(self):
        """
        Test that the Monte Carlo expected improvement acquisition function
        returns a float between [0., 1.] when pending points is a torch.tensor
        of size [5, d] if input is a numpy.ndarray.
        """

        # inputs
        n = 40
        n_pending = 5
        d = 4
        q = 2
        X = torch.rand((n, d))
        X_pending = torch.rand((n_pending, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCExpectedImprovement(gp=gp, y_best=torch.max(y), x_pending=X_pending, samples=32)
        X_test = np.float32(np.random.rand(q, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, float)
        self.assertTrue(-ei >= 0.)
        self.assertTrue(-ei <= 1.)

    def test_MC_EI_with_fixed_base_samples_and_torch_inputs(self):
        """
        Test that the Monte Carlo expected improvement acquisition function
        returns a torch.Tensor of size [1] between [0., 1.] when base samples
        are fixed if input is a torch.Tensor.
        """

        # inputs
        n = 40
        d = 4
        q = 5
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCExpectedImprovement(gp=gp, y_best=torch.max(y), samples=32, fix_base_samples=True)
        X_test = torch.rand((q, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, torch.Tensor)
        self.assertEqual(ei.size(), torch.Size([1]))
        self.assertTrue(-ei >= 0.)
        self.assertTrue(-ei <= 1.)

    def test_MC_EI_with_fixed_base_samples_and_numpy_inputs(self):
        """
        Test that the Monte Carlo expected improvement acquisition function
        returns a float between [0., 1.] when base samples are fixed if input
        is a numpy.ndarray.
        """

        # inputs
        n = 40
        d = 4
        q = 5
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCExpectedImprovement(gp=gp, y_best=torch.max(y), samples=32, fix_base_samples=True)
        X_test = np.float32(np.random.rand(q, d))
        ei = acq(X_test)

        # test
        self.assertIsInstance(ei, float)
        self.assertTrue(-ei >= 0.)
        self.assertTrue(-ei <= 1.)


class TestMCUpperConfidenceBound(unittest.TestCase):

    def test_MC_UCB_with_single_torch_input(self):
        """
        Test that the Monte Carlo upper confidence bound acquisition function
        returns a torch.Tensor of size [1] if input is a torch.Tensor of size
        [1, d].
        """

        # inputs
        n = 40
        d = 4
        q = 1
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCUpperConfidenceBound(gp=gp, samples=32)
        X_test = torch.rand((q, d))
        ucb = acq(X_test)

        # test
        self.assertIsInstance(ucb, torch.Tensor)
        self.assertEqual(ucb.size(), torch.Size([1]))

    def test_MC_UCB_with_multi_torch_inputs(self):
        """
        Test that the Monte Carlo upper confidence bound acquisition function
        returns a torch.Tensor of size [1] if input is a torch.Tensor of size
        [4, d].
        """

        # inputs
        n = 40
        d = 4
        q = 4
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCUpperConfidenceBound(gp=gp, samples=32)
        X_test = torch.rand((q, d))
        ucb = acq(X_test)

        # test
        self.assertIsInstance(ucb, torch.Tensor)
        self.assertEqual(ucb.size(), torch.Size([1]))

    def test_UCB_with_single_numpy_input(self):
        """
        Test that the upper confidence bound acquisition function returns a 
        float if input is a np.ndarray of shape (1, d).
        """

        # inputs
        n = 40
        d = 4
        q = 1
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCUpperConfidenceBound(gp=gp, samples=32)
        X_test = np.float32(np.random.rand(q, d))
        ucb = acq(X_test)

        # test
        self.assertIsInstance(ucb, float)

    def test_UCB_with_multi_numpy_inputs(self):
        """
        Test that the upper confidence bound acquisition function returns a 
        float if input is a np.ndarray of shape (4, d).
        """

        # inputs
        n = 40
        d = 4
        q = 4
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCUpperConfidenceBound(gp=gp, samples=32)
        X_test = np.float32(np.random.rand(q, d))
        ucb = acq(X_test)

        # test
        self.assertIsInstance(ucb, float)
    
    def test_MC_UCB_with_pending_points_and_torch_inputs(self):
        """
        Test that the Monte Carlo upper confidence bound acquisition function
        returns a torch.Tensor of size [1] when pending points is a
        torch.tensor of size [5, d] if input is a torch.Tensor.
        """

        # inputs
        n = 40
        n_pending = 5
        d = 4
        q = 2
        X = torch.rand((n, d))
        X_pending = torch.rand((n_pending, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCUpperConfidenceBound(gp=gp, x_pending=X_pending, samples=32)
        X_test = torch.rand((q, d))
        ucb = acq(X_test)

        # test
        self.assertIsInstance(ucb, torch.Tensor)
        self.assertEqual(ucb.size(), torch.Size([1]))

    def test_MC_UCB_with_pending_points_and_numpy_inputs(self):
        """
        Test that the Monte Carlo upper confidence bound acquisition function
        returns a float when pending points is a torch.tensor of size [5, d] if
        input is a numpy.ndarray.
        """

        # inputs
        n = 40
        n_pending = 5
        d = 4
        q = 2
        X = torch.rand((n, d))
        X_pending = torch.rand((n_pending, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCUpperConfidenceBound(gp=gp, x_pending=X_pending, samples=32)
        X_test = np.float32(np.random.rand(q, d))
        ucb = acq(X_test)

        # test
        self.assertIsInstance(ucb, float)

    def test_MC_UCB_with_fixed_base_samples_and_torch_inputs(self):
        """
        Test that the Monte Carlo upper confidence bound acquisition function
        returns a torch.Tensor of size [1] when base samples are fixed if input
        is a torch.Tensor.
        """

        # inputs
        n = 40
        d = 4
        q = 5
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCUpperConfidenceBound(gp=gp, samples=32, fix_base_samples=True)
        X_test = torch.rand((q, d))
        ucb = acq(X_test)

        # test
        self.assertIsInstance(ucb, torch.Tensor)
        self.assertEqual(ucb.size(), torch.Size([1]))

    def test_MC_UCB_with_fixed_base_samples_and_numpy_inputs(self):
        """
        Test that the Monte Carlo upper confidence bound acquisition function
        returns a float when base samples are fixed if input is a
        numpy.ndarray.
        """

        # inputs
        n = 40
        d = 4
        q = 5
        X = torch.rand((n, d))
        y = torch.sum(X**2, axis=1)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)

        # run code
        acq = MCUpperConfidenceBound(gp=gp, samples=32, fix_base_samples=True)
        X_test = np.float32(np.random.rand(q, d))
        ucb = acq(X_test)

        # test
        self.assertIsInstance(ucb, float)

if __name__ == "__main__":
    unittest.main()
