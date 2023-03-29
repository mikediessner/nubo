import unittest
import torch
import numpy as np

from nubo.optimisation import *
from nubo.models import *
from nubo.acquisition import *
from gpytorch.likelihoods import GaussianLikelihood


class TestCandidateGeneration(unittest.TestCase):

    def test_gen_candidates(self):
        """
        Test that a n x d torch.Tensor is returned that sticks to the bounds.
        """

        # inputs
        def f(x):
            if isinstance(x, torch.Tensor):
                res = torch.sum(x**2)
            elif isinstance(x, np.ndarray):
                res = np.sum(x**2)
            return res

        lb = torch.Tensor([-3., 0., -5.])
        ub = torch.Tensor([10., 5., 15.])
        bounds = torch.stack([lb, ub])
        n = 10
        
        # run code
        points = gen_candidates(f, bounds, n, 100)

        # test
        self.assertIsInstance(points, torch.Tensor)
        self.assertEqual(points.size(), (n, 3))
        self.assertTrue(torch.min(points[:, 0]) >= lb[0] and torch.max(points[:, 0] <= ub[0]))
        self.assertTrue(torch.min(points[:, 1]) >= lb[1] and torch.max(points[:, 1] <= ub[1]))
        self.assertTrue(torch.min(points[:, 2]) >= lb[2] and torch.max(points[:, 2] <= ub[2]))


class TestLBFGSB(unittest.TestCase):

    def test_lbfgsb(self):
        """
        Test that L-BGFGS-B returns inputs as a 1 x dims torch.Tensor and ouput
        as a () torch.Tensor.
        """

        # inputs
        def f(x):
            if isinstance(x, torch.Tensor):
                res = torch.sum(x**2)
            elif isinstance(x, np.ndarray):
                res = np.sum(x**2)
            return res

        bounds = torch.tensor([[-5., -5.], [5., 5.]])
        
        # run code
        x, f_x = lbfgsb(f, bounds)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), (1, 2))
        self.assertEqual(f_x.size(), ())


class TestSLSQP(unittest.TestCase):

    def test_slsqp_one_constraint(self):
        """
        Test that SLSQP returns inputs as a 1 x dims torch.Tensor and ouput as
        a () torch.Tensor with one constraint.
        """

        # inputs
        def f(x):
            if isinstance(x, torch.Tensor):
                res = torch.sum(x**2)
            elif isinstance(x, np.ndarray):
                res = np.sum(x**2)
            return res

        bounds = torch.tensor([[-5., -5.], [5., 5.]])
        constraint = {"type": "ineq", "fun": lambda x: 1.0 - x[0]}
        
        # run code
        x, f_x = slsqp(f, bounds=bounds, constraints=constraint)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), (1, 2))
        self.assertEqual(f_x.size(), ())
    
    def test_slsqp_two_constraints(self):
        """
        Test that SLSQP returns inputs as a 1 x dims torch.Tensor and ouput as
        a () torch.Tensor with two constraints.
        """

        # inputs
        def f(x):
            if isinstance(x, torch.Tensor):
                res = torch.sum(x**2)
            elif isinstance(x, np.ndarray):
                res = np.sum(x**2)
            return res

        bounds = torch.tensor([[-5., -5.], [5., 5.]])
        constraints = ({"type": "ineq", "fun": lambda x: 1.0 - x[0]},
                      {"type": "ineq", "fun": lambda x: -3.0 + x[1]})
        
        # run code
        x, f_x = slsqp(f, bounds=bounds, constraints=constraints)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), (1, 2))
        self.assertEqual(f_x.size(), ())


class TestAdam(unittest.TestCase):

    def test_adam(self):
        """
        Test that Adam returns inputs as a 1 x dims torch.Tensor and ouput as a
        () torch.Tensor.
        """

        # inputs
        def f(x):
            if isinstance(x, torch.Tensor):
                res = torch.sum(x**2)
            elif isinstance(x, np.ndarray):
                res = np.sum(x**2)
            return res

        bounds = torch.tensor([[-5., -5.], [5., 5.]])
        
        # run code
        x, f_x = adam(f, bounds)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), (1, 2))
        self.assertEqual(f_x.size(), ())


class TestJoint(unittest.TestCase):

    def test_joint_adam(self):
        """
        Test that Adam returns inputs as a 1 x dims torch.Tensor and ouput as a
        () torch.Tensor.
        """

        # inputs
        n = 20
        d = 2
        m = 4
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        bounds = torch.tensor([[-5., -5.], [5., 5.]])
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32)
        
        # run code
        x, f_x = joint(func=acq, method="Adam", batch_size=m, bounds=bounds, num_starts=1)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), (m, 2))
        self.assertEqual(f_x.size(), ()) # ?????


class TestSequential(unittest.TestCase):

    def test_sequential_adam(self):
        """
        Test that Adam returns inputs as a 1 x dims torch.Tensor and ouput as a
        () torch.Tensor.
        """

        # inputs
        n = 20
        d = 2
        m = 4
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        bounds = torch.tensor([[-5., -5.], [5., 5.]])
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32)
        
        # run code
        x, f_x = sequential(func=acq, method="Adam", batch_size=m, bounds=bounds, num_starts=1)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), (m, 2))
        self.assertEqual(f_x.size(), (m, ))
    
    # def test_sequential_lbfgsb(self):
    #     """
    #     Test that L-BFGS-B returns inputs as a 1 x dims torch.Tensor and ouput
    #     as a () torch.Tensor.
    #     """

    #     # inputs
    #     n = 20
    #     d = 2
    #     m = 4
    #     X = torch.rand((n, d), dtype=torch.float64)
    #     y = torch.sum(X**2, axis=1)
    #     bounds = torch.tensor([[-5., -5.], [5., 5.]])
    #     likelihood = GaussianLikelihood()
    #     gp = GaussianProcess(X, y, likelihood=likelihood)
    #     fit_gp(X, y, gp=gp, likelihood=likelihood)
    #     acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32, fix_base_samples=True)
        
    #     # run code
    #     x, f_x = sequential(func=acq, method="L-BFGS-B", batch_size=m, bounds=bounds, num_starts=1)

    #     # test
    #     self.assertIsInstance(x, torch.Tensor)
    #     self.assertIsInstance(f_x, torch.Tensor)
    #     self.assertEqual(x.size(), (m, 2))
    #     self.assertEqual(f_x.size(), (m, ))
    
    # def test_sequential_slsqp(self):
    #     """
    #     Test that SLSQP returns inputs as a 1 x dims torch.Tensor and ouput as
    #     a () torch.Tensor.
    #     """

    #     # inputs
    #     n = 20
    #     d = 2
    #     m = 4
    #     X = torch.rand((n, d), dtype=torch.float64)
    #     y = torch.sum(X**2, axis=1)
    #     bounds = torch.tensor([[-5., -5.], [5., 5.]])
    #     constraint = {"type": "ineq", "fun": lambda x: 5.0 - x[0]}
    #     likelihood = GaussianLikelihood()
    #     gp = GaussianProcess(X, y, likelihood=likelihood)
    #     fit_gp(X, y, gp=gp, likelihood=likelihood)
    #     acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32, fix_base_samples=True)
        
    #     # run code
    #     x, f_x = sequential(func=acq, method="SLSQP", batch_size=m, bounds=bounds, constraints=constraint, num_starts=1)

    #     # test
    #     self.assertIsInstance(x, torch.Tensor)
    #     self.assertIsInstance(f_x, torch.Tensor)
    #     self.assertEqual(x.size(), (m, 2))
    #     self.assertEqual(f_x.size(), (m, ))


if __name__ == "__main__":
    unittest.main()
