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
        Test that a (n, d) torch.Tensor is returned that sticks to the bounds.
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
        self.assertEqual(points.size(), torch.Size([n, 3]))
        self.assertTrue(torch.min(points[:, 0]) >= lb[0] and torch.max(points[:, 0] <= ub[0]))
        self.assertTrue(torch.min(points[:, 1]) >= lb[1] and torch.max(points[:, 1] <= ub[1]))
        self.assertTrue(torch.min(points[:, 2]) >= lb[2] and torch.max(points[:, 2] <= ub[2]))


class TestLBFGSB(unittest.TestCase):

    def test_lbfgsb(self):
        """
        Test that L-BGFGS-B returns inputs as a (1, dims) torch.Tensor and
        ouput as a (1, ) torch.Tensor. Tests that it sticks to bounds.
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
        x, f_x = single(f, method="L-BFGS-B", bounds=bounds)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([1, 2]))
        self.assertEqual(f_x.size(), torch.Size([1]))
        self.assertTrue(x[0, 0] >= bounds[0, 0])
        self.assertTrue(x[0, 0] <= bounds[1, 0])
        self.assertTrue(x[0, 1] >= bounds[0, 1])
        self.assertTrue(x[0, 1] <= bounds[1, 1])

    def test_lbfgsb_mixed(self):
        """
        Test that L-BGFGS-B with mixed parameters returns inputs as a (1, dims)
        torch.Tensor and ouput as a (1, ) torch.Tensor. Tests that it sticks to
        bounds.
        """

        # inputs
        def f(x):
            if isinstance(x, torch.Tensor):
                res = torch.sum(x**2)
            elif isinstance(x, np.ndarray):
                res = np.sum(x**2)
            return res

        bounds = torch.tensor([[-3., -5., -1.5], [2.2, 5., 2.]])
        discrete = {0: [-3., 0., 2.2], 2: [-1.5, 2.]}
        
        # run code
        x, f_x = single(f, method="L-BFGS-B", bounds=bounds, discrete=discrete)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([1, 3]))
        self.assertEqual(f_x.size(), torch.Size([1]))
        self.assertTrue(x[0, 0] in discrete[0])
        self.assertTrue(x[0, 1] >= bounds[0, 1])
        self.assertTrue(x[0, 1] <= bounds[1, 1])
        self.assertTrue(x[0, 2] in discrete[2])


class TestSLSQP(unittest.TestCase):

    def test_slsqp_one_constraint(self):
        """
        Test that SLSQP returns inputs as a (1, d) torch.Tensor and ouput as a
        (1, ) torch.Tensor with one constraint. Tests that it sticks to bounds
        and constraint.
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
        x, f_x = single(f, method="SLSQP", bounds=bounds, constraints=constraint)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([1, 2]))
        self.assertEqual(f_x.size(), torch.Size([1]))
        self.assertTrue(x[0, 0] <= 1.0)
        self.assertTrue(x[0, 1] >= bounds[0, 1])
        self.assertTrue(x[0, 1] <= bounds[1, 1])
    
    def test_slsqp_two_constraints(self):
        """
        Test that SLSQP returns inputs as a (1, d) torch.Tensor and ouput as a
        (1, ) torch.Tensor with two constraints. Tests that it sticks to bounds
        and constraints.
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
        x, f_x = single(f, method="SLSQP", bounds=bounds, constraints=constraints)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([1, 2]))
        self.assertEqual(f_x.size(), torch.Size([1]))
        self.assertTrue(x[0, 0] <= 1.0)
        self.assertTrue(x[0, 1] == 3.0)
    
    def test_slsqp_mixed(self):
        """
        Test that SLSQP with mixed parameters returns inputs as a (1, d)
        torch.Tensor and ouput as a (1, ) torch.Tensor with one constraint.
        Tests that it sticks to bounds and constraint.
        """

        # inputs
        def f(x):
            if isinstance(x, torch.Tensor):
                res = torch.sum(x**2)
            elif isinstance(x, np.ndarray):
                res = np.sum(x**2)
            return res

        bounds = torch.tensor([[-3., -5., -1.5, -5.], [2.2, 5., 2., 5.]])
        discrete = {0: [-3., 0., 2.2], 2: [-1.5, 2.]}
        constraint = {"type": "ineq", "fun": lambda x: 1.0 - x[3]}
       
        # run code
        x, f_x = single(f, method="SLSQP", bounds=bounds, discrete=discrete, constraints=constraint)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([1, 4]))
        self.assertEqual(f_x.size(), torch.Size([1]))
        self.assertTrue(x[0, 0] in discrete[0])
        self.assertTrue(x[0, 1] >= bounds[0, 1])
        self.assertTrue(x[0, 1] <= bounds[1, 1])
        self.assertTrue(x[0, 2] in discrete[2])
        self.assertTrue(x[0, 3] <= 1.0)


class TestAdam(unittest.TestCase):

    def test_adam(self):
        """
        Test that Adam returns inputs as a (1, d) torch.Tensor and ouput as a
        (1, ) torch.Tensor. Tests that it sticks to bounds.
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
        x, f_x = single(f, method="Adam", bounds=bounds)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([1, 2]))
        self.assertEqual(f_x.size(), torch.Size([1]))
        self.assertTrue(x[0, 0] >= bounds[0, 0])
        self.assertTrue(x[0, 0] <= bounds[1, 0])
        self.assertTrue(x[0, 1] >= bounds[0, 1])
        self.assertTrue(x[0, 1] <= bounds[1, 1])

    def test_adam_mixed(self):
        """
        Test that Adam with mixed parameters returns inputs as a (1, d)
        torch.Tensor and ouput as a (1, ) torch.Tensor.  Tests that it sticks
        to bounds.
        """

        # inputs
        def f(x):
            if isinstance(x, torch.Tensor):
                res = torch.sum(x**2)
            elif isinstance(x, np.ndarray):
                res = np.sum(x**2)
            return res

        bounds = torch.tensor([[-3., -5., -1.5], [2.2, 5., 2.]])
        discrete = {0: [-3., 0., 2.2], 2: [-1.5, 2.]}

        # run code
        x, f_x = single(f, method="Adam", bounds=bounds, discrete=discrete)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([1, 3]))
        self.assertEqual(f_x.size(), torch.Size([1]))
        self.assertTrue(x[0, 0] in discrete[0])
        self.assertTrue(x[0, 1] >= bounds[0, 1])
        self.assertTrue(x[0, 1] <= bounds[1, 1])
        self.assertTrue(x[0, 2] in discrete[2])


class TestJoint(unittest.TestCase):

    def test_joint_adam(self):
        """
        Test that Adam returns inputs as a (q, d) torch.Tensor and ouput as a
        (1, ) torch.Tensor for jointly computed batches.  Tests that it sticks
        to bounds.
        """

        # inputs
        n = 20
        d = 2
        q = 4
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        bounds = torch.tensor([[-5., -5.], [5., 5.]])
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32)
        
        # run code
        x, f_x = multi_joint(func=acq, method="Adam", batch_size=q, bounds=bounds, num_starts=2)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([q, 2]))
        self.assertEqual(f_x.size(), torch.Size([1]))
        self.assertTrue(torch.min(x[:, 0]) >= bounds[0, 0])
        self.assertTrue(torch.max(x[:, 0]) <= bounds[1, 0])
        self.assertTrue(torch.min(x[:, 1]) >= bounds[0, 1])
        self.assertTrue(torch.max(x[:, 1]) <= bounds[1, 1])

    def test_joint_adam_mixed(self):
        """
        Test that Adam with mixed parameters returns inputs as a (q, d)
        torch.Tensor and ouput as a (1, ) torch.Tensor for jointly computed
        batches. Tests that it sticks to bounds.
        """

        # inputs
        n = 20
        d = 3
        q = 2
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        bounds = torch.tensor([[-3., -5., -1.5], [2.2, 5., 2.]])
        discrete = {0: [-3., 0., 2.2], 2: [-1.5, 2.]}
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32)
        
        # run code
        x, f_x = multi_joint(func=acq, method="Adam", batch_size=q, bounds=bounds, discrete=discrete, num_starts=2)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([q, d]))
        self.assertEqual(f_x.size(), torch.Size([1]))
        self.assertTrue(all([x[i, 0] in discrete[0] for i in range(q)]))
        self.assertTrue(torch.min(x[:, 1]) >= bounds[0, 1])
        self.assertTrue(torch.max(x[:, 1]) <= bounds[1, 1])
        self.assertTrue(all([x[i, 2] in discrete[2] for i in range(q)]))


    def test_joint_lbfgsb(self):
        """
        Test that L-BFGS-B returns inputs as a (q, d) torch.Tensor and ouput as
        a (1, ) torch.Tensor for jointly computed batches with fixed base
        samples. Tests that it sticks to bounds.
        """

        # inputs
        n = 20
        d = 2
        q = 4
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        bounds = torch.tensor([[-5., -5.], [5., 5.]])
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32, fix_base_samples=True)
        
        # run code
        x, f_x = multi_joint(func=acq, method="L-BFGS-B", batch_size=q, bounds=bounds, num_starts=2)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([q, 2]))
        self.assertEqual(f_x.size(), torch.Size([1]))
        self.assertTrue(torch.min(x[:, 0]) >= bounds[0, 0])
        self.assertTrue(torch.max(x[:, 0]) <= bounds[1, 0])
        self.assertTrue(torch.min(x[:, 1]) >= bounds[0, 1])
        self.assertTrue(torch.max(x[:, 1]) <= bounds[1, 1])

    def test_joint_lbfgsb_mixed(self):
        """
        Test that L-BFGS-B with mixed parameters returns inputs as a (q, d)
        torch.Tensor and ouput as a (1, ) torch.Tensor for jointly computed
        batches. Tests that it sticks to bounds.
        """

        # inputs
        n = 20
        d = 3
        q = 2
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        bounds = torch.tensor([[-3., -5., -1.5], [2.2, 5., 2.]])
        discrete = {0: [-3., 0., 2.2], 2: [-1.5, 2.]}
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32, fix_base_samples=True)
        
        # run code
        x, f_x = multi_joint(func=acq, method="L-BFGS-B", batch_size=q, bounds=bounds, discrete=discrete, num_starts=2)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([q, d]))
        self.assertEqual(f_x.size(), torch.Size([1]))
        self.assertTrue(all([x[i, 0] in discrete[0] for i in range(q)]))
        self.assertTrue(torch.min(x[:, 1]) >= bounds[0, 1])
        self.assertTrue(torch.max(x[:, 1]) <= bounds[1, 1])
        self.assertTrue(all([x[i, 2] in discrete[2] for i in range(q)]))
    

class TestSequential(unittest.TestCase):

    def test_sequential_adam(self):
        """
        Test that Adam returns inputs as a (q, d) torch.Tensor and ouput as a
        (1, ) torch.Tensor for sequentially computed batches. Tests that it
        sticks to bounds.
        """

        # inputs
        n = 20
        d = 2
        q = 4
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        bounds = torch.tensor([[-5., -5.], [5., 5.]])
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32)
        
        # run code
        x, f_x = multi_sequential(func=acq, method="Adam", batch_size=q, bounds=bounds, num_starts=2)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([q, d]))
        self.assertEqual(f_x.size(), torch.Size([q]))
        self.assertTrue(torch.min(x[:, 0]) >= bounds[0, 0])
        self.assertTrue(torch.max(x[:, 0]) <= bounds[1, 0])
        self.assertTrue(torch.min(x[:, 1]) >= bounds[0, 1])
        self.assertTrue(torch.max(x[:, 1]) <= bounds[1, 1])

    def test_sequential_adam_mixed(self):
        """
        Test that Adam with mixed parameters returns inputs as a (q, d)
        torch.Tensor and ouput as a (1, ) torch.Tensor for sequentially
        computed batches with fixed base samples.  Tests that it sticks to
        bounds.
        """

        # inputs
        n = 20
        d = 3
        q = 2
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        bounds = torch.tensor([[-3., -5., -1.5], [2.2, 5., 2.]])
        discrete = {0: [-3., 0., 2.2], 2: [-1.5, 2.]}
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32)
        
        # run code
        x, f_x = multi_sequential(func=acq, method="Adam", batch_size=q, bounds=bounds, discrete=discrete, num_starts=2)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([q, d]))
        self.assertEqual(f_x.size(), torch.Size([q]))
        self.assertTrue(all([x[i, 0] in discrete[0] for i in range(q)]))
        self.assertTrue(torch.min(x[:, 1]) >= bounds[0, 1])
        self.assertTrue(torch.max(x[:, 1]) <= bounds[1, 1])
        self.assertTrue(all([x[i, 2] in discrete[2] for i in range(q)]))

    def test_sequential_lbfgsb(self):
        """
        Test that L-BFGS-B returns inputs as a (q, d) torch.Tensor and ouput as
        a (1, ) torch.Tensor for sequentially computed batches with fixed base
        samples. Tests that it sticks to bounds.
        """

        # inputs
        n = 20
        d = 2
        q = 4
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        bounds = torch.tensor([[-5., -5.], [5., 5.]])
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32, fix_base_samples=True)
        
        # run code
        x, f_x = multi_sequential(func=acq, method="L-BFGS-B", batch_size=q, bounds=bounds, num_starts=2)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([q, d]))
        self.assertEqual(f_x.size(), torch.Size([q]))
        self.assertTrue(torch.min(x[:, 0]) >= bounds[0, 0])
        self.assertTrue(torch.max(x[:, 0]) <= bounds[1, 0])
        self.assertTrue(torch.min(x[:, 1]) >= bounds[0, 1])
        self.assertTrue(torch.max(x[:, 1]) <= bounds[1, 1])

    def test_sequential_lbfgsb_mixed(self):
        """
        Test that L-BFGS-B with mixed parameters returns inputs as a (q, d)
        torch.Tensor and ouput as a (1, ) torch.Tensor for sequentially
        computed batches with fixed base samples. Tests that it sticks to
        bounds.
        """

        # inputs
        n = 20
        d = 3
        q = 2
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        bounds = torch.tensor([[-3., -5., -1.5], [2.2, 5., 2.]])
        discrete = {0: [-3., 0., 2.2], 2: [-1.5, 2.]}
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32, fix_base_samples=True)
        
        # run code
        x, f_x = multi_sequential(func=acq, method="L-BFGS-B", batch_size=q, bounds=bounds, discrete=discrete, num_starts=2)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([q, d]))
        self.assertEqual(f_x.size(), torch.Size([q]))
        self.assertTrue(all([x[i, 0] in discrete[0] for i in range(q)]))
        self.assertTrue(torch.min(x[:, 1]) >= bounds[0, 1])
        self.assertTrue(torch.max(x[:, 1]) <= bounds[1, 1])
        self.assertTrue(all([x[i, 2] in discrete[2] for i in range(q)]))

    
    def test_sequential_slsqp(self):
        """
        Test that SLSQP returns inputs as a (q, d) torch.Tensor and ouput as
        a (1, ) torch.Tensor for sequentially computed batches with fixed base
        samples and one constraint. Tests that it sticks to bounds and
        constraints.
        """

        # inputs
        n = 20
        d = 2
        q = 4
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        bounds = torch.tensor([[-5., -5.], [5., 5.]])
        constraint = {"type": "ineq", "fun": lambda x: -1.0 + x[0]}
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32, fix_base_samples=True)
        
        # run code
        x, f_x = multi_sequential(func=acq, method="SLSQP", batch_size=q, bounds=bounds, constraints=constraint, num_starts=2)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([q, d]))
        self.assertEqual(f_x.size(), torch.Size([q]))
        self.assertTrue(torch.min(x[:, 0]) >= 1.0)
        self.assertTrue(torch.min(x[:, 1]) >= bounds[0, 1])
        self.assertTrue(torch.max(x[:, 1]) <= bounds[1, 1])
    
    def test_sequential_slsqp_mixed(self):
        """
        Test that SLSQP with mixed parameters returns inputs as a (q, d)
        torch.Tensor and ouput as a (1, ) torch.Tensor for sequentially
        computed batches with fixed base samples and one constraint. Tests that
        it sticks to bounds and constraints.
        """

        # inputs
        n = 20
        d = 4
        q = 2
        X = torch.rand((n, d), dtype=torch.float64)
        y = torch.sum(X**2, axis=1)
        bounds = torch.tensor([[-3., -5., -1.5, -5.], [2.2, 5., 2., 5.]])
        constraint = {"type": "ineq", "fun": lambda x: -1.0 + x[3]}
        discrete = {0: [-3., 0., 2.2], 2: [-1.5, 2.]}
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        acq = MCUpperConfidenceBound(gp=gp, beta=1.96**2, samples=32)
        
        # run code
        x, f_x = multi_sequential(func=acq, method="SLSQP", batch_size=q, bounds=bounds, constraints=constraint, discrete=discrete, num_starts=2)

        # test
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(f_x, torch.Tensor)
        self.assertEqual(x.size(), torch.Size([q, d]))
        self.assertEqual(f_x.size(), torch.Size([q]))
        self.assertTrue(all([x[i, 0] in discrete[0] for i in range(q)]))
        self.assertTrue(torch.min(x[:, 1]) >= bounds[0, 1])
        self.assertTrue(torch.max(x[:, 1]) <= bounds[1, 1])
        self.assertTrue(all([x[i, 2] in discrete[2] for i in range(q)]))
        self.assertTrue(torch.min(x[:, 3]) >= 1.0)


if __name__ == "__main__":
    unittest.main()
