import unittest
import torch

from nubo.models import *
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal


class TestGP(unittest.TestCase):

    def test_GP(self):
        """
        Test that model is an exact Gaussian process and has constant mean
        function and a Matern kernel with outputscale
        """

        # inputs
        n = 20
        d = 4
        X = torch.rand((n, d))
        y = torch.rand(n)
        likelihood = GaussianLikelihood()

        # run code
        gp = GaussianProcess(X, y, likelihood=likelihood)
        
        # test
        self.assertIsInstance(gp, ExactGP)
        self.assertIsInstance(gp.mean_module, ConstantMean)
        self.assertIsInstance(gp.covar_module, ScaleKernel)
        self.assertIsInstance(gp.covar_module.base_kernel, MaternKernel)
    
    def test_GP_prediction(self):
        """
        Test that the Gaussian process returns a multivariate normal
        distribution when predicting.
        """

        # inputs
        n = 20
        d = 4
        X = torch.rand((n, d))
        y = torch.rand(n)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)

        # run code
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        gp.eval()
        pred = gp(torch.rand((1, d)))

        # test
        self.assertIsInstance(pred, MultivariateNormal)
    
    def test_GP_hyperparameter_estimation(self):
        """
        Test that constant mean, outputscale, length-scales, and noise is
        estimated when using `fit_gp()`.
        """

        # inputs
        n = 20
        d = 4
        X = torch.rand((n, d))
        y = torch.rand(n)
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(X, y, likelihood=likelihood)
        prior_mean = gp.mean_module.constant.clone()
        prior_outputscale = gp.covar_module.outputscale.clone()
        prior_lengthscales = gp.covar_module.base_kernel.lengthscale.clone()
        prior_noise = likelihood.noise.clone()

        # run code
        fit_gp(X, y, gp=gp, likelihood=likelihood)
        post_mean = gp.mean_module.constant
        post_outputscale = gp.covar_module.outputscale
        post_lengthscales = gp.covar_module.base_kernel.lengthscale
        post_noise = likelihood.noise

        # test
        self.assertNotEqual(prior_mean.item(), post_mean.item())
        self.assertNotEqual(prior_outputscale.item(), post_outputscale.item())
        for i in range(d):
            self.assertNotEqual(prior_lengthscales[0][i].item(), post_lengthscales[0][i].item())
        self.assertNotEqual(prior_noise.item(), post_noise.item())


if __name__ == "__main__":
    unittest.main()
