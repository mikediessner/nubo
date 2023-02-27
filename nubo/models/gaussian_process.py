from torch import Tensor
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior


class GaussianProcess(ExactGP):
    """
    Gaussian process model.
    """

    def __init__(self,
                 x_train: Tensor, 
                 y_train: Tensor,
                 likelihood: Likelihood) -> None:
        """
        Parameters
        ----------
        x_train : torch.Tensor
            n x d tensor containing the training inputs.
        y_train : torch.Tensor
            n tesnor containing the training outputs.
        """

        # initialise ExactGP
        super(GaussianProcess, self).__init__(x_train, y_train, likelihood)


        # specify mean function and covariance kernel
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(nu=5/2,
                                     ard_num_dims=x_train.shape[-1],
                                     lengthscale_prior=GammaPrior(3.0, 4.0)),
            outputscale_prior=GammaPrior(2.0, 0.15)
            )

    def forward(self, x: Tensor) -> MultivariateNormal:
        """
        Compute the mean and covariance for point x and returns a multivariate
        Normal distribution.
        """

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        self.mean_x = mean_x
        self.covar_x = covar_x

        return MultivariateNormal(mean_x, covar_x)


class BayesianGaussianProcess(ApproximateGP):

    def __init__(self):
        pass

    def forward(self, x):
        pass

    def guide(self, x, y):
        pass

    def model(self, x, y):
        pass