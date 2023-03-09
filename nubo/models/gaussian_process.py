from torch import Tensor
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood


class GaussianProcess(ExactGP):
    """
    Gaussian process model with constant mean function and Matern 5/2 kernel.
    
    Attributes
    ----------
    x_train : ``torch.Tensor``
        (size n x d) Training inputs.
    y_train : ``torch.Tensor``
        (size n) Training outputs.
    likelihood : ``gpytorch.likelihoods.Likelihood``
        Likelihood.
    mean_module : ``gpytorch.means``
        Zero mean function.
    covar_module : ``gpytorch.kernels``
        Automatic relevance determination Matern 5/2 covariance kernel.
    """

    def __init__(self,
                 x_train: Tensor, 
                 y_train: Tensor,
                 likelihood: Likelihood) -> None:
        """
        Parameters
        ----------
        x_train : ``torch.Tensor``
            (size n x d) Training inputs.
        y_train : ``torch.Tensor``
            (size n) Training targets.
        likelihood : ``gpytorch.likelihoods.Likelihood``
            Likelihood.
        """

        # initialise ExactGP
        super(GaussianProcess, self).__init__(x_train, y_train, likelihood)


        # specify mean function and covariance kernel
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(nu=5/2,
                                     ard_num_dims=x_train.shape[-1])
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        """
        Compute the mean vector and covariance matrix for some test points `x`
        and returns a multivariate normal distribution.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x d) Test points.
        
        Returns
        -------
        ``gpytorch.distributions.MultivariateNormal``
            Predictice multivariate normal distribution.
        """

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return MultivariateNormal(mean_x, covar_x)
