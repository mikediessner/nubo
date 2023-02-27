from torch import Tensor
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior


class MultiLevelGP:

    def __init__(self,
                 x_train: Tensor, 
                 y_train: Tensor,
                 likelihood: Likelihood) -> None:
        pass

    def forward(self, x: Tensor):
        pass