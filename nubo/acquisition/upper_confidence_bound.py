import torch
from torch import Tensor
from numpy import ndarray
import gpytorch
from gpytorch.models import GP
from .acquisition_function import AcquisitionFunction
from typing import Optional


class UpperConfidenceBound(AcquisitionFunction):

    def __init__(self, 
                 gp: GP,
                 beta: Optional[float]=1.0) -> None:
        self.beta = beta
        self.gp = gp

    def eval(self, x: Tensor) -> ndarray:
        
        # set Gaussian process to eval mode
        self.gp.eval()

        # make predictions
        pred = self.gp(x)

        mean = pred.mean
        variance = pred.variance.clamp_min(1e-10)
        std = torch.sqrt(variance)

        # compute Upper Confidence Bound
        ucb = mean + torch.sqrt(Tensor([self.beta]))*std

        return -ucb


class MCUpperConfidenceBound(AcquisitionFunction):

    def __init__(self,
                 samples: int,
                 gp: GP,
                 beta: Optional[float]=1.0,
                 x_pending: Optional[Tensor]=None,
                 fix_base_samples: Optional[bool]=True)-> None:
        
        self.samples = samples              # Monte Carlo samples
        self.gp = gp                        # surrogate model
        self.beta = torch.tensor(beta)      # UCB parameter
        self.beta_coeff = torch.sqrt(self.beta*torch.pi/2)
        self.x_pending = x_pending
        self.fix_base_samples = fix_base_samples
        self.base_samples = None

    def eval(self, x: Tensor) -> ndarray:

        # add pending points
        if isinstance(self.x_pending, Tensor):
            x = torch.cat([x, self.x_pending], dim=0)

        # set Gaussian process to eval mode
        self.gp.eval()

        # get predictive distribution
        pred = self.gp(x)
        mean = pred.mean
        covariance = pred.lazy_covariance_matrix
        
        # get samples from Multivariate Normal
        mvn = gpytorch.distributions.MultivariateNormal(mean, covariance)
        if self.base_samples == None and self.fix_base_samples == True:
            self.base_samples = mvn.get_base_samples(torch.Size([self.samples]))
        samples = mvn.rsample(torch.Size([self.samples]), base_samples=self.base_samples).double()

        # compute Upper Confidence Bound
        ucb = mean + self.beta_coeff * torch.abs(samples - mean)
        ucb = ucb.mean(dim=0) # average samples

        return -ucb[0] # optimise only w.r.t. newest point