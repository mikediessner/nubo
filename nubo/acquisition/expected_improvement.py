import torch
from torch import Tensor
from numpy import ndarray
import gpytorch
from gpytorch.models import GP
from .acquisition_function import AcquisitionFunction
from typing import Optional


class ExpectedImprovement(AcquisitionFunction):

    def __init__(self):
        NotImplementedError("Analytical EI hasn't been implemented yet.")


class MCExpectedImprovement(AcquisitionFunction):

    def __init__(self,
                 samples: int,
                 gp: GP,
                 y_best : Tensor,
                 x_pending: Optional[Tensor]=None,
                 fix_base_samples: Optional[bool]=True)-> None:
        
        self.samples = samples              # Monte Carlo samples
        self.gp = gp                        # surrogate model
        self.y_best = y_best                # EI target
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

        # compute Expected Improvement
        ei = torch.clamp(samples - self.y_best, min=0)
        ei = ei.max(dim=1)[0].mean(dim=0) # average samples

        return -ei # optimise only w.r.t. newest point


class JointMCExpectedImprovement(AcquisitionFunction):

    def __init__(self,
                 samples: int,
                 points: int,
                 gp: GP,
                 y_best : Tensor,
                 x_pending: Optional[Tensor]=None,
                 fix_base_samples: Optional[bool]=True)-> None:
        
        self.samples = samples              # Monte Carlo samples
        self.points = points
        self.gp = gp                        # surrogate model
        self.y_best = y_best                # EI target
        self.x_pending = x_pending
        self.fix_base_samples = fix_base_samples
        self.base_samples = None

    def eval(self, x: Tensor) -> ndarray:

        x = torch.reshape(x, (self.points, -1))
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
            self.base_samples = mvn.get_base_samples(torch.Size([self.points, self.samples]))
        samples = mvn.rsample(torch.Size([self.points, self.samples]), base_samples=self.base_samples).double()

        # compute Expected Improvement
        ei = torch.clamp(samples - self.y_best, min=0)
        ei = ei.mean(dim=0) # average samples

        return -torch.sum(ei)