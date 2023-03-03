import torch
from torch import Tensor
from torch.distributions.normal import Normal
import gpytorch
from gpytorch.models import GP
from .acquisition_function import AcquisitionFunction
from typing import Optional


class ExpectedImprovement(AcquisitionFunction):

    def __init__(self,
                 gp: GP,
                 y_best: Tensor) -> None:

        self.gp = gp
        self.y_best = y_best
        
    def eval(self, x: Tensor) -> Tensor:

        # set Gaussian Process to eval mode
        self.gp.eval()

        # make predictions
        pred = self.gp(x)

        mean = pred.mean
        variance = pred.variance.clamp_min(1e-10)
        std = torch.sqrt(variance)

        # compute Expected Improvement
        norm = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        z = (mean - self.y_best)/std
        ei = (mean - self.y_best)*norm.cdf(z) + std*torch.exp(norm.log_prob(z))

        return -ei


class MCExpectedImprovement(AcquisitionFunction):

    def __init__(self,
                 gp: GP,
                 y_best : Tensor,
                 x_pending: Optional[Tensor]=None,
                 samples: Optional[int]=512,
                 fix_base_samples: Optional[bool]=False)-> None:
        
        self.gp = gp                        # surrogate model
        self.y_best = y_best                # EI target
        self.x_pending = x_pending
        self.samples = samples              # Monte Carlo samples
        self.fix_base_samples = fix_base_samples
        self.base_samples = None
        self.dims = gp.train_inputs[0].size(1)

    def eval(self, x: Tensor) -> Tensor:
        
        # reshape tensor to (batch_size x dims)
        x = torch.reshape(x, (-1, self.dims))

        # add pending points
        if isinstance(self.x_pending, Tensor):
            x = torch.cat([x, self.x_pending], dim=0)

        # set Gaussian Process to eval mode
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
        ei = ei.max(dim=1).values
        ei = ei.mean(dim=0) # average samples
        
        return -ei
