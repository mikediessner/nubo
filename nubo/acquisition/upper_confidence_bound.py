import torch
from torch import Tensor
import gpytorch
from gpytorch.models import GP
from .acquisition_function import AcquisitionFunction
from typing import Optional


class UpperConfidenceBound(AcquisitionFunction):

    def __init__(self, 
                 gp: GP,
                 beta: Optional[float]=1.0) -> None:

        self.gp = gp
        self.beta = beta

    def eval(self, x: Tensor) -> Tensor:

        # set Gaussian Process to eval mode
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
                 fix_base_samples: Optional[bool]=False)-> None:
        
        self.samples = samples              # Monte Carlo samples
        self.gp = gp                        # surrogate model
        self.beta = torch.tensor(beta)      # UCB parameter
        self.beta_coeff = torch.sqrt(self.beta*torch.pi/2)
        self.x_pending = x_pending
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

        # compute Upper Confidence Bound
        ucb = mean + self.beta_coeff * torch.abs(samples - mean)
        ucb = ucb.max(dim=1).values
        ucb = ucb.mean(dim=0) # average samples

        return -ucb
