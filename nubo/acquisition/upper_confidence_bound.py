import torch
from torch import Tensor
import gpytorch
from gpytorch.models import GP
from .acquisition_function import AcquisitionFunction
from typing import Optional


class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper Confidence Bound acquisition function.

    Attributes
    ----------
    gp : :obj:`gpytorch.models.GP`
        Gaussian Process model.
    beta : :obj:`float`
        Trade-off parameter, default is 4.0.
    """

    def __init__(self, 
                 gp: GP,
                 beta: Optional[float]=4.0) -> None:
        """
        Parameters
        ----------
        gp : :obj:`gpytorch.models.GP`
            Gaussian Process model.
        beta : :obj:`float`, optional
            Trade-off parameter, default is 4.0.
        """

        self.gp = gp
        self.beta = beta

    def eval(self, x: Tensor) -> Tensor:
        """
        Computes the (negative) Upper Confidence Bound for some test points
        `x` analytically.

        Parameters
        ----------
        x : :obj:`torch.Tensor`
            (size n x d) Test points.

        Returns
        -------
        :obj:`torch.Tensor`
            (size n) (Negative) Upper Confidence Bound of `x`.
        """
        
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
    """
    Monte Carlo Upper Confidence Bound acquisition function.

    Attributes
    ----------
    gp : :obj:`gpytorch.models.GP`
        Gaussian Process model.
    beta : :obj:`float`
            Trade-off parameter, default is 4.0.
    x_pending : :obj:`torch.Tensor`
        (size n x d) Training inputs of currently pending points.
    samples : :obj:`int`
         Number of Monte Carlo samples, default is 512.
    fix_base_samples : :obj:`bool`
        Whether base samples used to compute Monte Carlo samples of
        acquisition function should be fixed for the optimisation step.
        If false (default) stochastic optimizer (Adam) have to be used. If
        true deterministic optimizer (L-BFGS-B, SLSQP) can be used.
    base_samples : :obj:`NoneType` or :obj:`torch.Tensor`
        Base samples used to compute Monte Carlo samples drawn if
        `fix_base_samples` is true.
    dims : :obj:`int`
        Number of input dimensions.
    """

    def __init__(self,
                 gp: GP,
                 beta: Optional[float]=4.0,
                 x_pending: Optional[Tensor]=None,
                 samples: Optional[int]=512,
                 fix_base_samples: Optional[bool]=False)-> None:
        """
        Parameters
        ----------
        gp : :obj:`gpytorch.models.GP`
            Gaussian Process model.
        beta : :obj:`float`
                Trade-off parameter, default is 4.0.
        x_pending : :obj:`torch.Tensor`
            (size n x d) Training inputs of currently pending points.
        samples : :obj:`int`
            Number of Monte Carlo samples, default is 512.
        fix_base_samples : :obj:`bool`
            Whether base samples used to compute Monte Carlo samples of
            acquisition function should be fixed for the optimisation step.
            If false (default) stochastic optimizer (Adam) have to be used. If
            true deterministic optimizer (L-BFGS-B, SLSQP) can be used.
        base_samples : :obj:`NoneType` or :obj:`torch.Tensor`
            Base samples used to compute Monte Carlo samples drawn if
            `fix_base_samples` is true.
        dims : :obj:`int`
            Number of input dimensions.
        """
        
        self.gp = gp                        # surrogate model
        self.beta = torch.tensor(beta)      # UCB parameter
        self.beta_coeff = torch.sqrt(self.beta*torch.pi/2)
        self.x_pending = x_pending
        self.samples = samples              # Monte Carlo samples
        self.fix_base_samples = fix_base_samples
        self.base_samples = None
        self.dims = gp.train_inputs[0].size(1)

    def eval(self, x: Tensor) -> Tensor:
        """
        Computes the (negative) Upper Confidence Bound for some test points
        `x` by averaging Monte Carlo samples.

        Parameters
        ----------
        x : :obj:`torch.Tensor`
            (size n x d) Test points.

        Returns
        -------
        :obj:`torch.Tensor`
            (size n) (Negative) Upper Confidence Bound of `x`.
        """

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
