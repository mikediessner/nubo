import torch
from torch import Tensor
import gpytorch
from gpytorch.models import GP
from .acquisition_function import AcquisitionFunction
from typing import Optional


class MCExpectedImprovement(AcquisitionFunction):
    """
    Monte Carlo Expected Improvement acquisition function.

    Attributes
    ----------
    gp : ``gpytorch.models.GP``
        Gaussian Process model.
    y_best : ``torch.Tensor``
        (size 1) Best output of training data.
    x_pending : ``torch.Tensor``
        (size n x d) Training inputs of currently pending points.
    samples : ``int``
         Number of Monte Carlo samples, default is 512.
    fix_base_samples : ``bool``
        Whether base samples used to compute Monte Carlo samples of
        acquisition function should be fixed for the optimisation step.
        If false (default) stochastic optimizer (Adam) have to be used. If
        true deterministic optimizer (L-BFGS-B, SLSQP) can be used.
    base_samples : ``NoneType`` or ``torch.Tensor``
        Base samples used to compute Monte Carlo samples drawn if
        `fix_base_samples` is true.
    dims : ``int``
        Number of input dimensions.
    """
    

    def __init__(self,
                 gp: GP,
                 y_best : Tensor,
                 x_pending: Optional[Tensor]=None,
                 samples: Optional[int]=512,
                 fix_base_samples: Optional[bool]=False)-> None:
        """
        Parameters
        ----------
        gp : ``gpytorch.models.GP``
            Gaussian Process model.
        y_best : ``torch.Tensor``
            (size 1) Best output of training data.
        x_pending : ``torch.Tensor``, optional
            (size n x d) Training inputs of currently pending points.
        samples : ``int``, optional
             Number of Monte Carlo samples, default is 512.
        fix_base_samples : ``bool``, optional
            Whether base samples used to compute Monte Carlo samples of
            acquisition function should be fixed for the optimisation step.
            If false (default) stochastic optimizer (Adam) have to be used. If
            true deterministic optimizer (L-BFGS-B, SLSQP) can be used.
        """
        
        self.gp = gp
        self.y_best = y_best
        self.x_pending = x_pending
        self.samples = samples
        self.fix_base_samples = fix_base_samples
        self.base_samples = None
        self.dims = gp.train_inputs[0].size(1)

    def eval(self, x: Tensor) -> Tensor:
        """
        Computes the (negative) Expected Improvement for some test points `x`
        by averaging Monte Carlo samples.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x d) Test points.

        Returns
        -------
        ``torch.Tensor``
            (size n) (Negative) Expected Imrpovement of `x`.
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

        # compute Expected Improvement
        ei = torch.clamp(samples - self.y_best, min=0)
        ei = ei.max(dim=1).values
        ei = ei.mean(dim=0) # average samples
        
        return -ei


class MCUpperConfidenceBound(AcquisitionFunction):
    """
    Monte Carlo Upper Confidence Bound acquisition function.

    Attributes
    ----------
    gp : ``gpytorch.models.GP``
        Gaussian Process model.
    beta : ``float``
            Trade-off parameter, default is 4.0.
    x_pending : ``torch.Tensor``
        (size n x d) Training inputs of currently pending points.
    samples : ``int``
         Number of Monte Carlo samples, default is 512.
    fix_base_samples : ``bool``
        Whether base samples used to compute Monte Carlo samples of
        acquisition function should be fixed for the optimisation step.
        If false (default) stochastic optimizer (Adam) have to be used. If
        true deterministic optimizer (L-BFGS-B, SLSQP) can be used.
    base_samples : ``NoneType`` or ``torch.Tensor``
        Base samples used to compute Monte Carlo samples drawn if
        `fix_base_samples` is true.
    dims : ``int``
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
        gp : ``gpytorch.models.GP``
            Gaussian Process model.
        beta : ``float``
                Trade-off parameter, default is 4.0.
        x_pending : ``torch.Tensor`
            (size n x d) Training inputs of currently pending points.
        samples : ``int``
            Number of Monte Carlo samples, default is 512.
        fix_base_samples : ``bool``
            Whether base samples used to compute Monte Carlo samples of
            acquisition function should be fixed for the optimisation step.
            If false (default) stochastic optimizer (Adam) have to be used. If
            true deterministic optimizer (L-BFGS-B, SLSQP) can be used.
        base_samples : ``NoneType`` or ``torch.Tensor``
            Base samples used to compute Monte Carlo samples drawn if
            `fix_base_samples` is true.
        dims : ``int``
            Number of input dimensions.
        """
        
        self.gp = gp
        self.beta = torch.tensor(beta)
        self.beta_coeff = torch.sqrt(self.beta*torch.pi/2)
        self.x_pending = x_pending
        self.samples = samples
        self.fix_base_samples = fix_base_samples
        self.base_samples = None
        self.dims = gp.train_inputs[0].size(1)

    def eval(self, x: Tensor) -> Tensor:
        """
        Computes the (negative) Upper Confidence Bound for some test points
        `x` by averaging Monte Carlo samples.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x d) Test points.

        Returns
        -------
        ``torch.Tensor``
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
