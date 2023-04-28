import torch
from torch import Tensor
import gpytorch
from gpytorch.models import GP
from .acquisition_function import AcquisitionFunction
from typing import Optional


class MCExpectedImprovement(AcquisitionFunction):
    r"""
    Monte Carlo expected improvement acquisition function:

    .. math::
        \alpha_{EI}^{MC} (\boldsymbol X_*) = \max \left(ReLU(\mu_n(\boldsymbol X_*) + \boldsymbol L \boldsymbol z - y^{best}) \right),

    where :math:`\mu_n(\cdot)` is the mean of the predictive distribution of
    the Gaussian process, :math:`\boldsymbol L` is the lower triangular matrix
    of the Cholesky decomposition of the covariance matrix 
    :math:`\boldsymbol L \boldsymbol L^T = K(\boldsymbol X_n, \boldsymbol X_n)`,
    :math:`\boldsymbol z` are samples from the standard normal distribution
    :math:`\mathcal{N} (0, 1)`, :math:`y^{best}` is the current best
    observation, and :math:`ReLU (\cdot)` is the rectified linear unit function
    that zeros all values below 0 and leaves the rest as is.
    
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
        If false (default) stochastic optimizer (Adam) has to be used. If
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
            If false (default) stochastic optimizer (Adam) has to be used. If
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
        Computes the (negative) expected improvement for some test point `x` by
        averaging Monte Carlo samples.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size 1 x d) Test point.
        Returns
        -------
        ``torch.Tensor``
            (size 1) (Negative) expected improvement of `x`.
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
        ei = ei.mean(dim=0, keepdim=True) # average samples
        
        return -ei

class MCUpperConfidenceBound(AcquisitionFunction):
    r"""
    Monte Carlo upper confidence bound acquisition function:

    .. math::
        \alpha_{UCB}^{MC} (\boldsymbol X_*) = \max \left(\mu_n(\boldsymbol X_*) + \sqrt{\frac{\beta \pi}{2}} \lvert \boldsymbol L \boldsymbol z \rvert \right),

    where :math:`\mu_n(\cdot)` is the mean of the predictive distribution of
    the Gaussian process, :math:`\boldsymbol L` is the lower triangular matrix
    of the Cholesky decomposition of the covariance matrix 
    :math:`\boldsymbol L \boldsymbol L^T = K(\boldsymbol X_n, \boldsymbol X_n)`,
    :math:`\boldsymbol z` are samples from the standard normal distribution
    :math:`\mathcal{N} (0, 1)`, and :math:`\beta` is the trade-off parameter.

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
        If false (default) stochastic optimizer (Adam) has to be used. If
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
            If false (default) stochastic optimizer (Adam) has to be used. If
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
        Computes the (negative) upper confidence bound for some test point `x`
        by averaging Monte Carlo samples.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size 1 x d) Test point.

        Returns
        -------
        ``torch.Tensor``
            (size 1) (Negative) upper confidence bound of `x`.
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
        ucb = ucb.mean(dim=0, keepdim=True) # average samples

        return -ucb
