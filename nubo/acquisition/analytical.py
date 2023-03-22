import torch
from torch import Tensor
from torch.distributions.normal import Normal
from gpytorch.models import GP
from .acquisition_function import AcquisitionFunction
from typing import Optional


class ExpectedImprovement(AcquisitionFunction):
    r"""
    Expected Improvement acquisition function:

    .. math::
        \alpha_{EI} (\boldsymbol X_*) = \left(\mu_n(\boldsymbol X_*) - y^{best} \right) \Phi(z) + \sigma_n(\boldsymbol X_*) \phi(z),

    where :math:`z = \frac{\mu_n(\boldsymbol X_*) - y^{best}}{\sigma_n(\boldsymbol X_*)}`,
    :math:`\mu_n(\cdot)` and :math:`\sigma_n(\cdot)` are the mean and the
    standard deviation of the posterior distribution of the Gaussian process,
    :math:`y^{best}` is the current best observation, and :math:`\Phi (\cdot)`
    and :math:`\phi  (\cdot)` are the cumulative distribution function and the
    probability density function of the standard normal distribution.

    Attributes
    ----------
    gp : ``gpytorch.models.GP``
        Gaussian Process model.
    y_best : ``torch.Tensor``
        (size 1) Best output of training data.
    """

    def __init__(self,
                 gp: GP,
                 y_best: Tensor) -> None:
        """
        Parameters
        ----------
        gp : ``gpytorch.models.GP``
            Gaussian Process model.
        y_best : ``torch.Tensor``
            (size 1) Best output of training data.
        """

        self.gp = gp
        self.y_best = y_best
        
    def eval(self, x: Tensor) -> Tensor:
        """
        Computes the (negative) Expected Improvement for some test points `x`
        analytically.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x d) Test points.

        Returns
        -------
        ``torch.Tensor``
            (size n) (Negative) Expected Imrpovement of `x`.
        """

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


class UpperConfidenceBound(AcquisitionFunction):
    r"""
    Upper Confidence Bound acquisition function:

    .. math::
        \alpha_{UCB} (\boldsymbol X_*) = \mu_n(\boldsymbol X_*) + \sqrt{\beta} \sigma_n(\boldsymbol X_*),

    where :math:`\beta` is a pre-defined trade-off parameter, and
    :math:`\mu_n(\cdot)` and :math:`\sigma_n(\cdot)` are the mean and the
    standard deviation of the posterior distribution of the Gaussian process.

    Attributes
    ----------
    gp : ``gpytorch.models.GP``
        Gaussian Process model.
    beta : ``float``
        Trade-off parameter, default is 4.0.
    """

    def __init__(self, 
                 gp: GP,
                 beta: Optional[float]=4.0) -> None:
        """
        Parameters
        ----------
        gp : ``gpytorch.models.GP``
            Gaussian Process model.
        beta : ``float``, optional
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
        x : ``torch.Tensor``
            (size n x d) Test points.

        Returns
        -------
        ``torch.Tensor``
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
