import torch
from torch import Tensor


def standardise(y: Tensor) -> Tensor:
    r"""
    Standardise data by subtracting the mean and dividing by the standard
    deviation:

    .. math::
        \hat{\boldsymbol y} = \frac{\boldsymbol y - \mu}{\sigma},

    where :math:`\mu` is the mean and :math:`\sigma` is the standard deviation.

    Parameters
    ----------
    y : ``torch.Tensor``
        (size n) Data.

    Returns
    -------
    ``torch.Tensor``
        (size n) Standardised data.
    """

    std, mean = torch.std_mean(y)

    out = (y - mean) / std

    return out


def normalise(x: Tensor, bounds: Tensor) -> Tensor:
    r"""
    Normalise data to the unit cube :math:`[0, 1]^d`.

    .. math::
        \hat{\boldsymbol x} = \frac{\boldsymbol x - lb}{ub - lb},
    
    where :math:`lb` are the lower bounds and :math:`ub` are the upper bounds.

    Parameters
    ----------
    x : ``torch.Tensor``
        (size n x d) Data.
    bounds : ``torch.Tensor``
        (size 2 x d) Bounds of input space.

    Returns
    -------
    ``torch.Tensor``
        Normalised data.
    """

    lower = bounds[0, :]
    upper = bounds[1, :]

    return (x - lower)/(upper - lower)


def unnormalise(x: Tensor, bounds: Tensor) -> Tensor:
    r"""
    Revere normalisation to the provided bounds.
    
    .. math::
        \boldsymbol x = \hat{\boldsymbol x} (ub - lb) + lb,
    
    where :math:`lb` are the lower bounds and :math:`ub` are the upper bounds.

    Parameters
    ----------
    x : ``torch.Tensor``
        (size n x d) Normalised data.
    bounds : ``torch.Tensor``
        (size 2 x d) Bounds of input space.

    Returns
    -------
    ``torch.Tensor``
        Data scaled to `bounds`.
    """

    lower = bounds[0, :]
    upper = bounds[1, :]

    return x * (upper - lower) + lower
