import torch
from torch import Tensor


def standardise(y: Tensor) -> Tensor:
    """
    Standardise data by subtracting the mean and dividing by the standard
    deviation.

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
    """
    Normalise data to the unit cube [0, 1]^d.

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
    """
    Revere normalisation to the provided bounds.

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
