import torch
from torch import Tensor


def standardise(y: Tensor) -> Tensor:

    std, mean = torch.std_mean(y)

    out = (y - mean) / std

    return out


def normalise(x: Tensor, bounds: Tensor) -> Tensor:
    """
    Normalise data to the range [0, 1].
    Bounds: 2xd array where the first row provides the lower bounds and the second row the upper bounds for each dimension.
    """

    lower = bounds[0, :]
    upper = bounds[1, :]

    return (x - lower)/(upper - lower)


def unnormalise(x: Tensor, bounds: Tensor) -> Tensor:
    """
    Revere normalisation to the bounds.
    Bounds: 2xd array where the first row provides the lower bounds and the second row the upper bounds for each dimension.
    """

    lower = bounds[0, :]
    upper = bounds[1, :]

    return x * (upper - lower) + lower