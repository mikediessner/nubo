import torch
from torch import Tensor
from typing import Optional
from .latin_hypercube import LatinHypercubeSampling
from .transform import unnormalise


def gen_inputs(num_points: int,
               num_dims: int,
               bounds: Optional[Tensor]=None,
               num_lhd: Optional[int]=1000) -> Tensor:
    """
    Generate data inputs from a maximin Latin hypercube design.

    Parameters
    ----------
    num_points : ``int``
        Number of points.
    num_dims : ``int``
        Number of input dimensions.
    bounds : ``torch.Tensor``, optional
        (size 2 x `num_dims`) Bounds of input space, default is none. If none,
        bounds are a [0, 1]^`num_dims`.
    num_lhd : ``int``, optional
        Number of Latin hypercube designs to consider, default is 1000.

    Returns
    -------
    ``torch.Tensor``
        (size `num_points` x `num_dims`) Input data.
    """

    if bounds == None:
        bounds = torch.Tensor([[0.]*num_dims, [1.]*num_dims])

    lhs = LatinHypercubeSampling(dims=num_dims)
    points = lhs.maximin(points=num_points, samples=num_lhd)
    points = unnormalise(points, bounds=bounds)

    return points
