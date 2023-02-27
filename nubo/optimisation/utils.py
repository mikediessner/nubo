import torch
from torch import Tensor
from nubo.utils import LatinHypercubeSampling, unnormalise
from typing import Callable


def gen_candidates(func: Callable,
                   bounds: Tensor,
                   num_candidates: int,
                   num_samples: int) -> Tensor:
    """
    Generate candidates for multi-start optimisation
    """

    dims = bounds.dims

    # generate samples
    lhs = LatinHypercubeSampling(dims)
    samples = lhs.maximin(num_samples)
    samples = unnormalise(samples, bounds=bounds)

    # evaluate samples
    samples_res = torch.zeros(num_samples)
    for n in range(num_samples):
        samples_res[n] = func(samples[n, :].reshape(1, -1))

    # select best candidates
    _, best_i = torch.topk(-samples_res, num_candidates)
    candidates = samples[best_i]
    
    return candidates
