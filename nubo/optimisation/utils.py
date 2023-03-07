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

    Parameters
    ----------
    func : :obj:`Callable`
        Function to optimise.
    bounds : :obj:`torch.Tensor`
        (size 2 x d) Optimisation bounds of input space.
    num_candidates : :obj:`int`
        Number of candidates.
    num_samples : :obj:`int`
        Number of samples from which to draw the starts.

    Returns
    -------
    :obj:`torch.Tensor`
        (size `num_candidates` x d) Candidates.
    """

    dims = bounds.size(1)

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
