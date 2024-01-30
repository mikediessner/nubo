import torch
from torch import Tensor
from nubo.utils import LatinHypercubeSampling, unnormalise
from typing import Callable, Optional, Tuple


def gen_candidates(func: Callable,
                   bounds: Tensor,
                   num_candidates: int,
                   num_samples: int,
                   args: Optional[Tuple]=()) -> Tensor:
    """
    Generate candidates for multi-start optimisation using a maximin Latin 
    hypercube design or a uniform distribution for one candidate point.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    num_candidates : ``int``
        Number of candidates.
    num_samples : ``int``
        Number of samples from which to draw the starts.
    args : ``Tuple``, optional
        Arguments for function to maximise in order.

    Returns
    -------
    ``torch.Tensor``
        (size `num_candidates` x d) Candidates.
    """

    dims = bounds.size(1)

    # generate samples
    if num_samples == 1:
        samples = torch.rand((1, dims))
    else:
        lhs = LatinHypercubeSampling(dims)
        samples = lhs.random(num_samples)

    samples = unnormalise(samples, bounds=bounds)

    # evaluate samples
    samples_res = torch.zeros(num_samples)
    for n in range(num_samples):
        samples_res[n] = func(samples[n, :].reshape(1, -1), *args)

    # select best candidates (smallest output)
    _, best_i = torch.topk(samples_res, num_candidates, largest=False)
    candidates = samples[best_i]
    
    return candidates
