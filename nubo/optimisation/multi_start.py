import torch
from torch import Tensor
from nubo.optimisation import minimise, autograd_minimise
from nubo.utils import LatinHypercubeSampling, unnormalise
from typing import Optional, Tuple, Callable, Any
from scipy.optimize import OptimizeResult


def gen_candidates(func: Callable,
                   dims: int,
                   bounds: Tensor,
                   num_candidates: int,
                   num_samples: int) -> Tensor:
    """
    Generate candidates for multi-start optimisation
    """

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


def multi_start(func: Callable,
                candidates: Tensor,
                method: Optional[str]="BFGS",
                bounds: Optional[Tensor]=None,
                constraints: Optional[dict]=(),                         
                **kwargs: Any) -> Tuple[Tensor, OptimizeResult]:
    """
    Multi-start optimisation.
    """
    
    n = candidates.size(0)

    # initialise objects for results
    opt_res = []
    fun_res = torch.zeros(n)

    # iterate over candidates
    for i in range(n):
        candidate = candidates[i]
        x, res = minimise(func, x0=candidate, method=method, bounds=bounds, constraints=constraints, **kwargs)
        opt_res.append(res)
        fun_res[i] = float(res["fun"])
    
    # select best start
    best_i = torch.argmax(fun_res)
    best_res = opt_res[best_i]
    best_x =  torch.reshape(torch.from_numpy(best_res["x"]), (1, -1))

    return best_x, best_res


def autograd_multi_start(func: Callable,
                         candidates: Tensor,
                         method: Optional[str]="BFGS",
                         bounds: Optional[Tensor]=None,
                         constraints: Optional[dict]=(),                         
                         **kwargs: Any) -> Tuple[Tensor, OptimizeResult]:
    """
    Multi-start optimisation.
    """
    
    n = candidates.size(0)

    # initialise objects for results
    opt_res = []
    fun_res = torch.zeros(n)

    # iterate over candidates
    for i in range(n):
        candidate = candidates[i]
        x, res = autograd_minimise(func=func, x0=candidate, method=method, bounds=bounds, constraints=constraints, **kwargs)
        opt_res.append(res)
        fun_res[i] = float(res["fun"])
    
    # select best start
    best_i = torch.argmax(fun_res)
    best_res = opt_res[best_i]
    best_x =  torch.reshape(torch.from_numpy(best_res["x"]), (1, -1))

    return best_x, best_res
