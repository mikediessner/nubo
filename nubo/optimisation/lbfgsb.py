import torch
from torch import Tensor
from scipy.optimize import minimize
from typing import Tuple, Optional, Callable, Any
from nubo.optimisation import gen_candidates


def lbfgsb(func: Callable,
           bounds: Tensor,
           num_starts: Optional[int]=10,
           num_samples: Optional[int]=100,
           **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Multi-start L-BFGS-B optimiser.

    Parameters
    ----------
    func : :obj:`Callable`
        Function to optimise.
    bounds : :obj:`torch.Tensor`
        (size 2 x d) Optimisation bounds of input space.
    num_starts : :obj:`int`, optional
        Number of start for multi-start optimisation, default is 10.
    num_samples : :obj:`int`, optional
        Number of samples from which to draw the starts, default is 100.
    **kwargs : :obj:`Any`
        Keyword argument passed to :obj:`scipy.optimize.minimize`.

    Returns
    -------
    best_result : :obj:`torch.Tensor`
        (size 1 x d) Minimiser inputs.
    best_func_result : :obj:`torch.Tensor`
        (size 1) Minimiser output.
    """

    dims = bounds.size(1)
    opt_bounds = bounds.numpy().T
    
    # generate candidates
    candidates = gen_candidates(func, bounds, num_starts, num_samples)

    # initialise objects for results
    results = torch.zeros((num_starts, dims))
    func_results = torch.zeros(num_starts)
    
    # iteratively optimise over candidates
    for i in range(num_starts):
        candidate = candidates[i]
        result = minimize(func, x0=candidate, method="L-BFGS-B", bounds=opt_bounds, **kwargs)
        results[i, :] = torch.from_numpy(result["x"].reshape(1, -1))
        func_results[i] = float(result["fun"])

    # select best candidate
    best_i = torch.argmax(func_results)
    best_result =  torch.reshape(results[best_i, :], (1, -1))
    best_func_result = func_results[best_i]

    return best_result, best_func_result
