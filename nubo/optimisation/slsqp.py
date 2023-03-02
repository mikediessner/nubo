import torch
from torch import Tensor
from scipy.optimize import minimize, Bounds
from typing import Tuple, Optional, Callable, Any
from nubo.optimisation import gen_candidates


def slsqp(func: Callable,
          bounds: Tensor,
          constraints: dict,
          num_starts: Optional[int]=10,
          num_samples: Optional[int]=100,
          **kwargs: Any) -> Tuple[Tensor, float]:
    """
    Multi-start SLSQP optimisation.
    """

    dims = bounds.size(1)
    opt_bounds = Bounds(lb=bounds[0, :], ub=bounds[1, :])
    
    # generate candidates
    candidates = gen_candidates(func, bounds, num_starts, num_samples)

    # initialise objects for results
    results = torch.zeros((num_starts, dims))
    func_results = torch.zeros(num_starts)
    
    # iteratively optimise over candidates
    for i in range(num_starts):
        result = minimize(func, x0=candidates[i], method="SLSQP", bounds=opt_bounds, constraints=constraints, **kwargs)
        results[i, :] = torch.from_numpy(result["x"].reshape(1, -1))
        func_results[i] = float(result["fun"])
    
    # select best candidate
    best_i = torch.argmax(func_results)
    best_result =  torch.reshape(results[best_i, :], (1, -1))
    best_func_result = func_results[best_i]

    return best_result, best_func_result
