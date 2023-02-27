import torch
import scipy
from torch import Tensor
from typing import Any, Tuple, Optional, Callable
from scipy.optimize import OptimizeResult
from .autograd import get_fun_and_jac


def minimise(func: Callable, 
             x0: Tensor,
             method: Optional[str]="BFGS",
             bounds: Optional[Tensor]=None,
             constraints: Optional[dict]=(),
             **kwargs: Any) -> Tuple[Tensor, OptimizeResult]:

    x0 = x0.numpy()
    if bounds != None: bounds = bounds.numpy().T

    res = scipy.optimize.minimize(func, x0=x0, method=method, bounds=bounds, constraints=constraints, **kwargs)
    
    x = torch.from_numpy(res["x"].reshape(1, -1))

    if res.success is False:
        print("Optimiser did not succeed.")
        print(res)

    return x, res


def autograd_minimise(func: Callable, 
                      x0: Tensor,
                      method: Optional[str]="BFGS",
                      bounds: Optional[Tensor]=None,
                      constraints: Optional[dict]=(),
                      **kwargs: Any) -> Tuple[Tensor, OptimizeResult]:

    x0 = x0.numpy()
    if bounds != None: bounds = bounds.numpy().T

    res = scipy.optimize.minimize(get_fun_and_jac, x0=x0, args=(func), method=method, bounds=bounds, constraints=constraints, jac=True, **kwargs)

    x = torch.from_numpy(res["x"].reshape(1, -1))

    if res.success is False:
        print("Optimiser did not succeed.")
        print(res)

    return x, res
