import torch
from torch import Tensor
from typing import Tuple, Optional, Callable, Any
from nubo.optimisation import lbfgsb, slsqp, adam


def sequential(func: Callable, 
               method: str, 
               batch_size: int, 
               bounds: Tensor,
               constraints: Optional[dict]=None,
               lr: Optional[float]=0.1,
               steps: Optional[int]=100, 
               num_starts: Optional[int]=10,
               num_samples: Optional[int]=100, 
               **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Sequential optimisation loop for Monte Carlo acquisition functions.
    """

    dims = bounds.size(1)

    # initialise tensors for results
    batch_result = torch.zeros((0, dims))
    batch_func_result = torch.zeros(batch_size)

    # sequential optimisation loop
    for i in range(batch_size):
        if method == "L-BFGS-B":
            x_new, f_x_new = lbfgsb(func, bounds, num_starts, num_samples **kwargs)
        elif method == "SLSQP":
            x_new, f_x_new = slsqp(func, bounds, constraints, num_starts, num_samples, **kwargs)
        elif method == "Adam":
            x_new, f_x_new = adam(func, bounds, lr, steps, num_starts, num_samples, **kwargs)
        else:
            raise NotImplementedError("Method must be one of L-BFGS-B, SLSQP or Adam.")
        
        # add new point to results
        batch_result = torch.cat([batch_result, x_new], dim=0)
        batch_func_result[i] = float(f_x_new)

        # add new point to pending points
        if isinstance(func.x_pending, Tensor):
            func.x_pending = torch.cat([func.x_pending, x_new], dim=0)
        else:
            func.x_pending = x_new

    return batch_result, batch_func_result


def joint(func: Callable,
          method: str,
          batch_size: int,
          bounds: Tensor,
          lr: Optional[float]=0.1,
          steps: Optional[int]=100,
          num_starts: Optional[int]=10,
          num_samples: Optional[int]=100,
          **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Joint optimisation loop for Monte Carlo acquisition functions.
    """

    # extend bounds to full batch
    full_bounds = torch.tile(bounds, (1, batch_size))

    # joint optimisation loop
    for i in range(batch_size):
        if method == "L-BFGS-B":
            batch_result, batch_func_result = lbfgsb(func, full_bounds, num_starts, num_samples, **kwargs)
        elif method == "Adam":
            batch_result, batch_func_result = adam(func, full_bounds, lr, steps, num_starts, num_samples, **kwargs)
        else:
            raise NotImplementedError("Method must be one of L-BFGS-B or Adam.")

    batch_result = torch.reshape(batch_result, (batch_size, -1))

    return batch_result, batch_func_result