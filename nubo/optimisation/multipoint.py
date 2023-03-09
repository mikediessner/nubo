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
    Sequential greedy optimisation loop for Monte Carlo acquisition functions.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    method : ``str``
        One of "L-BFGS-B", "SLSQP", or "Adam".
    batch_size : ``int``
        Number of points to return.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    constraints : ``dict`` or ``list`` of ``dict``, optional
        Optimisation constraints.
    lr : ``float``, optional
        Learning rate of ``torch.optim.Adam`` algorithm, default is 0.1.
    steps : ``int``, optional
        Optimisation steps of ``torch.optim.Adam`` algorithm, default is 200.
    num_starts : ``int``, optional
        Number of start for multi-start optimisation, default is 10.
    num_samples : ``int``, optional
        Number of samples from which to draw the starts, default is 100.
    **kwargs : ``Any``
        Keyword argument passed to ``torch.optim.Adam`` or
        ``scipy.optimize.minimze``.

    Returns
    -------
    batch_result : ``torch.Tensor``
        (size `batch_size` x d) Batch inputs.
    batch_func_result : ``torch.Tensor``
        (size `batch_size`) Batch outputs.
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

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    method : ``str``
        One of "L-BFGS-B", "SLSQP", or "Adam".
    batch_size : ``int``
        Number of points to return.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    lr : ``float``, optional
        Learning rate of ``torch.optim.Adam`` algorithm, default is 0.1.
    steps : ``int``, optional
        Optimisation steps of ``torch.optim.Adam`` algorithm, default is 200.
    num_starts : ``int``, optional
        Number of start for multi-start optimisation, default is 10.
    num_samples : ``int``, optional
        Number of samples from which to draw the starts, default is 100.
    **kwargs : ``Any``
        Keyword argument passed to ``torch.optim.Adam`` or
        ``scipy.optimize.minimze``.

    Returns
    -------
    batch_result : ``torch.Tensor``
        (sizq `batch_size` x d) Batch inputs.
    batch_func_result : ``torch.Tensor``
        (size `batch_size`) Batch outputs.
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