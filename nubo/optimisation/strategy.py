import torch
from torch import Tensor
from typing import Tuple, Optional, Callable, Any
from nubo.optimisation import lbfgsb, slsqp, adam, mixed


def single(func: Callable, 
           method: str, 
           bounds: Tensor,
           constraints: Optional[dict | list]=None,
           discrete: Optional[dict]=None,
           lr: Optional[float]=0.1,
           steps: Optional[int]=100, 
           num_starts: Optional[int]=10,
           num_samples: Optional[int]=100, 
           **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Single-point optimisation.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    method : ``str``
        One of "L-BFGS-B" or "Adam".
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    discrete : ``dict``
        Possible values for all discrete inputs in the shape {dim1: [values1],
        dim2: [values2], etc.}, e.g. {0: [1., 2., 3.], 3: [-0.1, -0.2, 100.]}.
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
    best_result : ``torch.Tensor``
        (size 1 x d) Minimiser inputs.
    best_func_result : ``torch.Tensor``
        (size 1) Minimiser output.
    """

    if method not in ["L-BFGS-B", "SLSQP", "Adam"]:
        raise NotImplementedError("Method must be one of L-BFGS-B, SLSQP or Adam.")

    # continuous parameter space
    if not isinstance(discrete, dict):
        if method == "L-BFGS-B":
            result, func_result = lbfgsb(func, bounds, num_starts, num_samples, **kwargs)
        elif method == "SLSQP":
            result, func_result = slsqp(func, bounds, constraints, num_starts, num_samples, **kwargs)
        elif method == "Adam":
            result, func_result = adam(func, bounds, lr, steps, num_starts, num_samples, **kwargs)
    
    # mixed parameter space
    else:
        result, func_result = mixed(func, method, bounds, discrete, constraints, lr, steps, num_starts, num_samples, **kwargs)

    return result, func_result


def multi_joint(func: Callable,
                method: str,
                batch_size: int,
                bounds: Tensor,
                discrete: Optional[dict]=None,
                lr: Optional[float]=0.1,
                steps: Optional[int]=100,
                num_starts: Optional[int]=10,
                num_samples: Optional[int]=100,
                **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Joint optimisation loop for Monte Carlo acquisition functions.
    
    Computes all points of a batch at once.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    method : ``str``
        One of "L-BFGS-B" or "Adam".
    batch_size : ``int``
        Number of points to return.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    discrete : ``dict``
        Possible values for all discrete inputs in the shape {dim1: [values1],
        dim2: [values2], etc.}, e.g. {0: [1., 2., 3.], 3: [-0.1, -0.2, 100.]}.
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

    if method not in ["L-BFGS-B", "Adam"]:
        raise NotImplementedError("Method must be one of L-BFGS-B or Adam.")

    # extend bounds to full batch
    full_bounds = torch.tile(bounds, (1, batch_size))

    # exetend discrete dims
    full_discrete = discrete

    # joint optimisation loop
    for i in range(batch_size):

        # continuous parameter space
        if not isinstance(discrete, dict):
            if method == "L-BFGS-B":
                batch_result, batch_func_result = lbfgsb(func, full_bounds, num_starts, num_samples, **kwargs)
            elif method == "Adam":
                batch_result, batch_func_result = adam(func, full_bounds, lr, steps, num_starts, num_samples, **kwargs)
    
        # mixed parameter space
        else:
            batch_result, batch_func_result = mixed(func, method, bounds, full_discrete, None, lr, steps, num_starts, num_samples, **kwargs)

    batch_result = torch.reshape(batch_result, (batch_size, -1))

    return batch_result, batch_func_result


def multi_sequential(func: Callable, 
                     method: str, 
                     batch_size: int, 
                     bounds: Tensor,
                     constraints: Optional[dict | list]=None,
                     discrete: Optional[dict]=None,
                     lr: Optional[float]=0.1,
                     steps: Optional[int]=100, 
                     num_starts: Optional[int]=10,
                     num_samples: Optional[int]=100, 
                     **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Sequential greedy optimisation loop for Monte Carlo acquisition functions.

    Computes one point after the other for a batch always keeping previous
    points fixed, i.e. compute point 1, compute point 2 holding point 1 fixed,
    compute point 3 holding point 1 and 2 fixed and so on until the batch is
    full.

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
    discrete : ``dict``
        Possible values for all discrete inputs in the shape {dim1: [values1],
        dim2: [values2], etc.}, e.g. {0: [1., 2., 3.], 3: [-0.1, -0.2, 100.]}.

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

    if method not in ["L-BFGS-B", "SLSQP", "Adam"]:
        raise NotImplementedError("Method must be one of L-BFGS-B, SLSQP or Adam.")
    
    dims = bounds.size(1)

    # initialise tensors for results
    batch_result = torch.zeros((0, dims))
    batch_func_result = torch.zeros(batch_size)

    # sequential optimisation loop
    for i in range(batch_size):

        # continuous parameter space
        if not isinstance(discrete, dict):
            if method == "L-BFGS-B":
                x_new, f_x_new = lbfgsb(func, bounds, num_starts, num_samples, **kwargs)
            elif method == "SLSQP":
                x_new, f_x_new = slsqp(func, bounds, constraints, num_starts, num_samples, **kwargs)
            elif method == "Adam":
                x_new, f_x_new = adam(func, bounds, lr, steps, num_starts, num_samples, **kwargs)

        # mixed parameter space
        else:
            x_new, f_x_new = mixed(func, method, bounds, discrete, constraints, lr, steps, num_starts, num_samples, **kwargs)

        # add new point to results
        batch_result = torch.cat([batch_result, x_new], dim=0)
        batch_func_result[i] = float(f_x_new)

        # add new point to pending points
        if isinstance(func.x_pending, Tensor):
            func.x_pending = torch.cat([func.x_pending, x_new], dim=0)
        else:
            func.x_pending = x_new

        # reset fixed base samples
        func.base_samples = None

    return batch_result, batch_func_result
