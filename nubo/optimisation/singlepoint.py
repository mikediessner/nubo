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

    Optimises the acquisition function with the L-BFGS-B, SLSQP, or Adam
    optimiser. Minimises `func`.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    method : ``str``
        One of "L-BFGS-B", "SLSQP", or "Adam".
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
