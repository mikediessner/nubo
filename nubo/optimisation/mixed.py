import torch
from torch import Tensor
import itertools
from typing import Tuple, Optional, Callable, Any
from nubo.optimisation import lbfgsb, slsqp, adam_mixed


def mixed(func: Callable,
          method: str,
          bounds: Tensor,
          discrete: dict,
          constraints: Optional[dict | list]=None,
          lr: Optional[float]=0.1,
          steps: Optional[int]=200,
          num_starts: Optional[int]=10,
          num_samples: Optional[int]=100,
          **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Mixed optimisation with continuous and discrete inputs.

    Optimises the acquisition over all continuous input dimensions by fixing a
    combination of the discrete inputs. Returns the best result over all
    possible discrete combinations. Minimises `func`.
    
    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    method : ``str``
        One of "L-BFGS-B", "SLSQP", or "Adam".
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    discrete : ``dict``
        Possible values for all discrete inputs in the shape {dim1: [values1],
        dim2: [values2], etc.}, e.g. {0: [1., 2., 3.], 3: [-0.1, -0.2, 100.]}.
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
    best_result : ``torch.Tensor``
        (size 1 x d) Minimiser inputs.
    best_func_result : ``torch.Tensor``
        (size 1) Minimiser output.
    """

    # get discrete dimensions and values
    discrete_dims = list(discrete.keys())
    discrete_values = list(discrete.values())

    # get all possible combinations of discrete inputs
    all_combos = list(itertools.product(*discrete_values))

    # initialise results
    results = torch.zeros((len(all_combos), bounds.size(1)))
    func_results = torch.zeros((len(all_combos)))
    
    # select discrete combination
    for i, fixed_value in enumerate(all_combos):

        # fix combination
        fixed_bounds = bounds
        for j, dim in enumerate(discrete_dims):
            fixed_bounds[:, dim] = fixed_value[j]

        # optimise acquisition function 
        if method == "L-BFGS-B":
            results[i], func_results[i] = lbfgsb(func, fixed_bounds, num_starts, num_samples, **kwargs)
        elif method == "SLSQP":
            results[i], func_results[i] = slsqp(func, fixed_bounds, constraints, num_starts, num_samples, **kwargs)
        elif method == "Adam":
            results[i], func_results[i] = adam_mixed(func, fixed_bounds, lr, steps, num_starts, num_samples, **kwargs)
        else:
            raise NotImplementedError("Method must be one of L-BFGS-B, SLSQP or Adam.")

    # get best result 
    best_i = torch.argmin(func_results)
    best_result =  torch.reshape(results[best_i, :], (1, -1))
    best_func_result = torch.reshape(func_results[best_i], (1,))

    return best_result, best_func_result
