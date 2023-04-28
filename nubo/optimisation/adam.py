import torch
from torch import Tensor
from torch.optim import Adam
from typing import Optional, Tuple, Callable, Any
from nubo.optimisation import gen_candidates
from nubo.utils import unnormalise, normalise


def _adam(func: callable,
          x: Tensor,
          lr: Optional[float]=0.1,
          steps: Optional[int]=200,
          **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Adam optimiser. Minimises `func`.

    Parameters
    ----------
    x : ``torch.Tensor``
        (size 1 x d) Initial starting point of ``torch.optim.Adam`` algorithm.
    lr : ``float``, optional
        Learning rate of ``torch.optim.Adam`` algorithm, default is 0.1.
    steps : ``int``, optional
        Optimisation steps of ``torch.optim.Adam`` algorithm, default is 200.
    num_starts : ``int``
        Number of start for multi-start optimisation, default is 10.
    num_samples : ``int``
        Number of samples from which to draw the starts, default is 100.
    **kwargs : ``Any``
        Keyword argument passed to ``torch.optim.Adam``.

    Returns
    -------
    x : ``torch.Tensor``
        (size 1 x d) Minimiser input.
    loss : ``torch.Tensor``
        (size 1) Minimiser output.
    """

    x.requires_grad_(True)

    # specify Adam
    adam = Adam([x], lr=lr, **kwargs)

    # fit Gaussian process
    for i in range(steps):

        # set gradients from previous iteration equal to 0
        adam.zero_grad()
        
        # calculate loss
        loss = func(x)

        # backpropagate gradients
        loss.backward()

        # take next optimisation step
        adam.step()
  
    return x.detach(), loss


def adam(func: Callable,
         bounds: Tensor,
         lr: Optional[float]=0.1,
         steps: Optional[int]=200,
         num_starts: Optional[int]=10,
         num_samples: Optional[int]=100,
         **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Multi-start Adam optimiser using the ``torch.optim.Adam`` implementation
    from ``PyTorch``.
    
    Used for optimising Monte Carlo acquisition function when base samples are
    not fixed. Bounds are enforced by transforming `func` with the sigmoid
    function and scaling results. Picks the best `num_starts` points from a
    total `num_samples` Latin hypercube samples to initialise the optimiser.
    Returns the best result. Minimises `func`.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
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
        Keyword argument passed to ``torch.optim.Adam``.

    Returns
    -------
    best_result : ``torch.Tensor``
        (size 1 x d) Minimiser input.
    best_func_result : ``torch.Tensor``
        (size 1) Minimiser output.
    """
    
    dims = bounds.size(1)

    # transform function s.t. it takes real numbers
    trans_func = lambda x: func(unnormalise(torch.sigmoid(x), bounds).reshape(1, -1))

    # generate candidates and transfrom to real numbers
    candidates = gen_candidates(func, bounds, num_starts, num_samples)
    inv_sigmoid = lambda x: torch.log(x/(1-x))
    trans_candidates = inv_sigmoid(normalise(candidates, bounds))

    # initialise objects for results
    results = torch.zeros((num_starts, dims))
    func_results = torch.zeros(num_starts)

    # iteratively optimise over candidates
    for i in range(num_starts):
        x, fun = _adam(trans_func, lr=lr, x=trans_candidates[i], steps=steps, **kwargs)
        results[i, :] = unnormalise(torch.sigmoid(x), bounds) # transfrom results to bounds
        func_results[i] = fun
    
    # select best candidate
    best_i = torch.argmin(func_results)
    best_result =  torch.reshape(results[best_i, :], (1, -1))
    best_func_result = func_results[best_i]

    return best_result, torch.reshape(best_func_result, (1,))


def _adam_mixed(func: callable,
          x: Tensor,
          bounds: Tensor,
          lr: Optional[float]=0.1,
          steps: Optional[int]=200,
          **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Adam optimiser for mixed parameters. Minimises `func`.

    Parameters
    ----------
    x : ``torch.Tensor``
        (size 1 x d) Initial starting point of ``torch.optim.Adam`` algorithm.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    lr : ``float``, optional
        Learning rate of ``torch.optim.Adam`` algorithm, default is 0.1.
    steps : ``int``, optional
        Optimisation steps of ``torch.optim.Adam`` algorithm, default is 200.
    num_starts : ``int``
        Number of start for multi-start optimisation, default is 10.
    num_samples : ``int``
        Number of samples from which to draw the starts, default is 100.
    **kwargs : ``Any``
        Keyword argument passed to ``torch.optim.Adam``.

    Returns
    -------
    x : ``torch.Tensor``
        (size 1 x d) Minimiser input.
    loss : ``torch.Tensor``
        (size 1) Minimiser output.
    """

    x.requires_grad_(True)

    # specify Adam
    adam = Adam([x], lr=lr, **kwargs)

    # fit Gaussian process
    for i in range(steps):

        # set gradients from previous iteration equal to 0
        adam.zero_grad()
        
        # calculate loss
        loss = func(x)

        # backpropagate gradients
        loss.backward()

        # take next optimisation step
        adam.step()

        # enforce bounds
        with torch.no_grad():
            x[:] = x.clamp(min=bounds[0, :], max=bounds[1, :])
  
    return x.detach(), loss


def adam_mixed(func: Callable,
               bounds: Tensor,
               lr: Optional[float]=0.1,
               steps: Optional[int]=200,
               num_starts: Optional[int]=10,
               num_samples: Optional[int]=100,
               **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """
    Multi-start Adam optimiser using the ``torch.optim.Adam`` implementation
    from ``PyTorch``.
    
    Used for optimising Monte Carlo acquisition function when base samples are
    not fixed. Bounds are enforced by clamping where values exceed them. Picks
    the best `num_starts` points from a total `num_samples` Latin hypercube
    samples to initialise the optimiser. Returns the best result. Minimises
    `func`.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
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
        Keyword argument passed to ``torch.optim.Adam``.

    Returns
    -------
    best_result : ``torch.Tensor``
        (size 1 x d) Minimiser input.
    best_func_result : ``torch.Tensor``
        (size 1) Minimiser output.
    """
    
    dims = bounds.size(1)

    # generate candidates and transfrom to real numbers
    candidates = gen_candidates(func, bounds, num_starts, num_samples)

    # initialise objects for results
    results = torch.zeros((num_starts, dims))
    func_results = torch.zeros(num_starts)

    # iteratively optimise over candidates
    for i in range(num_starts):
        results[i, :], func_results[i] = _adam_mixed(func, x=candidates[i], bounds=bounds, lr=lr, steps=steps, **kwargs)
        
    # select best candidate
    best_i = torch.argmin(func_results)
    best_result =  torch.reshape(results[best_i, :], (1, -1))
    best_func_result = func_results[best_i]

    return best_result, torch.reshape(best_func_result, (1,))
