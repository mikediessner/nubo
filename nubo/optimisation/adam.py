import torch
from torch import Tensor
from torch.optim import Adam
from typing import Optional, Tuple, Callable, Any
from nubo.optimisation import gen_candidates
from nubo.utils import unnormalise, normalise


def _adam(func: callable,
          x: Tensor,
          steps: Optional[int]=100,
          **kwargs: Any) -> None:
    
    x.requires_grad_(True)

    # specify Adam
    adam = Adam([x], **kwargs)

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
               steps: Optional[int]=100,
               num_starts: Optional[int]=10,
               num_samples: Optional[int]=100,
               **kwargs: Any) -> Tuple[Tensor, float]:
    """
    Multi-start optimisation.
    """
    
    dims = bounds.size(1)

    # transform function s.t. it takes real numbers
    trans_func = lambda x: func(unnormalise(torch.sigmoid(x), bounds))

    # generate candidates and transfrom to real numbers
    candidates = gen_candidates(func, bounds, num_starts, num_samples)
    inv_sigmoid = lambda x: torch.log(x/(1-x))
    trans_candidates = inv_sigmoid(normalise(candidates, bounds))

    # initialise objects for results
    results = torch.zeros((num_starts, dims))
    func_results = torch.zeros(num_starts)

    # iteratively optimise over candidates
    for i in range(num_starts):
        x, fun = _adam(trans_func, x=trans_candidates[i], steps=steps, **kwargs)
        results[i, :] = unnormalise(torch.sigmoid(x), bounds) # transfrom results to bounds
        func_results[i] = fun
    
    # select best candidate
    best_i = torch.argmax(func_results)
    best_result =  torch.reshape(results[best_i, :], (1, -1))
    best_func_result = func_results[best_i]

    return best_result, best_func_result