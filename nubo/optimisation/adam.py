import torch
from torch import Tensor
from torch.optim import Adam
from typing import Optional, Tuple, Callable, Any
from scipy.optimize import OptimizeResult

def sigmoid(x: Tensor) -> Tensor:
    return 1/(1+torch.exp(-x))


def adam(func: callable,
         x: Tensor,
         lr: Optional[float]=0.01,
         steps: Optional[int]=100) -> None:
    
    x.requires_grad_(True)

    # specify Adam
    adam = Adam([x], lr=lr)

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
        bounds = torch.stack([-torch.ones(x.size(1)), torch.ones(x.size(1))])
        for j, (lb, ub) in enumerate (zip(*bounds)):
            x.data[:, j].clamp_(lb, ub)
  
    return x.detach(), loss


def multi_adam(func: Callable,
               candidates: Tensor,
               lr: Optional[float]=0.01,
               steps: Optional[int]=100) -> Tuple[Tensor, OptimizeResult]:
    """
    Multi-start optimisation.
    """
    
    n = candidates.size(0)

    # initialise objects for results
    x_res = torch.zeros(candidates.size())
    fun_res = torch.zeros(n)

    # iterate over candidates
    for i in range(n):
        candidate = candidates[i].reshape(1, -1)
        x, fun = adam(func, x=candidate, lr=lr, steps=steps)
        x_res[i, :] = x
        fun_res[i] = fun
    
    # select best start
    best_i = torch.argmax(fun_res)
    best_res = fun_res[best_i]
    best_x =  torch.reshape(x_res[best_i, :], (1, -1))

    return best_x, best_res