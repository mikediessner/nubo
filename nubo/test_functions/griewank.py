import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class Griewank(TestFunction):

    def __init__(self,
                 dims: int,
                 noise_std: Optional[float]=0.0,
                 minimise: Optional[bool]=True) -> None:

        self.dims = dims
        self.bounds = Tensor([[-600.0, ] * dims, [600.0, ] * dims])
        self.optimum = {"inputs": Tensor([[0.0, ] * dims]), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

    def __call__(self, x: Tensor) -> Tensor:
        
        # compute output
        ii = torch.arange(1, self.dims+1)
        y = torch.sum(x**2/4000.0, dim=-1) - torch.prod(torch.cos(x / torch.sqrt(ii)), dim=-1) + 1

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = torch.normal(mean=0, std=self.noise_std, size=y.size())
        f = y + noise

        return f
