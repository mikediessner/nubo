import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class Rastrigin(TestFunction):

    def __init__(self,
                 dims: int,
                 noise_std: Optional[float]=0.0,
                 minimise: Optional[bool]=True) -> None:

        self.dims = dims
        self.bounds = Tensor([[-5.12, ] * dims, [5.12, ] * dims])
        self.optimum = {"inputs": Tensor([[0.0, ] * dims]), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

    def __call__(self, x: Tensor) -> Tensor:
        
        # compute output
        y = 10.0*self.dims + torch.sum(x**2 - 10.0*torch.cos(2.0*torch.pi*x), dim=-1)

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = torch.normal(mean=0, std=self.noise_std, size=y.size())
        f = y + noise

        return f
