import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class Levy(TestFunction):

    def __init__(self,
                 dims: int,
                 noise_std: Optional[float]=0.0,
                 minimise: Optional[bool]=True) -> None:

        self.dims = dims
        self.bounds = Tensor([[-10.0, ] * dims, [10.0, ] * dims])
        self.optimum = {"inputs": Tensor([[1.0, ] * dims]), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

    def __call__(self, x: Tensor) -> Tensor:
        
        # compute output
        w = 1.0 + (x - 1.0)/4.0
        term_1 = torch.sin(torch.pi * w[:, 0])**2
        term_2 = torch.sum((w[:, :-1] - 1.0)**2 * (1.0 + 10.0 * torch.sin(torch.pi * w[:, :-1] + 1.0)**2), dim=-1)
        term_3 = (w[:, -1] - 1.0)**2 * (1.0 + torch.sin(2.0 * torch.pi * w[:, -1])**2)
        y = term_1 + term_2 + term_3

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = torch.normal(mean=0, std=self.noise_std, size=y.size())
        f = y + noise

        return f
