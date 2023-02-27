import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class Schwefel(TestFunction):

    def __init__(self,
                 dims: int,
                 noise_std: Optional[float]=0.0,
                 minimise: Optional[bool]=True) -> None:

        self.dims = dims
        self.bounds = Tensor([[-500.0, ] * dims, [500.0, ] * dims])
        self.optimum = {"inputs": Tensor([[420., ] * dims]), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

    def __call__(self, x: Tensor) -> Tensor:
        
        # compute output
        y = 418.9829 * self.dims - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))), dim=-1)

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = torch.normal(mean=0, std=self.noise_std, size=y.size())
        f = y + noise

        return f
