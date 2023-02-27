import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class DixonPrice(TestFunction):

    def __init__(self,
                 dims: int,
                 noise_std: Optional[float]=0.0,
                 minimise: Optional[bool]=True) -> None:

        ii = torch.arange(1, dims+1)
        optimals_xs = 2.0**( -(2.0**ii - 2.0)/2.0**ii )

        self.dims = dims
        self.bounds = Tensor([[-10.0, ] * dims, [10.0, ] * dims])
        self.optimum = {"inputs": Tensor(optimals_xs), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

    def __call__(self, x: Tensor) -> Tensor:
        
        # compute output
        ii = torch.arange(2, self.dims+1)
        term_1 = (x[:, 0] - 1.0)**2
        term_2 = torch.sum(ii * (2.0*x[:, 1:]**2 - x[:, :-1])**2, dim=-1)
        y = term_1 + term_2

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = torch.normal(mean=0, std=self.noise_std, size=y.size())
        f = y + noise

        return f
