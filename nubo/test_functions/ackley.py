import torch
import math
from nubo.test_functions import TestFunction
from typing import Optional
from torch import Tensor


class Ackley(TestFunction):

    def __init__(self,
                 dims: int,
                 noise_std: Optional[float]=0.0,
                 minimise: Optional[bool]=True) -> None:

        self.dims = dims
        self.bounds = Tensor([[-32.768, ] * dims, [32.768, ] * dims])
        self.optimum = {"inputs": Tensor([[0.0, ] * dims]), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

        self.a = 20.0
        self.b = 0.2
        self.c = 2.0*torch.pi

    def __call__(self, x: Tensor) -> Tensor:
      
        # compute output
        term_1 = -self.a * torch.exp(-self.b * torch.sqrt(torch.mean(x**2, dim=-1)))
        term_2 = -torch.exp(torch.mean(torch.cos(self.c * x), dim=-1))
        y = term_1 + term_2 + self.a + math.exp(1.0)

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = torch.normal(mean=0, std=self.noise_std, size=y.size())
        f = y + noise

        return f
