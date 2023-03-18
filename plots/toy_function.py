import torch
import math
from nubo.test_functions import TestFunction
from typing import Optional
from torch import Tensor


class Multimodal_1D(TestFunction):

    def __init__(self,
                 minimise: Optional[bool]=False) -> None:

        self.dims = 1
        self.bounds = Tensor([[0.0, ], [10.0, ]])
        self.optimum = {"inputs": Tensor([[7.9787 ]]), "ouput": Tensor([[0.0]])}
        self.minimise = minimise

    def __call__(self, x: Tensor) -> Tensor:
      
        # compute output
        y =  x * torch.sin(x)

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        return y.squeeze()