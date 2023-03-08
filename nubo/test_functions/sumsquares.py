import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class SumSquares(TestFunction):
    """
    d-dimensional Sum-of-Squares function.

    Attributes
    ----------
    dims : ``int``
        Number of input dimensions.
    noise_std : ``float``
        Standard deviation of Gaussian noise.
    minimise : ``bool``
        Minimisation problem if true, maximisation problem if false.
    bounds : ``torch.Tensor``
        (size 2 x `dims`) Bounds of input space.
    optimum : ``dict``
        Contains inputs and output of global maximum.
    """

    def __init__(self,
                 dims: int,
                 noise_std: Optional[float]=0.0,
                 minimise: Optional[bool]=True) -> None:
        """
        Parameters
        ----------
        dims : ``int``
            Number of input dimensions.
        noise_std : ``float``, optional
            Standard deviation of Gaussian noise, default is 0.0.
        minimise : ``bool``, optional
            Minimisation problem if true (default), maximisation problem if
            false.
        """
        
        self.dims = dims
        self.bounds = Tensor([[-10.0, ] * dims, [10.0, ] * dims])
        self.optimum = {"inputs": Tensor([[0.0, ] * dims]), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

    def eval(self, x: Tensor) -> Tensor:
        """
        Compute output of Sum-of-Squares function for some test points `x`.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x `dims`) Test points.
        """

        # compute output
        ii = torch.arange(1, self.dims+1)
        y = torch.sum(ii * x**2, dim=-1)

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = torch.normal(mean=0, std=self.noise_std, size=y.size())
        f = y + noise

        return f
