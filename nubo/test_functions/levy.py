import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class Levy(TestFunction):
    r"""
    :math:`d`-dimensional Levy function.

    The Levy function has many local minima and one global minimum
    :math:`f(\boldsymbol x^*) = 0` at :math:`\boldsymbol x^* = (1, ..., 1)`. It
    is usually evaluated on the hypercube :math:`\boldsymbol x \in [-10, 10]^d`.

    .. math::
        f(\boldsymbol x) = \sin^2 (\pi w_1) + \sum_{i=1}^{d-1} (w_i - 1)^2 [1 + 10 \sin^2 (\pi w_i + 1)] + (w_d - 1)^2 [1 + \sin^2(2\pi w_d)],

    where :math:`w_i = 1 + \frac{x_i - 1}{4}`, for all :math:`i = 1, ..., d`.

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
        self.optimum = {"inputs": Tensor([[1.0, ] * dims]), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

    def eval(self, x: Tensor) -> Tensor:
        """
        Compute output of Levy function for some test points `x`.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x `dims`) Test points.
        """
        
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
