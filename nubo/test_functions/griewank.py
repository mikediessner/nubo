import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class Griewank(TestFunction):
    r"""
    :math:`d`-dimensional Griewank function.

    The Griewank function has many local minima and one global minimum
    :math:`f(\boldsymbol x^*) = 0` at :math:`\boldsymbol x^* = (0, ..., 0)`. It
    is usually evaluated on the hypercube
    :math:`\boldsymbol x \in [-600, 600]^d`.

    .. math::
        f(\boldsymbol x) = \sum_{i=1}^d \frac{x_i^2}{4000} - \prod_{i=1}^d \cos \left( \frac{x_i}{\sqrt{i}} \right) + 1.

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
        self.bounds = Tensor([[-600.0, ] * dims, [600.0, ] * dims])
        self.optimum = {"inputs": Tensor([[0.0, ] * dims]), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

    def eval(self, x: Tensor) -> Tensor:
        """
        Compute output of Griewank function for some test points `x`.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x `dims`) Test points.
        """
        
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
