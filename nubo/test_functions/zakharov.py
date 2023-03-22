import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class Zakharov(TestFunction):
    r"""
    :math:`d`-dimensional Zakharov function.

    The Zakharov function is plate-shaped and has one global minimum
    :math:`f(\boldsymbol x^*) = 0` at :math:`\boldsymbol x^* = (0, ..., 0)`.
    It is usually evaluated on the hypercube
    :math:`\boldsymbol x \in [-5, 10]^d`.

    .. math::
        f(\boldsymbol x) = \sum_{i=1}^d x_i^2 + \left( \sum_{i=1}^d 0.5 i x_i \right)^2 + \left( \sum_{i=1}^d 0.5 i x_i \right)^4.

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
        self.bounds = Tensor([[-5.0, ] * dims, [10.0, ] * dims])
        self.optimum = {"inputs": Tensor([[0.0, ] * dims]), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

    def eval(self, x: Tensor) -> Tensor:
        """
        Compute output of Zakharov function for some test points `x`.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x `dims`) Test points.
        """

        # compute output
        ii = torch.arange(1, self.dims+1)
        term_1 = torch.sum(x**2, dim=-1)
        term_2 = torch.sum(0.5 * ii * x, dim=-1) 
        y = term_1 + term_2**2 + term_2**4

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = torch.normal(mean=0, std=self.noise_std, size=y.size())
        f = y + noise

        return f
