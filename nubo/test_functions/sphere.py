import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class Sphere(TestFunction):
    r"""
    :math:`d`-dimensional Sphere function.

    The Sphere function is bowl-shaped and has one global minimum
    :math:`f(\boldsymbol x^*) = 0` at :math:`\boldsymbol x^* = (0, ..., 0)`. It
    is usually evaluated on the hypercube
    :math:`\boldsymbol x \in [-5.12, 5.12]^d`.

    .. math::
        f(\boldsymbol x) = \sum_{i=1}^d x_i^2.
    
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
        self.bounds = Tensor([[-5.12, ] * dims, [5.12, ] * dims])
        self.optimum = {"inputs": Tensor([[0.0, ] * dims]), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

    def eval(self, x: Tensor) -> Tensor:
        """
        Compute output of Sphere function for some test points `x`.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x `dims`) Test points.
        """

        # compute output
        y = torch.sum(x**2, dim=-1)

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = torch.normal(mean=0, std=self.noise_std, size=y.size())
        f = y + noise

        return f
