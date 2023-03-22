import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class Schwefel(TestFunction):
    r"""
    :math:`d`-dimensional Schwefel function.

    The Schwefel function has many local minima and one global minimum
    :math:`f(\boldsymbol x^*) = 0` at
    :math:`\boldsymbol x^* = (420.9687, ..., 420.9687)`. It is usually 
    evaluated on the hypercube :math:`\boldsymbol x \in [-500, 500]^d`.

    .. math::
        f(\boldsymbol x) = 418.9829 d - \sum_{i=1}^d x_i \sin (\sqrt{\lvert x_i \rvert}).
    
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
        self.bounds = Tensor([[-500.0, ] * dims, [500.0, ] * dims])
        self.optimum = {"inputs": Tensor([[420., ] * dims]), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

    def eval(self, x: Tensor) -> Tensor:
        """
        Compute output of Schwefel function for some test points `x`.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x `dims`) Test points.
        """

        # compute output
        y = 418.9829 * self.dims - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))), dim=-1)

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = torch.normal(mean=0, std=self.noise_std, size=y.size())
        f = y + noise

        return f
