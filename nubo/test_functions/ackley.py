import torch
import math
from nubo.test_functions import TestFunction
from typing import Optional
from torch import Tensor


class Ackley(TestFunction):
    r"""
    :math:`d`-dimensional Ackley function.

    The Ackley function has many local minima and one global minimum
    :math:`f(\boldsymbol x^*) = 0` at :math:`\boldsymbol x^* = (0, ..., 0)`.
    It is usually evaluated on the hypercube
    :math:`\boldsymbol x \in [-32.768, 32.768]^d`.

    .. math::
        f(\boldsymbol x) = -a \exp \left(-b \sqrt{\frac{1}{d} \sum^d_{i=1} x^2_i}\right) - \exp \left( \frac{1}{d} \sum^d_{i=1} \cos(c x_i)\right) + a + \exp(1),

    where :math:`a = 20`, :math:`b = 0.2`, and :math:`c = 2\pi`.

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
    a : ``float``
        Function parameter.
    b : ``float``
        Function parameter.
    c : ``float``
        Function parameter.
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
        self.bounds = Tensor([[-32.768, ] * dims, [32.768, ] * dims])
        self.optimum = {"inputs": Tensor([[0.0, ] * dims]), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

        self.a = 20.0
        self.b = 0.2
        self.c = 2.0*torch.pi

    def eval(self, x: Tensor) -> Tensor:
        """
        Compute output of Ackley function for some test points `x`.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x `dims`) Test points.
        """
      
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
