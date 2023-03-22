import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class DixonPrice(TestFunction):
    r"""
    :math:`d`-dimensional Dixon-Price function.

    The Dixon-Price function is valley-shaped and has one global minimum
    :math:`f(\boldsymbol x^*) = 0` at :math:`x^*_i = 2^{-\frac{2^i-2}{2^i}}`, 
    for all :math:`i = 1, ..., d`. It is usually evaluated on the hypercube
    :math:`\boldsymbol x \in [-10, 10]^d`.

    .. math::
        f(\boldsymbol x) = (x_1 - 1)^2 + \sum_{i=2}^d i (2x_i^2 - x_{i-1})^2.

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
        ii = torch.arange(1, dims+1)
        optimals_xs = 2.0**( -(2.0**ii - 2.0)/2.0**ii )

        self.dims = dims
        self.bounds = Tensor([[-10.0, ] * dims, [10.0, ] * dims])
        self.optimum = {"inputs": Tensor(optimals_xs), "ouput": Tensor([[0.0]])}
        self.noise_std = noise_std
        self.minimise = minimise

    def eval(self, x: Tensor) -> Tensor:
        """
        Compute output of Dixon-Price function for some test points `x`.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x `dims`) Test points.
        """
        
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
