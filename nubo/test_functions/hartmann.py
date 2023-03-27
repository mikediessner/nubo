import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class Hartmann3D(TestFunction):
    r"""
    3-dimensional Hartmann function.

    The 3-dimensional Hartmann function has four local minima and one global
    minimum :math:`f(\boldsymbol x^*) = -3.86278` at
    :math:`\boldsymbol x^* = (0.114614, 0.555649, 0.852547)`. It is usually
    evaluated on the hypercube :math:`\boldsymbol x \in (0, 1)^3`.

    .. math::
        f(\boldsymbol x) = - \sum_{i=1}^4 \alpha_i \exp \left( - \sum_{j=1}^3 A_{ij} (x_j - P_{ij})^2 \right),

    where 
    
    .. math::
        \alpha = (1.0, 1.2, 3.0, 3.2)^T,

    .. math::
        \boldsymbol A = \begin{pmatrix}
        3.0 & 10.0 & 30.0 \\
        0.1 & 10.0 & 35.0 \\
        3.0 & 10.0 & 30.0 \\
        0.1 & 10.0 & 35.0
        \end{pmatrix},
    
    .. math::
        \text{and } \boldsymbol P = 10^{-4} \begin{pmatrix}
                                            3689 & 1170 & 2673 \\
                                            4699 & 4387 & 7470 \\
                                            1091 & 8732 & 5547 \\
                                            381  & 5743 & 8828
                                            \end{pmatrix}.

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
    a : ``torch.Tensor``
        (size 4 x 1) Function parameters.
    A : ``torch.Tensor``
        (size 4 x 3) Function parameters.
    P : ``torch.Tensor``
        (size 4 x 3) Function parameters.
    """

    def __init__(self,
                 noise_std: Optional[float]=0.0,
                 minimise: Optional[bool]=True) -> None:
        """
        Parameters
        ----------
        noise_std : ``float``, optional
            Standard deviation of Gaussian noise, default is 0.0.
        minimise : ``bool``, optional
            Minimisation problem if true (default), maximisation problem if
            false.
        """

        self.dims = 3
        self.bounds = Tensor([[0.0, ] * self.dims, [1.0, ] * self.dims])
        self.optimum = {"inputs": Tensor([[0.114614, 0.555649, 0.852547]]), 
                        "ouput": Tensor([[-3.86278]])}
        self.noise_std = noise_std
        self.minimise = minimise

        self.a = Tensor([1.0, 1.2, 3.0, 3.2])
        self.A = Tensor([[3.0, 10.0, 30.0],
                         [0.1, 10.0, 35.0],
                         [3.0, 10.0, 30.0],
                         [0.1, 10.0, 35.0]])
        self.P = 10**-4 * Tensor([[3689.0, 1170.0, 2673.0],
                                  [4699.0, 4387.0, 7470.0],
                                  [1091.0, 8732.0, 5547.0],
                                  [ 381.0, 5743.0, 8828.0]])

    def eval(self, x: Tensor) -> Tensor:
        """
        Compute output of Hartmann function for some test points `x`.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x `dims`) Test points.
        """ 
        
        # reformat to allow matrix computation of multiple points
        n = x.size(0)
        d = x.size(1)
        x = torch.reshape(x, (n, 1, d))
        
        # compute output
        y = -torch.sum(self.a * torch.exp(-torch.sum(self.A * (x - self.P)**2, dim=-1)), dim=-1)

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = torch.normal(mean=0, std=self.noise_std, size=y.size())
        f = y + noise

        return f


class Hartmann6D(TestFunction):
    r"""
    6-dimensional Hartmann function.

    The 6-dimensional Hartmann function has six local minima and one global
    minimum :math:`f(\boldsymbol x^*) = -3.32237` at
    :math:`\boldsymbol x^* = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)`.
    It is usually evaluated on the hypercube :math:`\boldsymbol x \in (0, 1)^6`.

    .. math::
        f(\boldsymbol x) = - \sum_{i=1}^4 \alpha_i \exp \left( - \sum_{j=1}^6 A_{ij} (x_j - P_{ij})^2 \right),

    where 
    
    .. math::
        \alpha = (1.0, 1.2, 3.0, 3.2)^T,

    .. math::
        \boldsymbol A = \begin{pmatrix}
        10.00 &  3.00 & 17.00 &  3.50 &  1.70 &  8.00 \\
         0.05 & 10.00 & 17.00 &  0.10 &  8.00 & 14.00 \\
         3.00 &  3.50 &  1.70 & 10.00 & 17.00 &  8.00 \\
        17.00 &  8.00 &  0.05 & 10.00 &  0.10 & 14.00
        \end{pmatrix}, 
    
    .. math::
        \text{and } \boldsymbol P = 10^{-4} \begin{pmatrix}
                                            1312 & 1696 & 5569 &  124 & 8283 & 5886 \\
                                            2329 & 4135 & 8307 & 3736 & 1004 & 9991 \\
                                            2348 & 1451 & 3522 & 2883 & 3047 & 6650 \\
                                            4047 & 8828 & 8732 & 5743 & 1091 &  381
                                            \end{pmatrix}.

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
    a : ``torch.Tensor``
        (size 4 x 1) Function parameters.
    A : ``torch.Tensor``
        (size 4 x 6) Function parameters.
    P : ``torch.Tensor``
        (size 4 x 6) Function parameters.
    """    
    
    def __init__(self,
                 noise_std: Optional[float]=0.0,
                 minimise: Optional[bool]=True) -> None:
        """
        Parameters
        ----------
        noise_std : ``float``, optional
            Standard deviation of Gaussian noise, default is 0.0.
        minimise : ``bool``, optional
            Minimisation problem if true (default), maximisation problem if
            false.
        """

        self.dims = 6
        self.bounds = Tensor([[0.0, ] * self.dims, [1.0, ] * self.dims])
        self.optimum = {"inputs": Tensor([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]), 
                        "ouput": Tensor([[-3.32237]])}
        self.noise_std = noise_std
        self.minimise = minimise

        self.a = Tensor([1.0, 1.2, 3.0, 3.2])
        self.A = Tensor([[10.0,  3.0, 17.0,  3.5,  1.7,  8.0],
                        [0.05, 10.0, 17.0,  0.1,  8.0, 14.0],
                        [ 3.0,  3.5,  1.7, 10.0, 17.0,  8.0],
                        [17.0,  8.0, 0.05, 10.0,  0.1, 14.0]])
        self.P = 0.0001 * Tensor([[1312.0, 1696.0, 5569.0,  124.0, 8283.0, 5886.0],
                                  [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
                                  [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
                                  [4047.0, 8828.0, 8732.0, 5743.0, 1091.0,  381.0]])

    def eval(self, x: Tensor) -> Tensor:
        """
        Compute output of Hartmann function for some test points `x`.

        Parameters
        ----------
        x : ``torch.Tensor``
            (size n x `dims`) Test points.
        """ 
        
        # reformat to allow matrix computation of multiple points
        n = x.size(0)
        d = x.size(1)
        x = torch.reshape(x, (n, 1, d))

        # compute output
        y = -torch.sum(self.a * torch.exp(-torch.sum(self.A * (x - self.P)**2, dim=-1)), dim=-1)

        # turn into maximisation problem
        if not self.minimise:
            y = -y

        # add noise
        noise = torch.normal(mean=0, std=self.noise_std, size=y.size())
        f = y + noise

        return f