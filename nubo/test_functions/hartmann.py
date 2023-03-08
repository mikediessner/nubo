import torch
from nubo.test_functions import TestFunction
from torch import Tensor
from typing import Optional


class Hartmann3D(TestFunction):
    """
    3-dimensional Hartmann function.

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

        self.a = Tensor([1.0, 1.2, 3.0, 3.2]).T
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
    """
    6-dimensional Hartmann function.

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