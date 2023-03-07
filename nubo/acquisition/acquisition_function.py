import torch
from torch import Tensor
from numpy import ndarray


class AcquisitionFunction:
    """
    Parent class of all acquisition functions.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, x: Tensor | ndarray) -> Tensor | ndarray:
        """
        Wrapper to allow `x` to be a :obj:`torch.Tensor` or a
        :obj:`numpy.ndarray` to enable optimisation with :obj:`torch.optim`
        and :obj:`scipy.optimize`.

        Parameters
        ----------
        x : :obj:`torch.Tesor` or :obj:`numpy.ndarray`
            (size 1 x d) Test point.
        """

        if isinstance(x, ndarray):
            x = torch.from_numpy(x).unsqueeze(0)
            acq = self.eval(x)
            acq = acq.squeeze().detach().numpy()
        elif isinstance(x, Tensor):
            acq = self.eval(x)
        else:
            ValueError("x must be np.ndarray or torch.Tensor")

        return acq
