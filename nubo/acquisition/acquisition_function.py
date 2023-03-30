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
        Wrapper to allow `x` to be a ``torch.Tensor`` or a ``numpy.ndarray``
        to enable optimisation with ``torch.optim`` and ``scipy.optimize``.

        Parameters
        ----------
        x : ``torch.Tensor`` or ``numpy.ndarray``
            (size 1 x d) Test point.
        
        Returns
        -------
        ``torch.Tensor`` or ``float``
            (size 1 or ``float``) Acquisition.
        """

        if isinstance(x, ndarray):
            x = torch.from_numpy(x).reshape(1, -1)
            acq = self.eval(x)
            acq = float(acq)
        elif isinstance(x, Tensor):
            acq = self.eval(x)
        else:
            raise ValueError("x must be np.ndarray or torch.Tensor")

        return acq
