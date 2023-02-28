import torch
from torch import Tensor
from numpy import ndarray


class AcquisitionFunction:

    def __init__(self) -> None:
        pass

    def __call__(self, x: Tensor | ndarray) -> Tensor | ndarray:
        """
        Wrapper: torch to numpy conversion and back.
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
