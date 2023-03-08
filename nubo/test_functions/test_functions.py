from torch import Tensor


class TestFunction:
    """
    Parent class for all test functions.
    """

    def __init__(self):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        """
        Call ``eval`` method of the test function.
        """
        return self.eval(x)
