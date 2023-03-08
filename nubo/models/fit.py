from torch import Tensor
from torch.optim import Adam
from gpytorch.models import GP
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import MarginalLogLikelihood
from typing import Optional


def fit_gp(x: Tensor,
           y: Tensor,
           gp: GP,
           likelihood: Likelihood,
           mll: MarginalLogLikelihood,
           lr: Optional[float]=0.1,
           steps: Optional[int]=200,
           **kwargs) -> None:
    """
    Estimate hyper-parameters of the Gaussian process `gp` by maximum
    likelihood estimation (MLE) using ``torch.optim.Adam`` algorithm.

    Parameters
    ----------
    x : ``torch.Tensor``
        (size n x d) Training inputs.
    y : ``torch.Tensor``
        (size n) Training targets.
    gp : ``gpytorch.likelihoods.Likelihood``
        Gaussian Process model.
    mll : ``gpytorch.mlls.MarginalLogLikelihood``
        Marginal log likelihood.
    lr : ``float``, optional
        Learning rate of ``torch.optim.Adam`` algorithm, default is 0.1.
    steps : ``int``, optional
        Optimisation steps of ``torch.optim.Adam`` algorithm, default is 200.
    **kwargs : ``Any``
        Keyword argument passed to ``torch.optim.Adam``.
    """

    # set Gaussian process and likelihood to training mode
    gp.train()
    likelihood.train()

    # specify Adam
    adam = Adam(gp.parameters(), lr=lr, **kwargs)

    # fit Gaussian process
    for i in range(steps):

        # set gradients from previous iteration equal to 0
        adam.zero_grad()

        # output from GP
        output = gp(x)

        # calculate loss
        loss = -mll(output, y)

        # backpropagate gradients
        loss.backward()

        # take next optimisation step
        adam.step()
