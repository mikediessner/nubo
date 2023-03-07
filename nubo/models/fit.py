import torch
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
    likelihood estimation (MLE) using :obj:`torch.optim.Adam` algorithm.

    Parameters
    ----------
    x : :obj:`torch.Tensor`
        (size n x d) Training inputs.
    y : :obj:`torch.Tensor`
        (size n) Training targets.
    gp : :obj:`gpytorch.likelihoods.Likelihood`
        Gaussian Process model.
    mll : :obj:`gpytorch.mlls.MarginalLogLikelihood`
        Marginal log likelihood.
    lr : :obj:`float`, optional
        Learning rate of :obj:`torch.optim.Adam` algorithm, default is 0.1.
    steps : :obj:`int`, optional
        Optimisation steps of :obj:`torch.optim.Adam` algorithm, default is
        200.
    **kwargs : :obj:`Any`
        Keyword argument passed to :obj:`torch.optim.Adam`.
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
