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
           steps: Optional[int]=100,
           **kwargs) -> None:

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
