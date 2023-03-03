import torch
from nubo.acquisition import ExpectedImprovement, UpperConfidenceBound
from nubo.models import GaussianProcess, fit_gp
from nubo.optimisation import slsqp
from nubo.test_functions import Hartmann6D
from nubo.utils import LatinHypercubeSampling, unnormalise
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


# test function
func = Hartmann6D(minimise=False)
dims = func.dims
bounds = func.bounds

# training data
torch.manual_seed(1)
lhs = LatinHypercubeSampling(dims=dims)
x_train = lhs.maximin(points=dims*5)
x_train = unnormalise(x_train, bounds=bounds)
y_train = func(x_train)

# Bayesian optimisation loop
iters = 40

for iter in range(iters):
    
    # specify Gaussian process
    likelihood = GaussianLikelihood()
    gp = GaussianProcess(x_train, y_train, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(likelihood=likelihood, model=gp)
    
    # fit Gaussian process
    fit_gp(x_train, y_train, gp=gp, likelihood=likelihood, mll=mll, lr=0.1, steps=200)

    # specify acquisition function
    # acq = ExpectedImprovement(gp=gp, y_best=torch.max(y_train))
    acq = UpperConfidenceBound(gp=gp, beta=1.96**2)

    # define constraints
    cons = ({'type': 'ineq', 'fun': lambda x: 0.5 - x[0] - x[1] },
            {'type': 'eq', 'fun': lambda x: 1.2442 - x[3] - x[4] - x[5]})
    
    # optimise acquisition function
    x_new, _ = slsqp(func=acq, bounds=bounds, constraints=cons, num_starts=5)

    # evaluate new point
    y_new = func(x_new)
    
    # add to data
    x_train = torch.vstack((x_train, x_new))
    y_train = torch.hstack((y_train, y_new))

    # print new best
    if y_new > torch.max(y_train[:-1]):
        print(f"New best at evaluation {len(y_train)}: \t Inputs: {x_new.numpy().reshape(dims)}, \t Outputs: {-y_new.numpy()}")

# results
best_iter = int(torch.argmax(y_train))
print(f"Evaluation: {best_iter+1} \t Solution: {float(y_train[best_iter]):.4f}")
