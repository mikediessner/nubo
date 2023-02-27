import torch
from nubo.acquisition import MCUpperConfidenceBound, MCExpectedImprovement
from nubo.models import GaussianProcess, fit_adam
from nubo.optimisation import autograd_minimise, gen_candidates, autograd_multi_start, multi_start, adam, multi_adam
from nubo.test_functions import Ackley, DixonPrice, Griewank, Hartmann3D, Hartmann6D, Levy, Rastrigin, Schwefel, Sphere, SumSquares, Zakharov
from nubo.utils import LatinHypercubeSampling, normalise, unnormalise, standardise
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


# filter torch deprecation warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# test function
func = Ackley(dims=4, minimise=False) # maximisation problem
dims = func.dims
bounds = func.bounds

# training data
# torch.manual_seed(1)
lhs = LatinHypercubeSampling(dims=dims)
x_train = lhs.maximin(points=dims*10)
x_train = unnormalise(x_train, bounds=bounds)
y_train = func(x_train)

# Bayesian optimisation loop
iters = 20
batch_size = 4
mc_samples = 8

for iter in range(iters):

    # normalise and scale training data
    x = normalise(x_train, bounds=bounds)
    y = standardise(y_train)

    # specify Gaussian process
    likelihood = GaussianLikelihood()
    gp = GaussianProcess(x, y, likelihood)
    mll = ExactMarginalLogLikelihood(likelihood, gp)
    
    # fit Gaussian process
    fit_adam(x, y, gp=gp, likelihood=likelihood, mll=mll, steps=200)
    
    # initialise new points
    X_new = torch.zeros((0, dims))

    # compute batch of points
    for j in range(batch_size):
        # specify acquisition function
        acq = MCExpectedImprovement(mc_samples, gp, torch.max(y_train), x_pending=X_new)
        # optimise acquisition function with numpy/scipy
        # x0 = torch.randn((1, dims))
        # x_new = adam(func=acq, x=x0, lr=0.1, steps=200)
        opt_bounds = torch.Tensor([[0.]*dims, [1.]*dims])
        candidates = gen_candidates(acq, dims=dims, bounds=opt_bounds, num_candidates=5, num_samples=100)
        x_new, _ = multi_adam(acq, candidates, 0.1, 200)
        X_new = torch.cat([X_new, x_new], dim=0)

    # add to data
    X_new = unnormalise(X_new, bounds=bounds)
    y_new = func(X_new)
    x_train = torch.vstack((x_train, X_new))
    y_train = torch.hstack((y_train, y_new))
    print(X_new)

    # print new best
    if torch.max(y_new) > torch.max(y_train[:-batch_size]):
        i = torch.argmax(y_train)
        print(f"New best at evaluation {i+1}: \t Inputs: {x_train[i, :].numpy().reshape(dims)}, \t Outputs: {-y_train[i].numpy()}")


# results
best_iter = int(torch.argmax(y_train))
print(f"Evaluation: {best_iter+1} \t Solution: {-float(y_train[best_iter]):.4f}")
