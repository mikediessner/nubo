import torch
from nubo.acquisition import MCUpperConfidenceBound, MCExpectedImprovement
from nubo.models import GaussianProcess, fit_adam
from nubo.optimisation import autograd_minimise, gen_candidates, autograd_multi_start, multi_start
from nubo.test_functions import Ackley, DixonPrice, Griewank, Hartmann3D, Hartmann6D, Levy, Rastrigin, Schwefel, Sphere, SumSquares, Zakharov
from nubo.utils import LatinHypercubeSampling, normalise, unnormalise, standardise
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats.qmc import Sobol


# filter torch deprecation warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# test function
# func = Ackley(dims=4, minimise=False) # maximisation problem
func = Hartmann6D(minimise=False)
dims = func.dims
bounds = func.bounds

# training data
# torch.manual_seed(1)
# lhs = LatinHypercubeSampling(dims=dims)
# x_train = lhs.maximin(points=dims*10)
sobol = Sobol(dims, seed=1)
x_train = torch.from_numpy(sobol.random_base2(m=5))
x_train = unnormalise(x_train, bounds=bounds)
y_train = func(x_train)

# Bayesian optimisation loop
iters = 10
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

    # compute rest of points
    for j in range(batch_size):
        # specify acquisition function
        acq = MCExpectedImprovement(mc_samples, gp, torch.max(y_train), x_pending=X_new, fix_base_samples=True)
        # acq = MCUpperConfidenceBound(mc_samples, gp, 5.0, x_pending=X_new, fix_base_samples=True)
        # optimise acquisition function with numpy/scipy
        opt_bounds = torch.Tensor([[0.]*dims, [1.]*dims])
        candidates = gen_candidates(acq, dims=dims, bounds=opt_bounds, num_candidates=5, num_samples=100)
        x_new, _ = autograd_multi_start(func=acq, candidates=candidates, bounds=opt_bounds, method="L-BFGS-B")
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
