import torch
from nubo.acquisition import UpperConfidenceBound
from nubo.models import GaussianProcess, fit_adam
from nubo.optimisation import minimise
from nubo.test_functions import Ackley
from nubo.utils import LatinHypercubeSampling, normalise, unnormalise, standardise
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


# filter torch deprecation warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# test function
func = Ackley(dims=2, minimise=False) # maximisation problem
dims = func.dims
bounds = func.bounds

# training data
torch.manual_seed(1)
lhs = LatinHypercubeSampling(dims=dims)
x_train = lhs.maximin(points=dims*10)
x_train = unnormalise(x_train, bounds=bounds)
y_train = func(x_train)

# Bayesian optimisation loop
iters = 20

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

    # print parameters
    # print("Outputscale: ", gp.covar_module.outputscale)
    # print("Lengthscales: ", gp.covar_module.base_kernel.lengthscale)
    # print("Nugget: ", mll.likelihood.noise)
    
    # specify acquisition function
    ucb = UpperConfidenceBound(gp, beta=3.0)

    # optimise acquisition function with numpy/scipy
    opt_bounds = torch.Tensor([[0.] * dims, [1.] * dims])
    x0 = torch.rand(size=(1, dims))
    x_new, _ = minimise(func=ucb, x0=x0, bounds=opt_bounds, method="L-BFGS-B")

    # add to data
    x_new = unnormalise(x_new, bounds=bounds)
    y_new = func(x_new)
    x_train = torch.vstack((x_train, x_new))
    y_train = torch.hstack((y_train, y_new))

    # print new best
    if y_new > torch.max(y_train[:-1]):
        print(f"New best at evaluation {len(y_train)}: \t Inputs: {x_new.numpy().reshape(dims)}, \t Outputs: {-y_new.numpy()}")

# results
best_iter = int(torch.argmax(y_train))
print(f"Evaluation: {best_iter+1} \t Solution: {float(y_train[best_iter]):.4f}")
