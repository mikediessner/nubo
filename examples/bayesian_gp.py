import torch
from nubo.acquisition import ExpectedImprovement, UpperConfidenceBound
from nubo.models import GaussianProcess
from nubo.optimisation import lbfgsb
from nubo.test_functions import Hartmann6D
from nubo.utils import LatinHypercubeSampling, unnormalise
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from gpytorch.constraints import Positive
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior
from gpytorch.settings import fast_computations


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
    likelihood = GaussianLikelihood(noise_constraint=Positive())
    gp = GaussianProcess(x_train, y_train, likelihood=likelihood)
    gp.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
    gp.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")
    gp.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
    likelihood.register_prior("noise_prior", UniformPrior(0.01, 0.5), "noise")
    mll = ExactMarginalLogLikelihood(likelihood, gp)

    # set up pyro model for sampling
    def pyro_gp(x, y):
        with fast_computations(False, False, False):
            sampled_gp = gp.pyro_sample_from_prior()
            output = sampled_gp.likelihood(sampled_gp(x))
            pyro.sample("obs", output, obs=y)
        return y

    # run MCMC
    nuts_kernel = NUTS(pyro_gp)
    mcmc_run = MCMC(nuts_kernel, num_samples=128, warmup_steps=128)
    mcmc_run.run(x_train, y_train)

    # load MCMC samples into model
    gp.pyro_load_from_samples(mcmc_run.get_samples())

    # specify acquisition function
    # acq = ExpectedImprovement(gp=gp, y_best=torch.max(y_train))
    acq = UpperConfidenceBound(gp=gp, beta=1.96**2)

    # optimise acquisition function
    x_new, _ = lbfgsb(func=lambda x: sum(acq(x))/mcmc_run.num_samples, bounds=bounds, num_starts=5)

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
