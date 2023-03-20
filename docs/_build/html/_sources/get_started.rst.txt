.. _get_started:

Get started
===========
This brief introduction will teach you how to install NUBO from the GitHub
repository and how to set up a Bayeisan optimisation loop to maximise a toy
function using NUBO's pre-defined Gaussian process as the surrogate model.

Installing NUBO
---------------
Install NUBO and all its dependencies directly from the GitHub repository using
the packet manager `pip` with the following code. We recommend the use of a
virtual environment.

::

    pip install git+https://github.com/mikediessner/nubo


Optimising a toy function with NUBO
-----------------------------------
First, we set-up the toy function we want to optimise. In this case we choose
the 6-dimensional Hartmann function, a multi-modal function with one global
optimum.

.. code-block:: python

    from nubo.test_functions import Hartmann6D


    # test function
    func = Hartmann6D(minimise=False)
    dims = func.dims
    bounds = func.bounds

Then, we generate some initial training data. We decide to generate 5 data
points per input dimension resulting in a total of 30 data points.

.. code-block:: python

    import torch
    from nubo.utils import gen_inputs


    # training data
    x_train = gen_inputs(num_points=dims*5,
                        num_dims=dims,
                        bounds=bounds)
    y_train = func(x_train)

Now, we can prepare the Bayesian optimisation loop. We choose NUBO's
pre-defined Gaussian process that comes with a constant mean function and a
Matern 5/2 kernel. We also use the Gaussian likelihood to estimate potential
noise such as observational noise. We estimate its hyper-parameters via maximum
likelihood estimation (MLE) using the Adam optimiser. For the acquisition
function, we implement the analytical upper confidence bound (UCB) with
trade-off parameter :math:`\beta = 1.96^2` (corresponding to 95% confidence
intervals for the Gaussian distribution) and optimise it with the L-BFGS-B
algorithm using a multi-start approach with five starts. These multiple starts
ensure that the optimiser does not get stuck in a local optimum. The Bayesian
optimisation loop is run for 40 iterations giving an evaluation budget of 70.

 .. code-block:: python

    from nubo.acquisition import UpperConfidenceBound
    from nubo.models import GaussianProcess, fit_gp
    from nubo.optimisation import lbfgsb
    from gpytorch.likelihoods import GaussianLikelihood


    # Bayesian optimisation loop
    iters = 40

    for iter in range(iters):

        # specify Gaussian process
        likelihood = GaussianLikelihood()
        gp = GaussianProcess(x_train, y_train, likelihood=likelihood)

        # fit Gaussian process
        fit_gp(x_train, y_train, gp=gp, likelihood=likelihood, lr=0.1, steps=200)

        # specify acquisition function
        acq = UpperConfidenceBound(gp=gp, beta=1.96**2)

        # optimise acquisition function
        x_new, _ = lbfgsb(func=acq, bounds=bounds, num_starts=5)

        # evaluate new point
        y_new = func(x_new)

        # add to data
        x_train = torch.vstack((x_train, x_new))
        y_train = torch.hstack((y_train, y_new))

        # print new best
        if y_new > torch.max(y_train[:-1]):
            print(f"New best at evaluation {len(y_train)}: \t Inputs: {x_new.numpy().reshape(dims).round(4)}, \t Outputs: {-y_new.numpy().round(4)}")

::

    New best at evaluation 43:       Inputs: [0.18   0.147  0.     0.1909 0.3424 0.7121],    Outputs: [-2.1026]
    New best at evaluation 46:       Inputs: [0.2992 0.1852 0.     0.21   0.3398 0.6985],    Outputs: [-2.18]
    New best at evaluation 48:       Inputs: [0.2597 0.1744 0.     0.2323 0.3173 0.6544],    Outputs: [-2.337]
    New best at evaluation 50:       Inputs: [0.2486 0.1728 0.112  0.2413 0.2927 0.6674],    Outputs: [-2.6599]
    New best at evaluation 51:       Inputs: [0.234  0.1519 0.3204 0.2624 0.2972 0.6662],    Outputs: [-3.1372]
    New best at evaluation 52:       Inputs: [0.2117 0.1087 0.3731 0.313  0.3146 0.66  ],    Outputs: [-3.1906]
    New best at evaluation 54:       Inputs: [0.1698 0.1394 0.405  0.3109 0.2839 0.6623],    Outputs: [-3.1964]
    New best at evaluation 55:       Inputs: [0.1431 0.1126 0.4022 0.2795 0.3051 0.635 ],    Outputs: [-3.198]
    New best at evaluation 58:       Inputs: [0.2112 0.1557 0.4745 0.288  0.3086 0.6555],    Outputs: [-3.3158]
    New best at evaluation 64:       Inputs: [0.2013 0.1443 0.4779 0.2734 0.3131 0.6584],    Outputs: [-3.3218]

Finally, we print the overall best solution: We get -3.3218 on evaluation 66
which approximaties the true optimum of -3.3224 very well.

.. code-block:: python

    # results
    best_iter = int(torch.argmax(y_train))
    print(f"Evaluation: {best_iter+1} \t Solution: {float(y_train[best_iter]):.4f}")

::

    Evaluation: 64   Solution: 3.3218