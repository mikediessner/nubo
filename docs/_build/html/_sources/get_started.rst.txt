.. _get_started:

Get started
===========
This brief introduction will teach you how to install NUBO from the GitHub
repository and how to set up a Bayesian optimisation loop to maximise a toy
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
optimum. This synthetic test function acts as a surrogate for a black box
objective function, such as an experiment or a simulation.

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

In NUBO, training inputs ``x_train`` should be a two-dimensional
``torch.Tensor`` (a matrix) where the rows are individual points and the
columns are individual dimensions. In this example, our training data has size
30 x 6. The training outputs ``y_train`` should be a one-dimensional
``torch.Tensor`` (a vector) with one entry for each training input (here
``y_train`` has size 30). The ``bounds`` of the input space are defined as a
two-dimensional ``torch.Tensor`` where the first row gives the lower bounds for
all input dimensions and the second row gives the corresponding upper bounds.
In the example above, the bounds are given as an attribute of the
``Hartmann6D`` class. The code snippet below shows how you could set up the
bounds manually.

.. code-block:: python

    # specify bounds manually
    bounds = torch.tensor([[0., 0., 0., 0., 0., 0.],
                           [1., 1., 1., 1., 1., 1.]])

Now, we can prepare the Bayesian optimisation loop. We choose NUBO's
pre-defined Gaussian process that comes with a constant mean function and a
Matern 5/2 kernel. We also use the Gaussian likelihood to estimate
observational noise. We estimate its hyper-parameters via maximum likelihood
estimation (MLE) using the Adam optimiser. For the acquisition function, we
implement the analytical upper confidence bound (UCB) with trade-off parameter
:math:`\beta = 1.96^2` (corresponding to 95% confidence intervals for the
Gaussian distribution) and optimise it with the L-BFGS-B algorithm using a
multi-start approach with five starts. These multiple starts ensure that the
optimiser does not get stuck in a local optimum. The Bayesian optimisation loop
is run for 40 iterations giving an evaluation budget of 70.

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

    New best at evaluation 43: 	 Inputs: [0.3949 1.     1.     0.7699 0.0393 0.0369], 	 Outputs: [-1.9498]
    New best at evaluation 52: 	 Inputs: [0.2581 0.3436 0.5644 0.2322 0.3715 0.8276], 	 Outputs: [-2.1738]
    New best at evaluation 56: 	 Inputs: [0.4257 1.     1.     0.6889 0.094  0.003 ], 	 Outputs: [-2.4506]
    New best at evaluation 59: 	 Inputs: [0.2707 0.2744 0.5454 0.2384 0.3474 0.7427], 	 Outputs: [-2.8153]
    New best at evaluation 60: 	 Inputs: [0.3071 0.2052 0.4839 0.265  0.3319 0.6998], 	 Outputs: [-3.091]
    New best at evaluation 69: 	 Inputs: [0.2485 0.1512 0.4608 0.291  0.3224 0.6786], 	 Outputs: [-3.2718]

Finally, we print the overall best solution: We get -3.2718 on evaluation 69
which approximaties the true optimum of -3.3224.

.. code-block:: python

    # results
    best_iter = int(torch.argmax(y_train))
    print(f"Evaluation: {best_iter+1} \t Solution: {-float(y_train[best_iter]):.4f}")

::

    Evaluation: 69 	 Solution: 3.2718

The estimated parameters of the Gaussian process can be viewed as follows:

.. code-block:: python

    # estimated parameters
    print(f"Mean function constant: {gp.mean_module.constant}")
    print(f"Covariance kernel output-scale: {gp.covar_module.outputscale}")
    print(f"Covariance kernel length-scale: {gp.covar_module.base_kernel.lengthscale}")
    print(f"Estimated noise/nugget: {likelihood.noise}")

::

    Mean function constant: 0.1855
    Covariance kernel output-scale: 0.3659
    Covariance kernel length-scale: tensor([[0.3780, 0.4826, 0.6710, 0.3035, 0.3445, 0.3133]])
    Estimated noise/nugget: 0.0009
