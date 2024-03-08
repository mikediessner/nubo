.. _get_started:

Get started
===========
This brief introduction will teach you in detail how to install NUBO from the
`GitHub repository`_ and how to set up a Bayesian optimisation loop to maximise
a toy function using NUBO's predefined Gaussian process as the surrogate model.
You can also use one of our off-the-shelf algorithm to get started quickly. For
more details see the `Off-the-shelf algorithms` section in the menu on the left.

Installing NUBO
---------------
Install NUBO and all its dependencies directly from the `Python Package Index`_
*PyPI* using the `Python package manager`_ *pip* with the following code. We
recommend the use of a virtual environment.

::

    pip install nubopy


Optimising a toy function with NUBO
-----------------------------------
First, we set up the toy function we want to optimise. In this case, we choose
the 6-dimensional Hartmann function, a multi-modal function with one global
optimum. This synthetic test function acts as a substitute for a black-box
objective function, such as an experiment or a simulation. The ``bounds`` of
the input space are defined as a two-dimensional ``torch.Tensor`` where the
first row gives the lower bounds for all input dimensions and the second row
gives the corresponding upper bounds.

.. code-block:: python

    import torch
    from nubo.test_functions import Hartmann6D


    # test function
    func = Hartmann6D(minimise=False)
    dims = 6

    # specify bounds
    bounds = torch.tensor([[0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 1., 1.]])

Then, we generate some initial training data. We decide to generate 5 data
points per input dimension resulting in a total of 30 data points.

.. code-block:: python

    from nubo.utils import gen_inputs


    # training data
    x_train = gen_inputs(num_points=dims*5,
                         num_dims=dims,
                         bounds=bounds)
    y_train = func(x_train)

In NUBO, training inputs ``x_train`` should be a two-dimensional
``torch.Tensor`` (a matrix), where the rows are individual points and the
columns are individual dimensions. In this example, our training data has size
30 x 6. The training outputs ``y_train`` should be a one-dimensional
``torch.Tensor`` (a vector) with one entry for each training input (here
``y_train`` has size 30).

Now we can prepare the Bayesian optimisation loop. We choose NUBO's predefined
Gaussian process that by default has a constant mean function and a Matern 5/2
kernel. We also use the Gaussian likelihood to estimate observational noise. We
estimate the Gaussian processes hyper-parameters via maximum likelihood
estimation (MLE) using the Adam optimiser. For the acquisition function, we
implement the analytical upper confidence bound (UCB) with a trade-off
parameter :math:`\beta = 1.96^2` (corresponding to 95% confidence intervals for
the Gaussian distribution) and optimise it with the L-BFGS-B algorithm using a
multi-start approach with five starts. These multiple starts help to ensure
that the optimiser does not get stuck in a local optimum. The Bayesian
optimisation loop is run for 40 iterations, giving a total evaluation budget of
70.

 .. code-block:: python

    from nubo.acquisition import UpperConfidenceBound
    from nubo.models import GaussianProcess, fit_gp
    from nubo.optimisation import single
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
        x_new, _ = single(func=acq, method="L-BFGS-B", bounds=bounds, num_starts=5)

        # evaluate new point
        y_new = func(x_new)

        # add to data
        x_train = torch.vstack((x_train, x_new))
        y_train = torch.hstack((y_train, y_new))

        # print new best
        if y_new > torch.max(y_train[:-1]):
            print(f"New best at evaluation {len(y_train)}: \t Inputs: {x_new.numpy().reshape(dims).round(4)}, \t Outputs: {-y_new.numpy().round(4)}")

::

    New best at evaluation 31: 	 Inputs: [0.477  0.0444 0.0736 0.2914 0.3603 0.7323], 	 Outputs: [-1.9494]
    New best at evaluation 34: 	 Inputs: [0.4453 0.0418 0.0483 0.3164 0.3478 0.6925], 	 Outputs: [-2.0684]
    New best at evaluation 39: 	 Inputs: [0.4127 0.1638 0.     0.277  0.3385 0.679 ], 	 Outputs: [-2.1595]
    New best at evaluation 40: 	 Inputs: [0.3715 0.1565 0.     0.3261 0.3372 0.7126], 	 Outputs: [-2.1843]
    New best at evaluation 41: 	 Inputs: [0.3589 0.134  0.3895 0.2927 0.3222 0.7003], 	 Outputs: [-2.9809]
    New best at evaluation 42: 	 Inputs: [0.2754 0.1478 0.425  0.2529 0.3054 0.6874], 	 Outputs: [-3.2027]
    New best at evaluation 46: 	 Inputs: [0.1473 0.1864 0.427  0.2906 0.2993 0.666 ], 	 Outputs: [-3.2302]
    New best at evaluation 51: 	 Inputs: [0.1764 0.1303 0.4576 0.3022 0.3029 0.6827], 	 Outputs: [-3.2657]
    New best at evaluation 52: 	 Inputs: [0.2016 0.1447 0.4616 0.2798 0.3018 0.6716], 	 Outputs: [-3.31]
    New best at evaluation 53: 	 Inputs: [0.2063 0.144  0.465  0.2787 0.3138 0.6519], 	 Outputs: [-3.3192]
    New best at evaluation 58: 	 Inputs: [0.205  0.1516 0.4686 0.2725 0.3137 0.6614], 	 Outputs: [-3.3206]
    New best at evaluation 66: 	 Inputs: [0.2096 0.142  0.4767 0.2757 0.3112 0.6573], 	 Outputs: [-3.3209]
    New best at evaluation 70: 	 Inputs: [0.2076 0.1527 0.4728 0.2802 0.3109 0.6594], 	 Outputs: [-3.321]

Finally, we print the overall best solution: we get -3.3210 on evaluation 70,
which approximates the true optimum of -3.3224.

.. code-block:: python

    # results
    best_iter = int(torch.argmax(y_train))
    print(f"Evaluation: {best_iter+1} \t Solution: {-float(y_train[best_iter]):.4f}")

::

    Evaluation: 70 	 Solution: -3.3210

The estimated parameters of the Gaussian process can be viewed as follows:

.. code-block:: python

    # estimated parameters
    print(f"Mean function constant: {gp.mean_module.constant.item()}")
    print(f"Covariance kernel output-scale: {gp.covar_module.outputscale.item()}")
    print(f"Covariance kernel length-scale: {gp.covar_module.base_kernel.lengthscale.detach()}")
    print(f"Estimated noise/nugget: {likelihood.noise.item()}")

::

    Mean function constant: 0.1073
    Covariance kernel output-scale: 0.2943
    Covariance kernel length-scale: tensor([[0.5552, 0.5305, 0.6730, 0.3610, 0.2741, 0.3786]])
    Estimated noise/nugget: 0.0001


.. _`GitHub repository`: https://github.com/mikediessner/nubo/
.. _`Python Package Index`: https://pypi.org/
.. _`Python package manager`: https://pip.pypa.io/en/latest/
