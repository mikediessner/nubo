.. _custom_gp:

Custom Gaussian process
=======================
This notebook gives a introduction to specifying custom Gaussian process with
GPyTorch that can be used with NUBO.

Define Gaussian process
-----------------------
A Gaussian process is defined by its mean function and its covariance kernel.
Both are specified in the ``__init__()`` method of the ``GaussianProcess``
class below and can be easily swapped out by the desired function and kernel.
While ``GPyTorch`` offers many different options, the most common choices are
the zero mean or constant mean function and the Matern or RBF kernel. Some
kernels, such as the Matern and the RBF kernel, are only defined for a certain
range so that they need to be scaled through the ``ScaleKernel`` to be used
with all problems. The length-scale parameters of the covariance kernel can
either be represented as a single length-scale or as one length-scale parameter
for each input dimensions. The latter is known as automatic relevance
determination (ARD) and allows inputs to be differently correlated. The
``forward()`` method takes a test point and returns the predictive multivariate
normal distribution. All other properties of the Gaussian process are inherited
by the ExactGP making it easy to implement custom Gaussian processes. For more
information about Gaussian processes and about options for mean function and
covariance kernel see the documentation of ``GPyTorch``.

.. code-block:: python

    from torch import Tensor
    from gpytorch.models import ExactGP
    from gpytorch.means import ZeroMean, ConstantMean
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.likelihoods import Likelihood


    class GaussianProcess(ExactGP):

        def __init__(self,
                    x_train: Tensor,
                    y_train: Tensor,
                    likelihood: Likelihood) -> None:

            # initialise ExactGP
            super(GaussianProcess, self).__init__(x_train, y_train, likelihood)

            # specify mean function and covariance kernel
            self.mean_module = ZeroMean()
            self.covar_module = ScaleKernel(
                base_kernel=RBFKernel(ard_num_dims=x_train.shape[-1])
            )

        def forward(self, x: Tensor) -> MultivariateNormal:

            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)

            return MultivariateNormal(mean_x, covar_x)

Generate training data
----------------------
To use the Gaussian process, we first generate some training data.

.. code-block:: python

    from nubo.test_functions import Hartmann6D
    from nubo.utils import gen_inputs


    # test function
    func = Hartmann6D(minimise=False)
    dims = func.dims
    bounds = func.bounds

    # training data
    x_train = gen_inputs(num_points=dims*5,
                        num_dims=dims,
                        bounds=bounds)
    y_train = func(x_train)

Fit Gaussian process
--------------------
Before we fit the Gaussian process to the training data, we first have to
decide on the likelihood that should be used. There are two likelihoods we want
to consider here: First, we have the standard Gaussian likelihood. This
likelihood assumes a constant homoskedastic observation noise and estimates the
noise parameter :math:`\sigma^2` from the data. Second, there is the Gaussian
likelihood with fixed noise. You want to use this option when you know or can
measure the observation noise of your objective function. In this case, you can
still decide if you want to estimate the additional noise besides the
observations noise or not. This example continues with the full estimation of
the noise level. NUBO has the convenience function ``fit_gp`` that maximises
the log marginal likelihood with maximum likelihood estimation (MLE) using
``torch``'s Adam optimiser.

.. code-block:: python

    from nubo.models import fit_gp
        from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood


    # initialise Gaussian process
    likelihood = GaussianLikelihood()
    gp = GaussianProcess(x_train, y_train, likelihood=likelihood)

    # fit Gaussian process
    fit_gp(x_train, y_train, gp=gp, likelihood=likelihood, lr=0.1, steps=200)

Make predictions for test points
--------------------------------
With the fitted Gaussian process in hand, we can easily predict the mean and
the variance of previously unobserved test points. Below, we sample five points
randomly and print the predictive mean and variance that define the predictive
distribution for each test point based on the training data and our Gaussian
process specified above.

.. code-block:: python

    import torch


    # sample test point
    x_test = torch.rand((5, dims))

    # set Gaussian Process to eval mode
    gp.eval()

    # make predictions
    pred = gp(x_test)

    # predictive mean and variance
    mean = pred.mean
    variance = pred.variance.clamp_min(1e-10)

    print(f"Mean: {mean.detach()}")
    print(f"Variance: {variance.detach()}")

::

    Mean: tensor([ 0.2188,  0.1616, -0.0127,  0.0252, -0.0069], dtype=torch.float64)
    Variance: tensor([0.0136, 0.0191, 0.0252, 0.0164, 0.0343], dtype=torch.float64)
