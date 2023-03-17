.. _bo:

A primer on Bayesian optimisation
=================================

.. _objfunc:

Maximisation problem
--------------------
Bayesian optimisation aims to solve the $d$-dimensional maximisation problem

.. math::
    \boldsymbol  x^* = \arg \max_{\boldsymbol  x \in \mathcal{X}} f(\boldsymbol x)

where the input space is usually continuous and bounded by a hyper-rectangle
:math:`\mathcal{X} \in [a, b]^d` with :math:`a, b \in \mathbb{R}`. The function
:math:`f(\boldsymbol x)` is most commonly a derivative-free
expensive-to-evaluate black box problem that only allows inputs
:math:`\boldsymbol x_i` to be querried and outputs :math:`y_i` to be observed
without gaining any further insights into the underlying system. We assume any
noise :math:`\epsilon` that is introduced when taking measurements to be
independent and identically distributed Gaussian noise
:math:`\epsilon \sim \mathcal{N} (0, \sigma^2)` such that
:math:`y_i = f(\boldsymbol  x_i) + \epsilon`. Hence, a set of $n$ pairs of data
points and corresponding observations is defined as

.. math::
    \mathcal{D_n} = \{(\boldsymbol x_i, y_i)\}_{i=1}^n

and we further define training inputs as matrix
:math:`\boldsymbol X_n = \{\boldsymbol x_i \}_{i=1}^n` and their training
outputs as vector :math:`\boldsymbol y_n = \{y_i\}_{i=1}^n`.

Bayesian optimisation
---------------------
Bayesian optimisation [2]_ [4]_ [5]_ [8]_ [9]_ is a surrogate model-based
optimisation algorithm that aims to maximise the objective function
:math:`f(\boldsymbol x)` in a minimum number of function evaluations. Usually,
the objective function does not have a known mathematical expression and every
function evaluation is expensive requiring a cost-effective and
sample-efficient optimisation routine. Bayesian optimisation meets these
criteria by representing the objective function through a surrogate model
:math:`\mathcal{M}`, often a Gaussian process :math:`\mathcal{GP}`. This
representation can then be used to find the next point that should be evaluated
by maximising a criterion specified through an acquisition function
:math:`\alpha`. A popular criterion is, for example, the expected improvement
(EI) that is the expectation of the new point returning a better solution than
the previous best. Bayesian optimisation is performed in a loop where training
data :math:`\mathcal{D}_n` is used to fit the surrogate model before the next
point suggested by the acquisition function is evaluated and added to the
training data itself. The loop than restarts gaining more information about the
objective function with each iteration. Bayesian optimisation is run for as
many iterations as the evaluation budget $N$ allows, until a satisfying
solution is found, or unitl a pre-defined stopping criterion is met.

.. admonition:: Algorithm
    :class: seealso

    Specify evaluation budget :math:`N`, number of initial points :math:`n_0`, surrogate model :math:`\mathcal{M}`, acquisition function :math:`\alpha`.

    Sample :math:`n_0` initial training data points :math:`\boldsymbol X_0` via a space-filling design [7]_ and gather observations :math:`\boldsymbol y_0`.

    Set :math:`n = n_0` and :math:`\mathcal{D}_n = \{ \boldsymbol X_0, \boldsymbol y_0 \}`.

    **while** :math:`n \leq N` **do:**

    1. Fit surrogate model :math:`\mathcal{M}` to training data :math:`\mathcal{D}_n`.  
    2. Find :math:`\boldsymbol x_n^*` that maximises an acquisition criterion :math:`\alpha` based on model :math:`\mathcal{M}`.  
    3. Evaluate :math:`\boldsymbol x_n^*` observing :math:`y_n^*` and add to :math:`\mathcal{D}_n`.  
    4. Increment :math:`n`.

    **end while**

    Return point :math:`\boldsymbol x^*` with highest observation.

.. _model:

Surrogate model
---------------
A popular choice for the surrogate model :math:`\mathcal{M}` that acts as a
representation of the objective function :math:`f(\boldsymbol x)` is a Gaussian
process :math:`\mathcal{GP}` [4]_ [11]_, a flexible non-parametric regression
model. A Gaussian process is a finite collection of random variables that has a
joint Gaussian distribution and is defined by a mean function
:math:`\mu_0(\boldsymbol x) : \mathcal{X} \mapsto \mathbb{R}` and a covariance
kernel :math:`\Sigma_0(\boldsymbol x, \boldsymbol x')  : \mathcal{X} \times \mathcal{X} \mapsto \mathbb{R}`
resulting in the prior distribution

.. math::
    f(\boldsymbol X_n) \sim \mathcal{N} (m(\boldsymbol X_n), K(\boldsymbol X_n, \boldsymbol X_n)).

where :math:`m(\boldsymbol X_n)` is the mean vector of size $n$ over all
training inputs and :math:`K(\boldsymbol X_n, \boldsymbol X_n)` is the
:math:`n \times n` covariance matrix between all training inputs.

The posterior or predictive distribution for :math:`n_*` test points
:math:`\boldsymbol X_*` can be computed as the multivariate normal distribution
conditional on some training data :math:`\mathcal{D}_n`

.. math::
    f(\boldsymbol X_*) \mid \mathcal{D}_n, \boldsymbol X_* \sim \mathcal{N} \left(\mu_n (\boldsymbol X_*), \sigma^2_n (\boldsymbol X_*) \right)
.. math::
    \mu_n (\boldsymbol X_*) = K(\boldsymbol X_*, \boldsymbol X_n) \left[ K(\boldsymbol X_n, \boldsymbol X_n) + \sigma^2 I \right]^{-1} (\boldsymbol y - m (\boldsymbol X_n)) + m (\boldsymbol X_*)
.. math::
    \sigma^2_n (\boldsymbol X_*) = K (\boldsymbol X_*, \boldsymbol X_*) - K(\boldsymbol X_*, \boldsymbol X_n) \left[ K(\boldsymbol X_n, \boldsymbol X_n) + \sigma^2 I \right]^{-1} K(\boldsymbol X_n, \boldsymbol X_*)

where :math:`m(\boldsymbol X_*)` is the mean vector of size :math:`n_*` over
all test inputs, :math:`K(\boldsymbol X_*, \boldsymbol X_n)` is the
:math:`n_* \times n`, :math:`K(\boldsymbol X_n, \boldsymbol X_*)` is the
:math:`n \times n_*`, and :math:`K(\boldsymbol X_*, \boldsymbol X_*)` is the
:math:`n_* \times n_*` covariance matrix between training inputs
:math:`\boldsymbol X_n` and test inputs :math:`\boldsymbol X_*`.

Hyper-parameters of the Gaussian process such as any parameters :math:`\theta`
in the mean function and the covariance kernel or the noise variance
:math:`\sigma^2` can be estimated by maximising the log marginal likelihood
via maximum likelihood estimation (MLE):

.. math::
    \log p(\boldsymbol y_n \mid \boldsymbol X_n) = -\frac{1}{2} (\boldsymbol y_n - m(\boldsymbol X_n))^T [K(\boldsymbol X_n, \boldsymbol X_n) + \sigma^2 I]^{-1} (\boldsymbol y_n - m(\boldsymbol X_n)) - \frac{1}{2} \log \lvert K(\boldsymbol X_n, \boldsymbol X_n) + \sigma^2 I \rvert - \frac{n}{2} \log 2 \pi

NUBO uses the ``GPyTorch`` package [3]_ for surrogate modelling. This is a very
powerful package that allows the implementation of a wide selection of models
ranging from exact Gaussian processes to approximate and even deep Gaussian
processes. Besides maximum likelihood estimation (MLE) ``GPyTorch`` also
supports maximum a posteriori estimation (MAP) and fully Bayesian estimation
to estimate the hyper-parameter. It also comes with a rich documentation, many
practical examples, and a large community.

NUBO provides a Gaussian process for
off-the-shelf use with a constant mean function and a Matern 5/2 covariance
kernel that due to its flexibility is especially suited for practical
optimisation [9]_. A tutorial on how to implement a custom Gaussian process for
NUBO can be found in the examples section. For more complex models we recommend
consulting the ``GPyTorch`` `documentation`_.

.. _acquisition:

Acquisition function
--------------------
Acquisition functions use the posterior or predictive distribution of the
Gaussian process :math:`\mathcal{GP}` to compute a criterion that assess if a
test point is good potential solution when evaluated through the objective
function :math:`f(\boldsymbol x)`. Thus, maximising the acquisition function
suggests the test point that based on the current training data
:math:`\mathcal{D_n}` has the highest potential of being the global optimum. To
do this, an acquisition function :math:`\alpha` balances exploration and
exploitation. The former characterised by areas that lack of observed data
points and where the uncertainty of the Gaussian process is high, and the
latter by promising areas with a high posterior mean of the Gaussian process.
This exploration-exploitation trade-off ensures that Bayesian optimisation does
not converge to the first (potentially local) maximum it finds but explores the
full input space.

Analytical acquisition functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NUBO supports two of the most popular acquisition functions that are grounded
in a rich history of theoretical and empirical research. Expected improvement
(EI) [5]_ selects points with the biggest potential of improving on the current
best observation while upper confidence bound (UCB) [10]_ takes an optimistic
view of the posterior uncertainty and assumes a user-defined (through the
hyper-parameter :math:`\beta`) level of it to be true. Expected improvement
(EI) is defined as

.. math::
    \alpha_{EI} (\boldsymbol X_*) = \left(\mu_n(\boldsymbol X_*) - y^{best} \right) \Phi(z) + \sigma_n(\boldsymbol X_*) \phi(z)

where :math:`z = \frac{\mu_n(\boldsymbol X_*) - y^{best}}{\sigma_n(\boldsymbol X_*)}`,
:math:`\mu_n(\cdot)` and :math:`\sigma_n(\cdot)` are the mean and the standard
deviation of the predictive distribution of the Gaussian process, $y^{best}$ is
the current best observation, and :math:`\Phi` and :math:`\phi` are the
cumulative distribution function and the probability density function of the
standard normal distribution.

The upper confidence bound (UCB) can be computed by

.. math::
    \alpha_{UCB} (\boldsymbol X_*) = \mu_n(\boldsymbol X_*) + \sqrt{\beta} \sigma_n(\boldsymbol X_*)

where :math:`\beta` is a pre-defined trade-off parameter, and
:math:`\mu_n(\cdot)` and :math:`\sigma_n(\cdot)` are the mean and the standard
deviation of the predictive distribution of the Gaussian process.

Both of these acquisition functions can be computed analytically by maximising
them with a deterministic optimiser such as L-BFGS-B for bounded unconstraint
problems or SLSQP for bounded or constraint problems. However, this is only
true for the sequential single-point case in which every points suggested by
Bayesian optimisation is observed through the objective function
:math:`f( \boldsymbol x)` immediatley before the optimisation loop is repeated.

Monte Carlo acquisition functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For parallel multi-point batches or asynchronous optimisation, the analytical
acquisition functions are in general intractable. To allow Bayesian
optimisation in these cases, NUBO supports the approximation of the analytical
acquisition function through Monte Carlo sampling [9]_ [12]_.

The idea is to draw a large number of samples directly from the predicitve
distribution and then to approximate the acquisition by averaging these Monte
Carlo samples. This method is made viable by reparameterising the acquisition
functions and then computing samples from the predicitve distribution by
utilising base samples from a standard normal distribution
:math:`z \sim \mathcal{N} (0, 1)`.

.. math::
    \alpha_{EI}^{MC} (\boldsymbol X_*) = \max \left(ReLU(\mu_n(\boldsymbol X_*) + \boldsymbol L \boldsymbol z - y^{best}) \right)

.. math::
    \alpha_{UCB}^{MC} (\boldsymbol X_*) = \max \left(\mu_n(\boldsymbol X_*) + \sqrt{\frac{\beta \pi}{2}} \lvert \boldsymbol L \boldsymbol z \rvert \right)

where :math:`\mu_n(\cdot)` is the mean of the predictive distribution of the
Gaussian process, :math:`\boldsymbol L` is the lower triangular matrix of the
Cholesky decomposition of the covariance matrix 
:math:`\boldsymbol L \boldsymbol L^T = K(\boldsymbol X_n, \boldsymbol X_n)`,
:math:`\boldsymbol z` are samples from the standard normal distribution,
:math:`y^{best}` is the current best observation, :math:`\beta` is the
trade-off parameter, and :math:`ReLU (\cdot)` is the rectified linear unit
function that zeros all values below $0$ and leaves the rest as is.

Due to the randomness of the Monte Carlo samples, these acquisition functions
can only be optimised by stochastic optimisers such as Adam [6]_. However,
there is some empirical evidence that fixing the base samples for individual
Bayesian optimisation loops does not affect the performance negatively [1]_.
This method would allow deterministic optimiser to be used but could
potentially introduce bias due to sampling randomness.

Furthermore, two optimisation strategies for batches are possible [12]_: The
default is a joint optimisation approach where the acquisition functions are
optimised over all points of the batch. The second option is a greedy
sequential approach where one point after the other is selected holding each
previous point fixed until the batch is full. Empirical evidence shows that
both methods approximate the acquisition successfully. However, the greedy
approach seems to have a slight edge over the joint strategy for some examples
[12]_. It also is faster to compute for larger batches.

Asynchronous optimisation [9]_ leverages the same property as sequential greedy
optimisation: the pending points that have not yet been evaluated can be added
to the test points but are treated as fixed. In this way, they affect the joint
multivariate normal distribution but are not considered directly in the
optimisation.

.. image:: unnamed.png
    :width: 49 %
.. image:: unnamed-2.png
    :width: 49 %
.. image:: unnamed-3.png
    :width: 49 %
.. image:: unnamed-4.png
    :width: 49 %

Figure 1: Bayesian optimisation example. A Gaussian process is fitted to three
initial observations (dark blue dots) resulting in the posterior mean (solid
red line) and the posterior variance represented here as the 95% confidence
interval (blue area). The expected improvement (EI) acquisition function
(orange area) is maximised to find the next point that should be observed
(dashed black line) from the objective function. Once observed, the input and
output are added to the training data and the process is repeated two more
times. The final Gaussian process model is than compared to the true objective
function (solid black line). The last evaluated point approximates the
maximum.

----

.. _documentation: https://docs.gpytorch.ai/en/stable

.. [1] M Balandat *et al.*, "BoTorch: A framework for efficient Monte-CarloBayesian optimization," *Advances in neural information processing systems*, vol. 33, 2020.
.. [2] PI Frazier, "A tutorial on Bayesian optimization," *arXiv preprint arXiv:1807.02811*, 2018.
.. [3] J Gardner, G Pleiss, KQ Weinberger, D Bindel, and AG Wilson, "GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration," *Advances in neural information processing systems*, vol. 31, 2018.
.. [4] RB Gramacy, *Surrogates: Gaussian process modeling, design, and optimization for the applied sciences*, 1st ed. Boca Raton, FL: CRC press, 2020.
.. [5] DR Jones, M Schonlau, and WJ Welch, "Efficient global optimization of expensive black-box functions," *Journal of global optimization*, vol. 13, no. 4, p. 566, 1998.
.. [6] DP Kingma and J Ba, "Adam: A method for stochastic optimization," *Proceedings of the 3rd international conference on learning representations*, 2015.
.. [7] MD McKay, RJ Beckman, and WJ Conover, "A comparison of three methods for selecting values of input variables in the analysis of output from a computer code," *Technometrics*, vol. 42, no. 1, p. 55-61, 2000.
.. [8] B Shahriari, K Swersky, Z Wang, RP Adams, and N De Freitas, "Taking the human out of the loop: A review of Bayesian optimization," *Proceedings of the IEEE*, vol. 104, no. 1, p. 148-175, 2015.
.. [9] J Snoek, H Larochelle, and RP Adams, "Practical Bayesian optimization of machine learning algorithms," *Advances in neural information processing systems*, vol. 25, 2012.
.. [10] N Srinivas, A Krause, SM Kakade, and M Seeger, "Gaussian process optimization in the bandit setting: No regret and experimental design," *Proceedings of the 27th international conference on machine learning*, p. 1015-1022, 2010.
.. [11] CKI Williams, and CE Rasmussen, *Gaussian processes for machine learning*, 2nd ed. Cambridge, MA: MIT press, 2006.
.. [12] J Wilson, F Hutter, and M Deisenroth, "Maximizing acquisition functions for Bayesian optimization," *Advances in neural information processing systems*, vol. 31, 2018.
