.. _bo:

A primer on Bayesian optimisation
=================================
The following introduction aims to give a concise explanation of the Bayesian
optimisation algorithm and its element, including the surrogate model and
acquisition functions. While this introduction covers all critical details and
will be sufficient to get started with Bayesian optimisation and understand how
NUBO works, it should not be considered exhaustive. Where appropriate,
references highlight resources for additional reading that will present a more
detailed picture of Bayesian optimisation than is possible here.

.. _objfunc:

Maximisation problem
--------------------
Bayesian optimisation aims to solve the :math:`d`-dimensional maximisation
problem

.. math::
    \boldsymbol  x^* = \arg \max_{\boldsymbol  x \in \mathcal{X}} f(\boldsymbol x),

where the input space is usually continuous and bounded by a hyper-rectangle
:math:`\mathcal{X} \in [a, b]^d` with :math:`a, b \in \mathbb{R}`. The function
:math:`f(\boldsymbol x)` is most commonly a derivative-free
expensive-to-evaluate black-box function that only allows inputs
:math:`\boldsymbol x_i` to be queried and outputs :math:`y_i` to be observed
without gaining any further insights into the underlying system. We assume any
noise :math:`\epsilon` that is introduced when taking measurements to be
independent and identically distributed Gaussian noise
:math:`\epsilon \sim \mathcal{N} (0, \sigma^2)` such that
:math:`y_i = f(\boldsymbol  x_i) + \epsilon`. Hence, a set of :math:`n` pairs
of data points and corresponding observations is defined as

.. math::
    \mathcal{D_n} = \{(\boldsymbol x_i, y_i)\}_{i=1}^n

and we further define training inputs as matrix
:math:`\boldsymbol X_n = \{\boldsymbol x_i \}_{i=1}^n` and their training
outputs as vector :math:`\boldsymbol y_n = \{y_i\}_{i=1}^n`.

Many simulations and experiments in various disciplines can be formulated to
fit this description. For example, Bayesian optimisation was used in the field
of computational fluid dynamics to maximise drag reduction via active
control of blowing actuators [#Diessner2022]_ [#OConnor2023]_ [#Mahfoze2019]_,
in chemical engineering for molecular design, drug discovery, molecular
modelling, electrolyte design, and additive manufacturing [#Wang2022]_, and in
computer science to fine-tune hyper-parameters of machine learning models
[#Wu2019]_ or in architecture search for neural networks [#White2021]_.

Bayesian optimisation
---------------------
Bayesian optimisation [#Frazier2018]_ [#Gramacy2020]_ [#Jones1998]_
[#Shahriari2015]_ [#Snoek2012]_ is a surrogate model-based optimisation
algorithm that aims to maximise the objective function :math:`f(\boldsymbol x)`
in a minimum number of function evaluations. Usually, the objective function
does not have a known mathematical expression and every function evaluation is
expensive. Such problems require a cost-effective and sample-efficient
optimisation strategy. Bayesian optimisation meets these criteria by
representing the objective function through a surrogate model
:math:`\mathcal{M}`, often a Gaussian process :math:`\mathcal{GP}`. This
representation can then be used to find the next point that should be evaluated
by maximising a criterion specified through an acquisition function
:math:`\alpha (\cdot)`. A popular criterion is, the expected improvement (EI);
that is the expectation of the new point returning a better output value than
the best point to date. Bayesian optimisation is performed in a loop, where
training data :math:`\mathcal{D}_n` is used to fit the surrogate model before
the next point suggested by the acquisition function is evaluated and added to
the training data (see the algorithm below). The process then restarts and
gathers more information about the objective function with each iteration.
Bayesian optimisation is run for as many iterations as the evaluation budget
:math:`N` allows until a satisfactory solution is found, or until a predefined
stopping criterion is met.

.. admonition:: Algorithm
    :class: seealso

    Specify evaluation budget :math:`N`, number of initial points :math:`n_0`, surrogate model :math:`\mathcal{M}`, acquisition function :math:`\alpha`.

    Sample :math:`n_0` initial training data points :math:`\boldsymbol X_0` via a space-filling design [#McKay2000]_ and gather observations :math:`\boldsymbol y_0`.

    Set :math:`\mathcal{D}_n = \{ \boldsymbol X_0, \boldsymbol y_0 \}`.

    **while** :math:`n \leq N -n_0` **do:**

    1. Fit surrogate model :math:`\mathcal{M}` to training data :math:`\mathcal{D}_n`.  
    2. Find :math:`\boldsymbol x_n^*` that maximises an acquisition criterion :math:`\alpha` based on model :math:`\mathcal{M}`.  
    3. Evaluate :math:`\boldsymbol x_n^*` observing :math:`y_n^*` and add to :math:`\mathcal{D}_n`.  
    4. Increment :math:`n`.

    **end while**

    Return point :math:`\boldsymbol x^*` with highest observation.

The animation below illustrates how the Bayesian optimisation algorithm works
on an optimisation loop that runs for 20 iterations. The surrogate model uses
the available observations to provide a prediction and its uncertainty (here
shown as 95% confidence intervals around the prediction). This is our best
estimate of the underlying objective function. This estimate is then used in
the acquisition function to evaluate which point is most likely to improve over
the current best solution. Maximising the acquisition function yields the next
candidate to be observed from the objective function, before it is added to the
training data and the whole process is repeated again. The animation shows how
the surrogate model gets closer to the truth with each iteration and how the
acquisition function explores the input space by exploring regions with high
uncertainty and exploiting regions with a high prediction. This property also
called the exploration-exploitation trade-off, is a cornerstone of the
acquisition functions provided in NUBO.

.. only:: html

    .. figure:: bo.gif

        Figure 1: Bayesian optimisation of a 1D toy function with a budget of
        20 evaluations.

.. _model:

Surrogate model
---------------
A popular choice for the surrogate model :math:`\mathcal{M}` that acts as a
representation of the objective function :math:`f(\boldsymbol x)` is a Gaussian
process :math:`\mathcal{GP}` [#Gramacy2020]_ [#Williams2006]_, a flexible
non-parametric regression model. A Gaussian process is a finite collection of
random variables that has a joint Gaussian distribution and is defined by a
prior mean function
:math:`\mu_0(\boldsymbol x) : \mathcal{X} \mapsto \mathbb{R}` and a prior 
covariance kernel 
:math:`\Sigma_0(\boldsymbol x, \boldsymbol x')  : \mathcal{X} \times \mathcal{X} \mapsto \mathbb{R}`
resulting in the prior distribution

.. math::
    f(\boldsymbol X_n) \sim \mathcal{N} (m(\boldsymbol X_n), K(\boldsymbol X_n, \boldsymbol X_n)),

where :math:`m(\boldsymbol X_n)` is the mean vector of size :math:`n` over all
training inputs and :math:`K(\boldsymbol X_n, \boldsymbol X_n)` is the
:math:`n \times n` covariance matrix between all training inputs.

The posterior distribution for :math:`n_*` test points :math:`\boldsymbol X_*`
can be computed as the multivariate normal distribution conditional on some
training data :math:`\mathcal{D}_n`

.. math::
    f(\boldsymbol X_*) \mid \mathcal{D}_n, \boldsymbol X_* \sim \mathcal{N} \left(\mu_n (\boldsymbol X_*), \sigma^2_n (\boldsymbol X_*) \right)
.. math::
    \mu_n (\boldsymbol X_*) = K(\boldsymbol X_*, \boldsymbol X_n) \left[ K(\boldsymbol X_n, \boldsymbol X_n) + \sigma^2 I \right]^{-1} (\boldsymbol y - m (\boldsymbol X_n)) + m (\boldsymbol X_*)
.. math::
    \sigma^2_n (\boldsymbol X_*) = K (\boldsymbol X_*, \boldsymbol X_*) - K(\boldsymbol X_*, \boldsymbol X_n) \left[ K(\boldsymbol X_n, \boldsymbol X_n) + \sigma^2 I \right]^{-1} K(\boldsymbol X_n, \boldsymbol X_*),

where :math:`m(\boldsymbol X_*)` is the mean vector of size :math:`n_*` over
all test inputs, :math:`K(\boldsymbol X_*, \boldsymbol X_n)` is the
:math:`n_* \times n`, :math:`K(\boldsymbol X_n, \boldsymbol X_*)` is the
:math:`n \times n_*`, and :math:`K(\boldsymbol X_*, \boldsymbol X_*)` is the
:math:`n_* \times n_*` covariance matrix between training inputs
:math:`\boldsymbol X_n` and test inputs :math:`\boldsymbol X_*`.

Hyper-parameters of the Gaussian process, such as parameters :math:`\theta`
in the mean function and the covariance kernel and the noise variance
:math:`\sigma^2`, can be estimated by maximising the log marginal likelihood 
below via maximum likelihood estimation (MLE).

.. math::
    \log p(\boldsymbol y_n \mid \boldsymbol X_n) = -\frac{1}{2} (\boldsymbol y_n - m(\boldsymbol X_n))^T [K(\boldsymbol X_n, \boldsymbol X_n) + \sigma^2 I]^{-1} (\boldsymbol y_n - m(\boldsymbol X_n)) - \frac{1}{2} \log \lvert K(\boldsymbol X_n, \boldsymbol X_n) + \sigma^2 I \rvert - \frac{n}{2} \log 2 \pi

.. only:: html

    .. figure:: gp.gif

        Figure 2: Change of Gaussian process model (prediction and
        corresponding uncertainty) over 20 iterations.

NUBO uses the `GPyTorch` package [#Gardner2018]_ for surrogate modelling. This
is a very powerful package that allows the implementation of a wide selection
of models ranging from exact Gaussian processes to approximate and even deep
Gaussian processes. Besides maximum likelihood estimation (MLE) `GPyTorch` also
supports maximum a posteriori estimation (MAP) and fully Bayesian estimation to
estimate the hyper-parameter. It also comes with rich documentation, many
practical examples, and a large community.

NUBO provides a Gaussian process for off-the-shelf use with a constant mean
function and a Matern 5/2 covariance kernel that, due to its flexibility, is
especially suited for practical optimisation [#Snoek2012]_. A tutorial on how
to implement a custom Gaussian process to use with NUBO can be found in the
examples section. For more complex models we recommend consulting the
`GPyTorch` `documentation`_.

.. _acquisition:

Acquisition function
--------------------
Acquisition functions use the posterior distribution of the Gaussian process
:math:`\mathcal{GP}` to compute a criterion that assesses if a test point is a 
good potential candidate point to evaluate via the objective function
:math:`f(\boldsymbol x)`. Thus, maximising the acquisition function suggests
the test point that, based on the current training data :math:`\mathcal{D_n}`,
has the highest potential to be the global optimum. To do this, an acquisition
function :math:`\alpha (\cdot)` balances exploration and exploitation. The
former is characterised by areas with no or only a few observed data points
where the uncertainty of the Gaussian process is high, and the latter by areas
where the posterior mean of the Gaussian process is high. This
exploration-exploitation trade-off ensures that Bayesian optimisation does not
converge to the first (potentially local) maximum it encounters, but
efficiently explores the full input space.

Analytical acquisition functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NUBO supports two of the most popular acquisition functions that are, grounded
in a rich history of theoretical and empirical research. Expected improvement
(EI) [#Jones1998]_ selects points with the biggest potential to improve on
the current best observation, while upper confidence bound (UCB) 
[#Srinivas2010]_ takes an optimistic view of the posterior uncertainty and
assumes it to be true to a user-defined level. Expected improvement (EI) is
defined as

.. math::
    \alpha_{EI} (\boldsymbol X_*) = \left(\mu_n(\boldsymbol X_*) - y^{best} \right) \Phi(z) + \sigma_n(\boldsymbol X_*) \phi(z),

where :math:`z = \frac{\mu_n(\boldsymbol X_*) - y^{best}}{\sigma_n(\boldsymbol X_*)}`,
:math:`\mu_n(\cdot)` and :math:`\sigma_n(\cdot)` are the mean and the standard
deviation of the posterior distribution of the Gaussian process,
:math:`y^{best}` is the current best observation, and :math:`\Phi (\cdot)` and
:math:`\phi  (\cdot)` are the cumulative distribution function and
probability density function of the standard normal distribution.

.. only:: html

    .. figure:: bo_ei.gif
        
        Figure 3: Bayesian optimisation using the analytical expected
        improvement acquisition function of a 1D toy function with a budget of
        20 evaluations.

The upper confidence bound (UCB) acquisition function can be computed as

.. math::
    \alpha_{UCB} (\boldsymbol X_*) = \mu_n(\boldsymbol X_*) + \sqrt{\beta} \sigma_n(\boldsymbol X_*),

where :math:`\beta` is a predefined trade-off parameter, and 
:math:`\mu_n(\cdot)` and :math:`\sigma_n(\cdot)` are the mean and the standard
deviation of the posterior distribution of the Gaussian process. The animation
below shows how the acquisition would look when :math:`\beta` is set to 16. For
comparison, the posterior uncertainty shown as the 95% confidence interval
around the posterior mean of the Gaussian process is equal to using
:math:`\beta = 1.96^2`.

.. only:: html

    .. figure:: bo_ucb.gif

        Figure 4: Bayesian optimisation using the analytical upper confidence
        bound acquisition function of a 1D toy function with a budget of
        20 evaluations. 

Both of these acquisition functions can be maximised with a deterministic
optimiser, such as L-BFGS-B [#Zhu1997]_ for bounded and unconstrained problems
or SLSQP [#Kraft1994]_ for bounded constrained problems. However, this only
works for the sequential single-point problems for which every point suggested
by Bayesian optimisation is observed via the objective function
:math:`f( \boldsymbol x)` immediatley, before the optimisation loop is
repeated.

Monte Carlo acquisition functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For parallel multi-point batches or asynchronous optimisation, the analytical
acquisition functions are in general intractable. To use Bayesian optimisation
in these cases, NUBO supports the approximation of the analytical acquisition
function through Monte Carlo sampling [#Snoek2012]_ [#Wilson2018]_.

The idea is to draw a large number of samples directly from the posterior
distribution and then to approximate the acquisition functions by averaging
these Monte Carlo samples. This method is made viable by reparameterising the
acquisition functions and then computing samples from the posterior
distribution by utilising base samples from a standard normal distribution
:math:`z \sim \mathcal{N} (0, 1)`.

.. math::
    \alpha_{EI}^{MC} (\boldsymbol X_*) = \max \left(ReLU(\mu_n(\boldsymbol X_*) + \boldsymbol L \boldsymbol z - y^{best}) \right)

.. math::
    \alpha_{UCB}^{MC} (\boldsymbol X_*) = \max \left(\mu_n(\boldsymbol X_*) + \sqrt{\frac{\beta \pi}{2}} \lvert \boldsymbol L \boldsymbol z \rvert \right),

where :math:`\mu_n(\cdot)` is the mean of the posterior distribution of the
Gaussian process, :math:`\boldsymbol L` is the lower triangular matrix of the
Cholesky decomposition of the covariance matrix 
:math:`\boldsymbol L \boldsymbol L^T = K(\boldsymbol X_n, \boldsymbol X_n)`,
:math:`\boldsymbol z` are samples from the standard normal distribution
:math:`\mathcal{N} (0, 1)`, :math:`y^{best}` is the current best observation,
:math:`\beta` is the trade-off parameter, and :math:`ReLU (\cdot)` is the
rectified linear unit function that zeros all values below 0 and leaves the
rest unchanged.

Due to the randomness in the Monte Carlo samples, these acquisition functions
can only be optimised by stochastic optimisers, such as Adam [#Kingma2015]_.
However, there is some empirical evidence that fixing the base samples for
individual Bayesian optimisation loops does not affect the performance
negatively [#Balandat2020]_. This method would allow deterministic optimisers
to be used, but could potentially introduce bias due to sampling randomness.
NUBO lets you decide which variant you prefer by setting ``fix_base_samples``
and choosing the preferred optimiser. Bounded problems can be solved with Adam 
(``fix_base_samples = False``) or L-BFGS-B (``fix_base_samples = True``) and
constraint problems can be solved with SLSQP (``fix_base_samples = True``).

Furthermore, two optimisation strategies for batches are possible
[#Wilson2018]_: The default is a joint optimisation approach, where the
acquisition functions are optimised over all points of the batch
simultaneously. The second option is a greedy sequential approach where one
point after another is selected, holding all previous points fixed until the
batch is full. Empirical evidence shows that both methods approximate the
acquisition successfully. However, the greedy approach seems to have a slight
edge over the joint strategy for some examples [#Wilson2018]_. It is also
faster to compute for larger batches. At the moment, constrained optimisation
with SLSQP is only supported for the sequential strategy.

Asynchronous optimisation [#Snoek2012]_ leverages the same property as
sequential greedy optimisation: the pending points that have not yet been
evaluated can be added to the test points but are treated as fixed. In this
way, they affect the joint multivariate normal distribution but are not
considered directly in the optimisation.

.. figure:: flowchart.png
  :target: https://mikediessner.github.io/nubo/_build/html/_images/flowchart.png
  
  Figure 5: Flowchart to determine what Bayesian optimisation algorithm is recommended.
  Click to expand.

----

.. _documentation: https://docs.gpytorch.ai/en/stable

.. [#Balandat2020] M Balandat *et al.*, "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization," *Advances in Neural Information Processing Systems*, vol. 33, 2020.
.. [#Diessner2022] M Diessner, J O'Connor, A Wynn, S Laizet, Y Guan, KJ Wilson, and RD Whalley, "Investigating Bayesian Optimization for Expensive-To-Evaluate Black Box Functions: Application in Fluid Dynamics," *Frontiers in Applied Mathematics and Statistics*, 2022. 
.. [#Frazier2018] PI Frazier, "A Tutorial on Bayesian Optimization," *arXiv preprint arXiv:1807.02811*, 2018.
.. [#Gardner2018] J Gardner, G Pleiss, KQ Weinberger, D Bindel, and AG Wilson, "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with Gpu Acceleration," *Advances in Neural Information Processing Systems*, vol. 31, 2018.
.. [#Gramacy2020] RB Gramacy, *Surrogates: Gaussian Process Modeling, Design, and Optimization for the Applied Sciences*, 1st ed. Boca Raton, FL: CRC press, 2020.
.. [#Jones1998] DR Jones, M Schonlau, and WJ Welch, "Efficient Global Optimization of Expensive Black-Box Functions," *Journal of Global Optimization*, vol. 13, no. 4, p. 566, 1998.
.. [#Kingma2015] DP Kingma and J Ba, "Adam: A Method for Stochastic Optimization," *Proceedings of the 3rd International Conference on Learning Representations*, 2015.
.. [#Kraft1994] D Kraft, "Algorithm 733: TOMP-Fortran Modules for Optimal Control Calculations," *ACM Transactions on Mathematical Software (TOMS)*, vol. 20, no. 3, p. 262-281, 1994.
.. [#Mahfoze2019] OA Mahfoze, A Moody, A Wynn, RD Whalley, and S Laizet, "Reducing the Skin-Friction Drag of a Turbulent Boundary-Layer Flow with Low-Amplitude Wall-Normal Blowing within a Bayesian Optimization Framework," *Physical Review Fluids*, vol. 4, no. 9, 2019.
.. [#McKay2000] MD McKay, RJ Beckman, and WJ Conover, "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code," *Technometrics*, vol. 42, no. 1, p. 55-61, 2000.
.. [#OConnor2023] J O'Connor, M Diessner, KJ Wilson, RD Whalley, A Wynn, and S Laizet, "Optimisation and Analysis of Streamwise-Varying Wall-Normal Blowing in a Turbulent Boundary Layer," *Flow, Turbulence and Combustion*, 2023.
.. [#Shahriari2015] B Shahriari, K Swersky, Z Wang, RP Adams, and N De Freitas, "Taking the Human Out of the Loop: A Review of Bayesian Optimization," *Proceedings of the IEEE*, vol. 104, no. 1, p. 148-175, 2015.
.. [#Snoek2012] J Snoek, H Larochelle, and RP Adams, "Practical Bayesian Optimization of Machine Learning Algorithms," *Advances in Neural Information Processing Systems*, vol. 25, 2012.
.. [#Srinivas2010] N Srinivas, A Krause, SM Kakade, and M Seeger, "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design," *Proceedings of the 27th International Conference on Machine Learning*, p. 1015-1022, 2010.
.. [#Wang2022] K Wang and AW Dowling, "Bayesian optimization for chemical products and functional materials," *Current Opinion in Chemical Engineering*, vol. 36, 2022.
.. [#White2021] C White, W Neiswanger, and Y Savani, "Bananas: Bayesian Optimization with Neural Architectures for Neural Architecture Search," *Proceedings of the Aaai Conference on Artificial Intelligence*, vol. 35, no. 12, 2021.
.. [#Williams2006] CKI Williams, and CE Rasmussen, *Gaussian Processes for Machine Learning*, 2nd ed. Cambridge, MA: MIT press, 2006.
.. [#Wilson2018] J Wilson, F Hutter, and M Deisenroth, "Maximizing Acquisition Functions for Bayesian Optimization," *Advances in Neural Information Processing Systems*, vol. 31, 2018.
.. [#Wu2019] J Wu, XY Chen, H Zhang, LD Xiong, H Lei, and SH Deng, "Hyperparameter Optimization for Machine Learning Models Based on Bayesian Optimization," *Journal of Electronic Science and Technology*, vol. 17, no. 1, p. 26-40, 2019.
.. [#Zhu1997] C Zhu, RH Byrd, P Lu, J Nocedal, "Algorithm 778: L-BFGS-B: Fortran Subroutines for Large-Scale Bound-Constrained Optimization," *ACM Transactions on Mathematical Software (TOMS)*, vol. 23, no. 4, p. 550-560, 1997.
