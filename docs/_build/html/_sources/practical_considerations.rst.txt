Practical considerations
========================

General
-------
**How many initial data points do I need?**
    A rule of thumb for Gaussian process models is to have at least 10 points
    per input dimension. However, empirical evidence shows that reducing this
    to 5 or even 1 point(s) per input dimensions does not result in worse
    solutions for Bayesian optimisation [1]_.

Gaussian process
----------------
**What prior mean function and prior covariance kernel should I use?**
    Zero or constant mean function with a Matern 5/2 kernel [3]_.

**What likelihood should I specify?**
    For exact Gaussian processes ``GPyTorch`` provides two main options that
    differ with regards to the handling of the observational noise
    :math:`\sigma^2`. The ``GaussianLikelihood`` estimates the observation
    noise while the ``FixedNoiseGaussianLikelihood`` holds them fixed. If
    you cannot measure the observational noise, the former likelihood is
    recommended. If you have a clear idea of the observational noise the latter
    can also be used. Here, you can decide if you want the Gaussian process to
    also estimate any additional noise besides the observational noise [2]_.

Acquisition function
--------------------
**What acquisition function should I use?**
    NUBO supports two acquisition functions: Expected improvement (EI) and
    upper confidence bound (UCB). While both are widely-used options that have
    proven to give good results, there is empirical evidence that UCB performs
    better on a wider range of synthetic test functions [1]_.

**Should I use analytical or Monte Carlo acquisition functions?**
    We recommend using analytical acquisition functions for sequential
    single-point optimisation problems. Where it is advantageous to evaluate
    potential solutions in parallel Monte Carlo acquisition functions allow the
    computation of batches. Furhtermore, if you want to continue the
    optimisation loop while some potential solutions are still being evaluated,
    Monte Carlo acquisition functions enable asynchronous optimisation [3]_ 
    [4]_.

**What optimiser should I choose?**
    We recommend L-BFGS-B for analytical acquisition functions and SLSQP for
    constrained analytical acquisition functions. For Monte Carlo acquisition
    functions, the stochastic optimiser Adam should be used if the base samples
    are resampled. If the base samples are fixed, deterministic optimisers can
    be used in the same way as for the analytical acquisition functions.

----

.. [1] M Diessner, J O’Connor, A Wynn, S Laizet, Y Guan, K Wilson, and RD Whalley, "Investigating Bayesian optimization for expensive-to-evaluate black box functions: Application in fluid dynamics," *Frontiers in Applied Mathematics and Statistics*, 2022. 
.. [2] RB Gramacy, and HKH Lee, "Cases for the nugget in modeling computer experiments," *Statistics and computing*, vol. 22, p. 713-722, 2012.
.. [3] J Snoek, H Larochelle, and RP Adams, "Practical Bayesian optimization of machine learning algorithms," *Advances in neural information processing systems*, vol. 25, 2012.
.. [4] J Wilson, F Hutter, and M Deisenroth, "Maximizing acquisition functions for Bayesian optimization," *Advances in neural information processing systems*, vol. 31, 2018.
