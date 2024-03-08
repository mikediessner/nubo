Practical considerations
========================
Below are some considerations that should be taken into account when
deciding on the design of the Bayesian optimisation loop with NUBO. This
section features some of the most common questions about Bayesian optimisation
and NUBO and is frequently updated.

General
-------
**How many initial data points do I need?**
    A rule of thumb for Gaussian process models is to have at least 10 points
    per input dimension [#Baker2021]_ [#Domingo2019]_ [#Owen2017]_. However,
    empirical evidence shows that reducing this to 5 or even 1 point(s) per
    input dimension does not result in worse solutions for Bayesian
    optimisation [#Diessner2022]_.

**How does NUBO optimise a mixed parameter space with continuous and discrete variables?**
    NUBO supports the optimisation over a mixed parameter space by fixing a
    combination of the discrete inputs and optimising over the remaining
    continuous inputs. The best point found over all possible discrete
    combinations is used. While this avoids issues due to rounding, it can be
    time-consuming for many discrete dimensions and possible values.

Gaussian process
----------------
**What prior mean function and prior covariance kernel should I use?**
    For practical Bayesian optimisation, a zero or constant mean function with
    a Matern 5/2 kernel is recommended [#Snoek2012]_. Other kernels, such as
    the RBF kernel, might be too smooth to be able to represent realistic
    experiments and simulations.

**What likelihood should I specify?**
    For exact Gaussian processes, `GPyTorch` provides two main options that
    differ with regards to their computation of the observational noise
    :math:`\sigma^2`: The ``GaussianLikelihood`` estimates the observation
    noise while the ``FixedNoiseGaussianLikelihood`` holds it fixed. If
    you cannot measure the observational noise, the former likelihood is
    recommended. If you have a clear idea of the observational noise the latter
    can also be used. Then, you can decide if you want the Gaussian process to
    also estimate any additional noise besides the observational noise
    [#Gramacy2012]_.

Acquisition function
--------------------
**Which acquisition function should I use?**
    NUBO supports two acquisition functions: Expected improvement (EI)
    [#Jones1998]_ and upper confidence bound (UCB) [#Srinivas2010]_. While both
    are widely-used options that have proven to give good results, there is
    empirical evidence that UCB performs better on a wider range of synthetic
    test functions [#Diessner2022]_.

**Should I use analytical or Monte Carlo acquisition functions?**
    We recommend using analytical acquisition functions for sequential
    single-point optimisation problems. Where it is advantageous to evaluate
    potential solutions in parallel, Monte Carlo acquisition functions allow
    the computation of batches. Furthermore, if you want to continue the
    optimisation loop while some potential solutions are still being evaluated,
    Monte Carlo acquisition functions enable asynchronous optimisation
    [#Snoek2012]_ [#Wilson2018]_.

**Which optimiser should I choose?**
    We recommend L-BFGS-B [#Zhu1997]_ for analytical acquisition functions and
    SLSQP [#Kraft1994]_ for constrained analytical acquisition functions. For
    Monte Carlo acquisition functions, the stochastic optimiser Adam
    [#Kingma2015]_ should be used if the base samples are resampled. If you
    decide to fix the base samples, deterministic optimisers can be used in the
    same way as for the analytical acquisition functions. While fixing the base
    samples could introduce some sampling bias, there is empirical evidence
    that it does not affect performance negatively [#Balandat2020]_.

----

.. [#Baker2021] E Baker, "Emulation of Stochastic Computer Models with an Application to Building Design," Ph.D. dissertation, Department of Mathematics, Univ. Exeter, Exeter, 2021.
.. [#Balandat2020] M Balandat *et al.*, "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization," *Advances in Neural Information Processing Systems*, vol. 33, 2020.
.. [#Diessner2022] M Diessner, J O'Connor, A Wynn, S Laizet, Y Guan, KJ Wilson, and RD Whalley, "Investigating Bayesian Optimization for Expensive-To-Evaluate Black Box Functions: Application in Fluid Dynamics," *Frontiers in Applied Mathematics and Statistics*, 2022. 
.. [#Domingo2019] D Domingo, "Gaussian Process Emulation: Theory and Applications to the Problem of Past Climate Reconstruction," Ph.D. dissertation, School of Mathematics, Univ. Leeds, Leeds, 2019.
.. [#Gramacy2012] RB Gramacy, and HKH Lee, “Cases for the Nugget in Modeling Computer Experiments,” *Statistics and Computing*, vol. 22, p. 713-722, 2012.
.. [#Jones1998] DR Jones, M Schonlau, and WJ Welch, "Efficient Global Optimization of Expensive Black-Box Functions," *Journal of Global Optimization*, vol. 13, no. 4, p. 566, 1998.
.. [#Kingma2015] DP Kingma and J Ba, "Adam: A Method for Stochastic Optimization," *Proceedings of the 3rd International Conference on Learning Representations*, 2015.
.. [#Kraft1994] D Kraft, "Algorithm 733: TOMP-Fortran Modules for Optimal Control Calculations," *ACM Transactions on Mathematical Software (TOMS)*, vol. 20, no. 3, p. 262-281, 1994.
.. [#Owen2017] NE Owen, "A Comparison of Polynomial Chaos and Gaussian Process Emulation for Uncertainty Quantification in Computer Experiments," Ph.D. dissertation, Department of Mathematics, Univ. Exeter, Exeter, 2017.
.. [#Snoek2012] J Snoek, H Larochelle, and RP Adams, "Practical Bayesian Optimization of Machine Learning Algorithms," *Advances in Neural Information Processing Systems*, vol. 25, 2012.
.. [#Srinivas2010] N Srinivas, A Krause, SM Kakade, and M Seeger, "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design," *Proceedings of the 27th International Conference on Machine Learning*, p. 1015-1022, 2010.
.. [#Wilson2018] J Wilson, F Hutter, and M Deisenroth, "Maximizing Acquisition Functions for Bayesian Optimization," *Advances in Neural Information Processing Systems*, vol. 31, 2018.
.. [#Zhu1997] C Zhu, RH Byrd, P Lu, J Nocedal, "Algorithm 778: L-BFGS-B: Fortran Subroutines for Large-Scale Bound-Constrained Optimization," *ACM Transactions on Mathematical Software (TOMS)*, vol. 23, no. 4, p. 550-560, 1997.
