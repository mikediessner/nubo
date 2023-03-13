Overview
========
NUBO, short for Newcastle University Bayesian optimisation, is a Bayesian
optimisation framework for the optimisation of expensive-to-evaluate black box
functions, such as physical experiments and computer simulations, developed by
the `experimental fluid dynamics research group`_ at `Newcastle University`_.
It focuses on transparent implementations of algorithms and precise references
and documentation for all methods to make Bayesian optimisation accessible for
researchers from all disciplines. NUBO is written in Python_ and distributed
open-source under the `BSD 3-Clause license`_.

Bayesian optimisation
---------------------
:ref:`Bayesian optimisation <bo>` [1]_ [3]_ [4]_ [7]_ [8]_ is a surrogate model-based optimisation 
algorithm that aims to maximise an :ref:`objective function <objfunc>` in a
minimum number of function evaluations. Usually, the objective function does
not have a known mathematical expression and every function evaluation is
expensive requiring a cost-effective and sample-efficient optimisation
routine. Bayesian optimisation meets these criteria by representing the
objective function through a :ref:`surrogate model <model>`, often a Gaussian
process [3]_ [11]_. This representation can then be used to find the next point that
should be evaluated by maximising a criterion specified through an
:ref:`acquisition function <acquisition>`. A popular criterion is, for
example, the Expected Improvement [4]_ that is the expectation of the new point
returning a better solution than the current best. Bayesian optimisation is
performed in a loop where training data is used to fit the surrogate model
before the next point suggested by the acquisition function is evaluated and
added to the training data itself. The loop than restarts gaining more
information about the objective function with each iteration. Bayesian
optimisation is run for as many iterations as the evaluation budget allows,
until a satisfying solution is found, or unitl a pre-defined stopping
criterion is met.

Contents
--------
- **Surrogate models**: Gaussian Processes [3]_ [11]_ are specified through the
  ``GPyTorch`` package [2]_, a powerful package that allows the implementation
  of a wide selection of models ranging from exact Gaussian processes to
  approximate and even deep Gaussian processes. Hyper-parameters can be
  estimated via maximum likelihood estimation (MLE), maximum a posteriori
  estimation (MAP) or fully Bayesian estimation.
- **Acquisition function**: NUBO supports the use of analytical acquisition
  functions and approximations through Monte Carlo sampling. Analytical
  Expected Improvement [4]_ and Upper Confidence Bound [10]_ can be used for
  sequential   single-point problems where results are evaluated after each
  iteration. Multi-point batches for parallel evaluation or asynchronous
  problems where the optimnisation algorithm is continued while other points
  are still being evaluated can be performed via Monte Carlo acquisition
  functions [12]_.
- **Optimisers**: The deterministic analytical acquisition functions are
  optimised via multi-start L-BFGS-B if the input space is restricted by box
  bounds or multi-start SLSQP if the input space is also restricted by
  constraints. The stochastic Monte Carlo acquisition functions that are based
  on random samples are maximised with the stochastic optimiser Adam [5]_.  
- **Design of experiments**: Initial data points can be generated with space-
  filling experimental designs. NUBO supports random or maximin Latin
  hypercube designs [6]_.
- **Synthetic test functions**: NUBO provides ten synthetic test functions
  that allow validating Bayesian optimisation algorithms before applying them
  to expensive experiments [9]_.

----

.. _`experimental fluid dynamics research group`: https://www.experimental-fluid-dynamics.com/
.. _`Newcastle University`: https://www.ncl.ac.uk/
.. _Python: https://www.python.org/
.. _`BSD 3-Clause license`: https://joinup.ec.europa.eu/licence/bsd-3-clause-new-or-revised-license/

.. [1] Frazier, Peter I. "A tutorial on Bayesian optimization." *arXiv preprint arXiv:1807.02811* (2018).
.. [2] Gardner, Jacob, Geoff Pleiss, Kilian Q. Weinberger, David Bindel, and Andrew G. Wilson. "GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration." *Advances in neural information processing systems* 31 (2018).
.. [3] Gramacy, Robert B. *Surrogates: Gaussian process modeling, design, and optimization for the applied sciences.* CRC press, 2020.
.. [4] Jones, Donald R., Matthias Schonlau, and William J. Welch. "Efficient global optimization of expensive black-box functions." *Journal of Global optimization* 13, no. 4 (1998): 455.
.. [5] Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." *Proceedings of the 3rd International Conference on Learning Representations* (2015).
.. [6] McKay, Michael D., Richard J. Beckman, and William J. Conover. "A comparison of three methods for selecting values of input variables in the analysis of output from a computer code." Technometrics 42, no. 1 (2000): 55-61.
.. [7] Shahriari, Bobak, Kevin Swersky, Ziyu Wang, Ryan P. Adams, and Nando De Freitas. "Taking the human out of the loop: A review of Bayesian optimization." *Proceedings of the IEEE* 104, no. 1 (2015): 148-175.
.. [8] Snoek, Jasper, Hugo Larochelle, and Ryan P. Adams. "Practical bayesian optimization of machine learning algorithms." *Advances in neural information processing systems* 25 (2012).
.. [9] Surjanovic, Sonja, and Derek Bingham. "Virtual library of simulation experiments: Test functions and datasets." Sfu. March 11, 2023. https://www.sfu.ca/~ssurjano/optimization.html/.
.. [10] Srinivas, Niranjan, Andreas Krause, Sham M. Kakade, and Matthias Seeger. "Gaussian process optimization in the bandit setting: No regret and experimental design."" *Proceedings of the 27th International Conference on Machine Learning* (2010): 1015-1022.
.. [11] Williams, Christopher KI, and Carl Edward Rasmussen. *Gaussian processes for machine learning.* Vol. 2, no. 3. Cambridge, MA: MIT press, 2006.
.. [12] Wilson, James, Frank Hutter, and Marc Deisenroth. "Maximizing acquisition functions for Bayesian optimization." *Advances in neural information processing systems* 31 (2018).
