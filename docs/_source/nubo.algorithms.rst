Algorithms module
=================

This module provides some implementations of algorithms that address a specific
problem or challenge for easy, off-the-shelf use.


Optimisation
------------

.. admonition:: References
   :class: seealso

   For expected improvement with noisy observations see:
   - RB Gramacy, *Surrogates: Gaussian Process Modeling, Design, and Optimization for the Applied Sciences*, 1st ed. Boca Raton, FL: CRC press, 2020.


.. automodule:: nubo.algorithms.optimise
   :members: optimise
   :undoc-members:
   :show-inheritance:


Environmental conditions
------------------------

.. admonition:: References
   :class: seealso

   - M Diessner, KJ Wilson, and RD Whalley, "On the development of a practical Bayesian optimisation algorithm for expensive experiments and simulations with changing environmental conditions," *arXiv preprint arXiv:2402.03006*, 2024.

.. automodule:: nubo.algorithms.environmental_conditions
   :members: envbo, ENVBOPredictionModel
   :undoc-members:
   :show-inheritance:
