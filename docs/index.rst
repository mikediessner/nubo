NUBO: a transparent python package for Bayesian optimisation
============================================================
NUBO, short for Newcastle University Bayesian optimisation, is a Bayesian
optimisation framework for the optimisation of expensive-to-evaluate black box
functions, such as physical experiments and computer simulations, developed by
the `experimental fluid dynamics research group`_ at `Newcastle University`_.
It focuses on transparent implementations of algorithms and precise references
and documentation for all methods to make Bayesian optimisation accessible for
researchers from all disciplines.  NUBO is written in Python_ and distributed
open-source under the `BSD 3-Clause license`_.

NUBO
----
The NUBO section contains general information about the package and gives an
in-depth explanation of Bayesian optimisation and its components such as the
surrogate models and the acquisition functions. It also gives a quickstart
guide to NUBO so you can start optimising your simulations and experiments in
minutes. This is the place to start your Bayesian optimisation journey with
NUBO.

.. toctree::
   :maxdepth: 1
   :caption: NUBO:

   overview
   get_started.ipynb
   bayesian_optimisation
   citation
   Github <http://github.com/mikediessner/nubo-dev>

Examples
--------
The examples section provides guides to some problems that NUBO is capable
of optimising. This boilerplate code is a good starting point when tailoring a
Bayesian optimisation algorithm to your specfic problem.

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   singlepoint.ipynb
   multipoint_joint.ipynb
   multipoint_sequential.ipynb
   asynchronous_bo.ipynb
   constrained_bo.ipynb
   bayesian_gp.ipynb

Package reference
-----------------
The package reference section gives detailed documentation to all of NUBO's
functionality. This is where you want to go first when you are not sure how 
a specific object or function should be used.

.. toctree::
   :maxdepth: 1
   :caption: Package reference:

   nubo.acquisition
   nubo.models
   nubo.optimisation
   nubo.test_functions
   nubo.utils

.. _`experimental fluid dynamics research group`: https://www.experimental-fluid-dynamics.com
.. _`Newcastle University`: https://www.ncl.ac.uk
.. _Python: https://www.python.org
.. _`BSD 3-Clause license`: https://joinup.ec.europa.eu/licence/bsd-3-clause-new-or-revised-license
