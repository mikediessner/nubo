.. NUBO documentation master file, created by
   sphinx-quickstart on Sat Mar  4 14:32:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NUBO: a transparent python package for Bayesian Optimisation
============================================================
NUBO, short for Newcastle University Bayesian Optimisation, is a Bayesian
Optimisation framework for the optimisation of expensive-to-evaluate black box
functions such as physical experiments and computer simulations developed by
the `experimental fluid dynamics research group <https://www.experimental-fluid-dynamics.com>`_
at `Newcastle University <https://www.ncl.ac.uk>`_. It focuses on transparent
and comprehensible implementations of algorithms and provides extensive
explanations and references for all methods to make Bayesian Optimisation
accessible for researchers from all disciplines. 

NUBO
--------
The NUBO section contains general information about the package and gives an
in-depth explanation of Bayesian Optimisation and its components such as the
surrogate models and the acquisition functions. It also gives a quickstart
guide to NUBO. This is the place to start your Bayesian Optimisation journey
with NUBO.

.. toctree::
   :maxdepth: 1
   :caption: NUBO:

   overview
   get_started.ipynb
   bayesian_optimisation
   surrogate_models
   acquisition_functions
   citation
   Github <http://github.com/mikediessner/nubo-dev>

Examples
--------
The examples section provides guides to some problems that NUBO is capable
of optimising. This boilerplate code is a good place to start from when
implementing Bayesian Optimisation for specfic problems.

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
functionality. This is the place to go when you are not sure how a specific
object or function should be used.

.. toctree::
   :maxdepth: 1
   :caption: Package reference:

   nubo.acquisition
   nubo.models
   nubo.optimisation
   nubo.test_functions
   nubo.utils
