NUBO: A transparent Python package for Bayesian optimisation
============================================================
NUBO, short for Newcastle University Bayesian optimisation, is a Bayesian
optimisation framework for the optimisation of expensive-to-evaluate black box
functions, such as physical experiments and computer simulations. It is
developed and maintained by the `Fluid Dynamics Lab`_ at
`Newcastle University`_. NUBO focuses primarly on transparency and user
experience to make Bayesian optimisation easily accessible to researchers from
all disciplines. Transparency is ensured by clean and comprehensible code,
precise references, and thorough documentation. User experience is guaranteed
by a modular and flexible design, easy-to-write syntax, and careful selection
of Bayesian optimisation algorithms. NUBO allows you to tailor Bayesian 
optimisation to your specific problem by writing the optimisation loop yourself
using the provided building blocks. Only algorithms and methods that are
sufficiently tested and proven to perform well are included in NUBO. This
ensures that the package remains compact and does not overwhelm with an
unnecessary large number of options. The package is written in Python_ but does
not require expert knowledge to optimise your simulations and experiments. NUBO
is distributed as an open-source software under the `BSD 3-Clause license`_.

.. admonition:: Contact
   :class: seealso

   Thanks for considering NUBO. If you have any questions, comments, or issues
   feel free to email us at m.diessner2@newcastle.ac.uk. Any feedback is highly
   appreciated and will help make NUBO better in the future.

On this page you can find an overview of the three main documentation sections
consisting of (i) an introduction to Bayesian optimisation with NUBO, (ii) a
selection of examples that can be used as boilerplate code, and (iii) detailed
references for all of NUBO's functions and objects.

NUBO
----
The NUBO section contains general information about the package, gives a
concise introduction to :ref:`Bayesian optimisation <bo>`, and explains its
components, such as the :ref:`surrogate model <model>` and the
:ref:`acquisition functions <acquisition>`. It also provides a
:ref:`quickstart guide <get_started>` to NUBO allowing you to start optimising
your simulations and experiments in minutes. This is the place to start your
Bayesian optimisation journey with NUBO.

.. toctree::
   :maxdepth: 1
   :caption: NUBO:

   overview.rst
   get_started.rst
   bayesian_optimisation.rst
   practical_considerations.rst
   citation.rst
   GitHub <http://github.com/mikediessner/nubo>

Examples
--------
The Examples section provides guides to some problems that NUBO is capable
of optimising, and shows how to implement :ref:`custom surrogate models <custom_gp>`.
This boilerplate code is a good starting point when tailoring a Bayesian
optimisation algorithm to your specfic problem.

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   custom_gp.rst
   singlepoint.ipynb
   multipoint_joint.ipynb
   multipoint_sequential.ipynb
   multipoint_fixed.ipynb
   asynchronous_bo.ipynb
   constrained_bo.ipynb
   mixed_parameters.ipynb
   fixed_noise.ipynb

Package reference
-----------------
The Package reference section gives detailed documentation to all of NUBO's
functionality. This is where you want to go first when you are not sure how 
a specific object or function should be used.

.. toctree::
   :maxdepth: 1
   :caption: Package reference:

   nubo.acquisition.rst
   nubo.models.rst
   nubo.optimisation.rst
   nubo.test_functions.rst
   nubo.utils.rst

.. _`Fluid Dynamics Lab`: https://www.experimental-fluid-dynamics.com
.. _`Newcastle University`: https://www.ncl.ac.uk
.. _Python: https://www.python.org
.. _`BSD 3-Clause license`: https://github.com/mikediessner/nubo/blob/main/LICENSE.md
