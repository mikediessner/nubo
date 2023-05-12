# NUBO

NUBO, short for Newcastle University Bayesian optimisation, is a Bayesian
optimisation framework for the optimisation of expensive-to-evaluate black-box
functions, such as physical experiments and computer simulations. It is
developed and maintained by the
[Fluid Dynamics Lab](https://www.experimental-fluid-dynamics.com) at
[Newcastle University](https://www.ncl.ac.uk). NUBO focuses primarily on
transparency and user experience to make Bayesian optimisation easily
accessible to researchers from all disciplines. Transparency is ensured by
clean and comprehensible code, precise references, and thorough documentation.
User experience is ensured by a modular and flexible design, easy-to-write
syntax, and careful selection of Bayesian optimisation algorithms. NUBO allows
you to tailor Bayesian optimisation to your specific problem by writing the
optimisation loop yourself using the provided building blocks. Only algorithms
and methods that are sufficiently tested and validated to perform well are
included in NUBO. This ensures that the package remains compact and does not
overwhelm the user with an unnecessary large number of options. The package is
written in [Python](https://www.python.org) but does not require expert
knowledge of Python to optimise your simulations and experiments. NUBO is
distributed as an open-source software under the
[BSD 3-Clause licence](https://joinup.ec.europa.eu/licence/bsd-3-clause-new-or-revised-license).

 > Thanks for considering NUBO. If you have any questions, comments, or issues
 > feel free to email us at m.diessner2@newcastle.ac.uk. Any feedback is highly
 > appreciated and will help make NUBO better in the future.

## Install NUBO

Install NUBO and all its dependencies directly from the
[Python Package Index](https://pypi.org) *PyPI* using the
[Python package manager](https://pip.pypa.io/en/latest/) *pip* with the
following code. We recommend the use of a virtual environment.S

    pip install nubopy

## Cite NUBO

If you are using NUBO for your research, please cite as:

    Mike Diessner, Kevin Wilson, and Richard D. Whalley. "NUBO: A Transparent Python Package for Bayesian Optimisation," arXiv preprint arXiv:2305.06709, 2023.

If you are using Bibtex, please cite as:

```
@article{diessner2023nubo,
         title={NUBO: A Transparent Python Package for Bayesian Optimisation},
         author={Diessner, Mike and Wilson, Kevin and Whalley, Richard D},
         journal={arXiv preprint arXiv:2305.06709},
         year={2023}
}
```
