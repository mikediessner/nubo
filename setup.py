from setuptools import setup, find_packages


VERSION = "1.0.0"
DESCRIPTION = "Bayesian optimisation framework"
LONG_DESCRIPTION = """Bayesian optimisation framework for the optimisation of
                      expensive-to-evaluate black box functions."""

# Setting up
setup(
    name="nubo",
    version=VERSION,
    author="Mike Diessner",
    author_email="m.diessner2@newcastle.ac.uk>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/mikediessner/nubo",
    packages=find_packages(),
    install_requires=["torch", "gpytorch", "scipy", "numpy"],
    keywords=["Optimisation", "Bayesian optimisation", "Gaussian process",
              "Acquisition function", "black box functions"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ]
)
