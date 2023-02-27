from setuptools import setup, find_packages


VERSION = "0.1.0"
DESCRIPTION = "Bayesian optimisation framework (Newcastle University)"
LONG_DESCRIPTION = """Bayesian optimisation framework including Gaussian
                    process regression models, acquisition functions,
                    optimisation algorithms and benchmark functions. Developed
                    as part of project INTEL at Newcastle University."""

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
        install_requires=["torch", "gpytorch", "numpy", "scipy", "matplotlib"],
        keywords=["Optimisation", "Bayesian optimisation", "Gaussian process",
                  "Acquisition function", "black box functions"],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
        ]
)
