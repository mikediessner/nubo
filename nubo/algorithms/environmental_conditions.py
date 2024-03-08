import torch
import numpy as np
from nubo.utils import normalise, unnormalise, standardise
from nubo.models import GaussianProcess, fit_gp
from nubo.acquisition import ExpectedImprovement
from nubo.optimisation import gen_candidates
from gpytorch.likelihoods import GaussianLikelihood
from copy import deepcopy
from scipy.optimize import minimize


from typing import Callable, List, Optional, Tuple, Any


def envbo(x_train: torch.Tensor,
          y_train: torch.Tensor,
          env_dims: int | List[int], 
          env_values: float | List[float],
          bounds: torch.Tensor,
          constraints: Optional[dict | Tuple[dict]]=(),
          normalise_x: Optional[bool]=False,
          standardise_y: Optional[bool]=False,
          num_starts: Optional[int]=10,
          num_samples: Optional[int]=100) -> torch.Tensor:
    """
    ENVBO is a Bayesian optimisation algorithm for problems with uncontrollable
    variables that are given externally by environmental conditions. This
    function represents a single optimisation step that needs to be wrapped in
    a loop where new measurements of the environmental variables are provided
    at each iteration. Assumes a maximisation problem.

    Parameters
    ----------
    x_train : ``torch.Tensor``
        (size n x d) Training inputs.
    y_train : ``torch.Tensor``
        (size n) Training outputs.
    env_dims : ``int`` or ``List`` of ``int``
        List of indices of environmental variables.
    env_values : ``float`` or ``List`` of ``float``
        List of values of environmental variables.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    constraints : ``dict`` or ``Tuple`` of ``dict``, optional
        Optimisation constraints on inputs, default is no constraints.
    normalise_x: bool, optional
        Whether inputs should be normalised before optimisation, default is
        False.
    standardise_y: bool, optional
        Whether outpus should be standardised before optimisation, default is
        False
    num_starts : ``int``, optional
        Number of start for multi-start optimisation, default is 10.
    num_samples : ``int``, optional
        Number of samples from which to draw the starts, default is 100.

    Returns
    -------
    ``torch.Tensor``
        (size 1 x d) New candidate inputs.
    """

    # get number of parameters
    dims = bounds.size((1))

    opt_bounds = deepcopy(bounds)
    
    # normalise inputs
    if normalise_x:
        x_train = normalise(x_train, bounds)
        opt_bounds = torch.tensor([[0,]*dims, [1,]*dims])
        env_values = normalise(torch.tensor([env_values]), bounds[:, env_dims]).reshape(-1).tolist()

    # standardise outputs
    if standardise_y:
        y_train = standardise(y_train)

    # OPTIMISATION STEP
    # specify Gaussian process
    likelihood = GaussianLikelihood()
    gp = GaussianProcess(x_train, y_train, likelihood=likelihood)

    # fit Gaussian process
    fit_gp(x_train, y_train,
            gp=gp,
            likelihood=likelihood,
            lr=0.1,
            steps=200)

    # specify acquisition function
    acq = ExpectedImprovement(gp=gp, y_best=torch.max(y_train))

    # optimise acquisition function conditional on environmental conditions
    x_new, ei = _cond_optim(func=acq,
                            env_dims=env_dims,
                            env_values=env_values,
                            bounds=opt_bounds,
                            constraints=constraints,
                            num_starts=num_starts,
                            num_samples=num_samples)

    # unnormalise new point
    if normalise_x:
        x_new = unnormalise(x_new, bounds)

    return x_new


def _cond_optim(func: Callable,
                env_dims: int | List[int],
                env_values: float | List[float],
                bounds: torch.Tensor,
                constraints: Optional[dict | Tuple[dict]]=(),
                num_starts: Optional[int]=10,
                num_samples: Optional[int]=100,
                **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Conditional optimisation for environmental conditions. Holds envrionmental
    variables with indices `env_dims` fixed at measuremens `env_values`.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    env_dims : ``List`` of ``int``
        List of indices of environmental variables.
    env_values : ``List`` of ``float``
        List of values of environmental variables.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    constraints : ``dict`` or ``Tuple`` of ``dict``
        Optimisation constraints on inputs, default is no constraints.
    num_starts : ``int``, optional
        Number of start for multi-start optimisation, default is 10.
    num_samples : ``int``, optional
        Number of samples from which to draw the starts, default is 100.

    Returns
    -------
    best_result : ``torch.Tensor``
        (size 1 x d) Minimiser inputs.
    best_func_result : ``torch.Tensor``
        (size 1) Minimiser output.
    """

    # make sure env_dims and env_values are lists
    if not isinstance(env_dims, (int, list)):
        raise TypeError("env_dims must be an int or a list of ints.")
    if isinstance(env_dims, int):
        env_dims = [env_dims,]
    
    if not isinstance(env_values, (int, float, list)):
        raise TypeError("env_values must be an int, float or a list of ints/floats.")
    if isinstance(env_values, (int, float)):
        env_values = [env_values,]

    # make sure constraints have the correct type
    if not isinstance(constraints, (dict, tuple)):
        raise TypeError("Constraints must be dict or a tuple of dicts.")
    if isinstance(constraints, dict):
        constraints = [constraints,]
    if isinstance(constraints, tuple):
        constraints = list(constraints)

    # add environmental conditions to constraints
    def create_con(dim, value):
        return {"type": "eq", "fun": lambda x: x[dim] - value}

    for i in range(len(env_dims)):
        constraints.append(create_con(env_dims[i], env_values[i]))

    # optimise
    best_results, best_func_result = _slsqp(func=func,
                                            env_dims=env_dims,
                                            env_values=env_values,
                                            bounds=bounds,
                                            constraints=constraints,
                                            num_starts=num_starts,
                                            num_samples=num_samples,
                                            **kwargs)

    return best_results, best_func_result


def _slsqp(func: Callable,
           env_dims: List[int],
           env_values: List[float],
           bounds: torch.Tensor,
           constraints: Optional[dict | Tuple[dict]]=(),
           num_starts: Optional[int]=10,
           num_samples: Optional[int]=100,
           **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Special version of the multi-start SLSQP optimiser for optimisation with
    environmental conditions using the ``scipy.optimize.minimize``
    implementation from ``SciPy``.
    
    Used for optimising the acquisition function within the ENVBO algorithm.
    Picks the best `num_starts` points from a total `num_samples` Latin
    hypercube samples to initialise the optimser. Returns the best result.
    Minimises `func`.

    Parameters
    ----------
    func : ``Callable``
        Function to optimise.
    env_dims : ``List`` of ``int``
        List of indices of environmental variables.
    env_values : ``List`` of ``float``
        List of values of environmental variables.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    constraints : ``dict`` or ``Tuple`` of ``dict``, optional
        Optimisation constraints, default is no constraints.
    num_starts : ``int``, optional
        Number of start for multi-start optimisation, default is 10.
    num_samples : ``int``, optional
        Number of samples from which to draw the starts, default is 100.
    **kwargs : ``Any``
        Keyword argument passed to ``scipy.optimize.minimize``.
    
    Returns
    -------
    best_result : ``torch.Tensor``
        (size 1 x d) Minimiser inputs.
    best_func_result : ``torch.Tensor``
        (size 1) Minimiser output.
    """

    dims = bounds.size(1)

    # restrict bounds of environmental variables for candidate sampling in SLSQP
    env_bounds = deepcopy(bounds.float())
    for i in range(len(env_dims)):
        env_bounds[0, env_dims[i]] = env_values[i]
        env_bounds[1, env_dims[i]] = env_values[i]
    
    # generate candidates
    candidates = gen_candidates(func, env_bounds, num_starts, num_samples)
    candidates = candidates.numpy()
    
    # initialise objects for results
    results = torch.zeros((num_starts, dims))
    func_results = torch.zeros(num_starts)
    
    # iteratively optimise over candidates
    for i in range(num_starts):
        result = minimize(func,
                          x0=candidates[i],
                          method="SLSQP",
                          bounds=bounds.numpy().T,
                          constraints=constraints,
                          **kwargs)
        results[i, :] = torch.from_numpy(result["x"].reshape(1, -1))
        func_results[i] = float(result["fun"])
    
    # select best candidate
    best_i = torch.argmin(func_results)
    best_result =  torch.reshape(results[best_i, :], (1, -1))
    best_func_result = torch.reshape(func_results[best_i], (1,))

    return best_result, best_func_result


class ENVBOPredictionModel:
    """
    Prediction model for ENVBO algorithm.
    
    Predicts optimal values for controllable parameters conditional on
    measurements of environmental variables. First, it uses a Gaussian process
    to model the relationship between all controllable and environmental inputs
    and the outputs. Second, it predicts optimal input values of the
    controllable parameters by holding environmental inputs fixed to the
    provided measurements and maximising the posterior mean of the Gaussian
    process model. Assumes a maximisation problem.

    Attributes
    ----------
    x_train : ``torch.Tensor``
        (size n x d) Training inputs.
    y_train : ``torch.Tensor``
        (size n) Training outputs.
    env_dims : ``List`` of ``int``
        List of indices of environmental variables.
    env_values : ``List`` of ``float``
        List of values of environmental variables.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    constraints : ``dict`` or ``Tuple`` of ``dict``, optional
        Optimisation constraints, default is no constraints.
    likelihood : ``gpytorch.likelihoods.Likelihood``
        Likelihood.
    model : ``gpytorch.models.GP``
        Gaussian Process model.
    """

    def __init__(self,
                 x_train : torch.Tensor,
                 y_train : torch.Tensor,
                 env_dims : int | List[int],
                 bounds : torch.Tensor,
                 constraints : Optional[dict | Tuple[dict]]=()) -> None:
        """
        Initialise Gaussian process as prediction model.

        Parameters
        ----------
        x_train : ``torch.Tensor``
            (size n x d) Training inputs.
        y_train : ``torch.Tensor``
            (size n) Training outputs.
        env_dims : ``int`` or ``List`` of ``int``
            List of indices of environmental variables.
        env_values : ``List`` of ``float``
            List of values of environmental variables.
        bounds : ``torch.Tensor``
            (size 2 x d) Optimisation bounds of input space.
        constraints : ``dict`` or ``Tuple`` of ``dict``, optional
            Optimisation constraints, default is no constraints.
        """

        self.x_train = x_train
        self.y_train = y_train
        self.env_dims = env_dims
        self.bounds = bounds
        self.constraints = constraints

        self.likelihood = GaussianLikelihood()
        self.model = GaussianProcess(self.x_train, self.y_train, self.likelihood)
        fit_gp(self.x_train, self.y_train, self.model, self.likelihood)
   

    def predict(self,
                env_values : float | List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict values for controllable parameters conditional on measurements
        of environmental variables.

        Parameters
        ----------
        env_values : ``float`` or ``List`` of ``float``
            List of values of environmental variables.

        Returns
        -------
        x_new : ``torch.Tensor``
            (size 1 x d) Minimiser inputs conditional on environmental values.
        pred_new : ``torch.Tensor``
            (size 1) Predicted minimiser output conditional on environmental
            values.
        """

        self.model.eval()

        # Objective function for predictions
        def predict(x):

            x = x.reshape((1, -1))
            numpy = False

            if isinstance(x, np.ndarray):
                numpy = True
                x = torch.from_numpy(x)

            pred = self.model(x)
            mean = pred.mean.detach()

            if numpy:
                mean = mean.numpy()

            return -mean

        # Optimise
        x_new, pred_new = _cond_optim(func=predict,
                                      env_dims=self.env_dims,
                                      env_values=env_values,
                                      bounds=self.bounds,
                                      constraints=self.constraints,
                                      num_starts=10,
                                      num_samples=200)
        
        return x_new, -pred_new
