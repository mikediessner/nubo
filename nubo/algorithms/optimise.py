import torch
import numpy as np
from nubo.utils import normalise, unnormalise, standardise
from nubo.models import GaussianProcess, fit_gp
from nubo.acquisition import ExpectedImprovement, UpperConfidenceBound, MCExpectedImprovement, MCUpperConfidenceBound
from nubo.optimisation import single, multi_sequential
from gpytorch.likelihoods import GaussianLikelihood
from copy import deepcopy

from typing import Optional, Tuple


def optimise(x_train: torch.Tensor,
             y_train: torch.Tensor,
             bounds: torch.Tensor,
             batch_size: Optional[int]=1,
             acquisition: Optional[str]="EI",
             beta: Optional[float]=4.0,
             constraints: Optional[dict | Tuple[dict]]=[],
             discrete: Optional[dict]=None,
             noisy: Optional[bool]=False,
             x_pending: Optional[torch.Tensor]=None,
             mc_samples: Optional[int]=128,
             normalise_x: Optional[bool]=False,
             standardise_y: Optional[bool]=False,
             num_starts: Optional[int]=10,
             num_samples: Optional[int]=100) -> torch.Tensor:
    """
    Off-the-shelf optimisation step. Allows optimisation of expensive
    experiments and simulations with single-point and multi-point candidates,
    input constraints, continuous and discrete parameters, noisy observations,
    and pending training data points. You can select between expected
    improvement and upper confidence bound acquisition functions.

    Uses analytical acquisition functions with the L-BFGS-B optimiser for
    single-point optimisation and Monte Carlo acquisition functions with the
    Adam optimiser for multi-point and asynchronous optimisation. Fixes base
    samples when input constraints are provided for the latter and optimises all
    constrained problems via the SLSQP optimiser.

    For expected improvement with noisy observaions, the maximum of the Gaussian
    process posterior mean is taken as a plugin value.

    Parameters
    ----------
    x_train : ``torch.Tensor``
        (size n x d) Training inputs.
    y_train : ``torch.Tensor``
        (size n) Training outputs.
    bounds : ``torch.Tensor``
        (size 2 x d) Optimisation bounds of input space.
    batch_size: ``int``, optional
        Number of new candidate points to return, default is 1.
    acquisition: ``str``, optional
        Acquisition function to use. Must be "EI" or "UCB", default is "EI".
    beta: ``int``, optional
        Trade-off parameter for UCB acquisition function, default is 4.0. No
        impact on when EI is specified.
    constraints : ``dict`` or ``Tuple`` of ``dict``, optional
        Optimisation constraints on inputs, default is no constraints.
    discrete : ``dict``, optional
        Possible values for all discrete inputs in the shape {dim1: [values1],
        dim2: [values2], etc.}, e.g. {0: [1., 2., 3.], 3: [-0.1, -0.2, 100.]}.
    noisy : ``bool``, optional
        Specifies if observations are noisy.
    x_pending : ``torch.Tensor``, optional
        (size n x d) Training inputs of currently pending points, default is no
        pending points.
    mc_samples : ``int``, optional
        Number of Monte Carlo samples to approximate the acquisition function,
        default is 128. Has no effect on analytical acquisition functions.
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
        (size batch_size x d) New candidate inputs.
    """

    opt_bounds = deepcopy(bounds)
 
    # normalise inputs
    if normalise_x:
        x_train = normalise(x_train, bounds)
        dims = bounds.size((1))
        opt_bounds = torch.tensor([[0,]*dims, [1,]*dims])

    # standardise outputs
    if standardise_y:
        y_train = standardise(y_train)

    # GAUSSIAN PROCESS
    # specify Gaussian process
    likelihood = GaussianLikelihood()
    gp = GaussianProcess(x_train, y_train, likelihood=likelihood)

    # fit Gaussian process
    fit_gp(x_train, y_train,
           gp=gp,
           likelihood=likelihood,
           lr=0.1,
           steps=200)
    
    # OPTIMISE ACQUISITION FUNCTION
    if batch_size == 1 and x_pending == None:

        # specify analytical acquisition function
        if acquisition == "EI":

            if noisy:
                target = _find_max_pred(gp, bounds, constraints, discrete)
            else:
                target = torch.max(y_train)

            acq = ExpectedImprovement(gp=gp, y_best=target)
        elif acquisition == "UCB":
            acq = UpperConfidenceBound(gp=gp, beta=beta)
        else:
            raise NotImplementedError("Argument acquisition must be EI or UCB.")

        if constraints == None:

            # CASE 1: Single-point

            # optimise acquisition function
            x_new, _ = single(func=acq,
                              method="L-BFGS-B",
                              bounds=opt_bounds,
                              discrete=discrete,
                              num_starts=num_starts,
                              num_samples=num_samples)
            
        elif isinstance(constraints, (dict, list)):
        
            # CASE 2: Single-point, constrained

            # optimise acquisition function
            x_new, _ = single(func=acq,
                              method="SLSQP",
                              bounds=opt_bounds,
                              constraints=constraints,
                              discrete=discrete,
                              num_starts=num_starts,
                              num_samples=num_samples)
        
        else:
            raise TypeError("""Argument constraints must be None, dict or a list
                            of dicts.""")

    elif batch_size > 1 or isinstance(x_pending, torch.Tensor):
        
        if constraints == None:

            # CASE 3+4: Asynchronous + multi-point

            # specify Monte Carlo acquisition function
            if acquisition == "EI":

                if noisy:
                    target = _find_max_pred(gp, bounds, constraints, discrete)
                else:
                    target = torch.max(y_train)

                acq = MCExpectedImprovement(gp=gp,
                                            y_best=target,
                                            x_pending=x_pending)
            elif acquisition == "UCB":
                acq = MCUpperConfidenceBound(gp=gp,
                                             beta=beta,
                                             x_pending=x_pending)
            else:
                raise NotImplementedError("""Argument acquisition must be EI or
                                          UCB.""")

            # optimise acquisition function
            x_new, _ = multi_sequential(func=acq,
                                        method="Adam",
                                        batch_size=batch_size,
                                        bounds=opt_bounds,
                                        discrete=discrete,
                                        num_starts=num_starts,
                                        num_samples=num_samples)
            
        elif isinstance(constraints, (dict, list, Tuple)):
        
            # CASE 5+6+7+8: Asynchronous + multi-point, constrained

            # specify Monte Carlo acquisition function
            if acquisition == "EI":
                if noisy:
                    target = _find_max_pred(gp, bounds, constraints, discrete)
                else:
                    target = torch.max(y_train)

                acq = MCExpectedImprovement(gp=gp,
                                            y_best=target,
                                            x_pending=x_pending,
                                            samples=mc_samples,
                                            fix_base_samples=True)
            elif acquisition == "UCB":
                acq = MCUpperConfidenceBound(gp=gp,
                                             beta=beta,
                                             x_pending=x_pending,
                                             samples=mc_samples,
                                             fix_base_samples=True)
            else:
                raise NotImplementedError("""Argument acquisition must be EI or
                                          UCB.""")

            # optimise acquisition function
            x_new, _ = multi_sequential(func=acq,
                                        method="SLSQP",
                                        batch_size=batch_size,
                                        bounds=opt_bounds,
                                        constraints=constraints,
                                        discrete=discrete,
                                        num_starts=num_starts,
                                        num_samples=num_samples)

    else:
        raise TypeError("""Argument batch_size must be positive int and
                        x_pending None or torch.Tensor.""")


    # unnormalise new point
    if normalise_x:
        x_new = unnormalise(x_new, bounds)

    return x_new


def _find_max_pred(gp, bounds, constraints, discrete):
    """
    Find maximum of the Gaussian process posterior mean to use as a target for
    expected improvement with noisy observations.
    """
    
    gp.eval()

    def gp_mean(x):
        x = x.reshape((1, -1))
        numpy = False

        if isinstance(x, np.ndarray):
            numpy = True
            x = torch.from_numpy(x)

        pred = gp(x)
        mean = pred.mean.detach()

        if numpy:
            mean = mean.numpy()

        return -mean
    
    # maximise posterior mean
    if len(constraints)==0:
        max_x, max_y = single(gp_mean,
                              method="L-BFGS-B",
                              bounds=bounds,
                              constraints=constraints,
                              discrete=discrete)
        
    elif len(constraints)>0:
        max_x, max_y = single(gp_mean,
                              method="SLSQP",
                              bounds=bounds,
                              constraints=constraints,
                              discrete=discrete)
    
    return -max_y
