# NUBO
Bayesian Optimisation framework for the optimisation of expensive-to-evaluate black box functions developed by the experimental fluid dynamics research group at Newcastle University.

***

## Contents
- **Surrogate models**: Uses Gaussian Processes implemented in `GPyTorch`.  
- **Acquisition function**: Sequential single-point optimisation via analytical Expected Improvement (Jones et al., 1998) and Upper Confidence Bound (Srinivas et al., 2010). Parallel and asynchronous multi-point optimisation via Monte Carlo Sampling (Wilson et al., 2018).  
- **Optimisers**: Bounded deterministic optimisation via multi-start L-BFGS-B. Constraint deterministic optimisation via multi-start SLSQP. Stochastic optimisation via Adam.  
- **Design of experiments**: Generation of initial data points via Maximin Latin Hypercube Sampling.  
- **Synthetic test functions**: Test Bayesian Optimisation algorithms on one of ten synthetic test functions before deploying it for expensive experiments.

***

## Choice of acquisition function
| Property | Analytical | Monte Carlo |
| -------- | ---------- | ----------- |
| Single-point | Yes | Yes |
| Multi-point | - | joint strategy for small batches,sequential strategy for large batches |
| Bounds | L-BFGS-B optimiser | Adam optimiser or L-BFGS-B when base samples fixed |
| Constraints| SLSQP optimiser | SLSQP optimiser when base samples fixed |

***

## References
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. Journal of Global optimization, 13(4), 455.
- Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian process optimization in the bandit setting: No regret and experimental design. Proceedings of the 27th International Conference on Machine Learning, 1015-1022.
- Wilson, J., Hutter, F., & Deisenroth, M. (2018). Maximizing acquisition functions for Bayesian optimization. Advances in neural information processing systems, 31.
