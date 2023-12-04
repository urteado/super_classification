# super_classification

Implementation of key asymptotic performance metrics presented in the paper [Classification of Heavy-tailed Features in High Dimensions: a Superstatistical Approach](https://arxiv.org/abs/2304.02912).
The current version does is for a setting that demonstrates the key phenomenology and is numerically lightest: binary classification, square loss and l2 regularisation, with variance of datapoints distributed as an inverse-gamma random variable with shape parameter $a$ and scale parameter $c$.

Contains: 
- `how_to.ipynb` python notebook made to set parameters, run theory and simulations, and plot both
- `functions/experiment`: contains function files for running empirical risk minimisation (ERM) to be called in the notebook
- `functions/saddle_point_evolution`: contains function files for the main result: the state-evolution equations that converge to order parameters in terms of which performance metrics are computed, to be called in the notebook
- settings.py global nice plot settings (borrowed from [this project](https://github.com/Shmoo137/Hessian-and-Decision-Boundary))


Requirements:
besides the usual `numpy`, `scipy`, `pyplot`, `pandas`, install `numba`


