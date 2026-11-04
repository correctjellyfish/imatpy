# Welcome to iMATpy

![iMATpy Logo](imatpy_logo.png "iMATpy Logo")

[![Documentation Status](https://readthedocs.org/projects/imatpy/badge/?version=latest)](https://imatpy.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/Braden-Griebel/imatpy/actions/workflows/tests.yml/badge.svg)](https://github.com/Braden-Griebel/imatpy/actions/workflows/tests.yml/badge.svg)

## About

iMATpy is a python package implementing the iMAT algorithm for integrating gene
expression information with a genome scale metabolic model, described in
[(Shlomi T, et al. Network-based prediction of human tissue-specific metabolism, Nat. Biotechnol., 2008, vol. 26 (pg. 1003-1010))](https://www.nature.com/articles/nbt.1487).

## Background

The iMAT algorithm takes a genome scale metabolic model (GSMM), and gene
expression weights, then tries to find a feasible flux distribution which
maximizes the sum of reactions with high expression which have a flux above
`epsilon`, and reactions with low expression which have a flux below
`threshold`.

## Installation

This package can be installed via pip

```python
pip install imatpy
```

## Solvers

In order to solve the optimization problems associated with iMAT, a LP/MILP
solver is required. By default this package uses the
[GLPK](https://www.gnu.org/software/glpk/) solver which is installed
automatically by COBRApy. [Gurobi](https://www.gurobi.com/) and
[CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer)
can also be used (for improved performance). Both Gurobi and CPLEX have free
academic licenses available
([Gurobi Academic Licenses](https://www.gurobi.com/academics) and
[CPLEX Academic Initiative](https://www.ibm.com/support/pages/ibm-ilog-optimization-academic-initiative)).

## Workflow

The general workflow for using this package is as follows:

1. Normalize gene expression data for sequencing depth and gene length (TPM or
   RPKM both work well)
1. Quantile normalize the gene expression data (for example with the
   [`qnorm` package](https://pypi.org/project/qnorm/)) (This is optional, but
   can help make the results be less sensitive to parameters)
1. Combine gene expression values across samples, for example by taking the mean
   or median
1. Convert the gene expression data into qualitative weights, of -1, 0, or 1,
   where -1 represents low expression, 1 represents high expression, and 0
   represents all other genes. This can be done by setting the bottom 15% of
   genes to -1, the top 15% of genes to 1, and all other genes to 0. Other
   percentiles can be used, but 15% is a decent starting point.
1. Using the `parse_gpr` sub-module, specifically the `gene_to_rxn_weights`
   function, convert the gene weights into reaction weights using the GPR rules
   in the model.
1. Either generate a flux distribution with the iMAT algorithm, or construct a
   new model with restrictions based on iMAT results.
   - To generate a flux distribution, use the `imatpy.imat` sub-module,
     specifically the `imat` function. This function takes a model, and a series
     of reaction weights, and returns a flux distribution which maximizes the
     sum of reactions with high expression which have a flux above `epsilon`,
     and reactions with low expression which have a flux below `threshold`.
   - To generate a model with restrictions based on the iMAT results, use the
     `model_creation` sub-module. It includes a variety of different methods for
     generating models based on iMAT, with a wrapper function `generate_model`
     for convenience.

## Parse GPR

The `parse_gpr` sub-module has functions for applying the gene-protein-reaction
(GPR) rules in a model to gene weights in order to convert gene weights into
reaction weights which can act as input to the iMAT algorithm.

The input to this method is a pandas Series with the index being the gene
identifiers, and the values being the gene weights. The output is a pandas
Series with the index being the reaction identifiers, and the values being the
reaction weights. This method only handles a single series, so the gene
expression data must be processed into single weights (with -1 representing
lowly expressed genes, 1 representing highly expressed genes, and 0 for all
other genes).

Here is an example of how to use this method:

```python
# External imports
import pandas as pd

# iMATpy imports
from imatpy.model_utils import read_model
from imatpy.parse_gpr import gene_to_rxn_weights

# Read in the model
model = read_model("./tests/data/test_model.xml")

# Create a pandas Series representing gene expression weights
model_weights = pd.Series({
            "g_A_imp": 1,
            "g_B_imp": -1,
            "g_C_imp": -1,
            "g_F_exp": 0,
            "g_G_exp": -1,
            "g_H_exp": 0,
            "g_A_B_D_E": 0,
            "g_C_E_F": -1,
            "g_C_H": 0,
            "g_D_G": 1,
        })

# Convert the gene weights into reaction weights
reaction_weights = gene_to_rxn_weights(model=model, gene_weights=model_weights)

# Print the reaction weights
print(reaction_weights)
```

```
R_A_e_ex     0.0
R_B_e_ex     0.0
R_C_e_ex     0.0
R_F_e_ex     0.0
R_G_e_ex     0.0
R_H_e_ex     0.0
R_A_imp      1.0
R_B_imp     -1.0
R_C_imp     -1.0
R_F_exp      0.0
R_G_exp     -1.0
R_H_exp      0.0
r_A_B_D_E    0.0
r_C_E_F     -1.0
r_C_H        0.0
r_D_G        1.0
dtype: float64
```

## iMAT Methods

The `imat` sub-module contains functions for running the iMAT algorithm. The
main function is `imat`, which takes a model, and a series of reaction weights,
and returns a flux distribution which maximizes the sum of reactions with high
expression which have a flux above `epsilon`, and reactions with low expression
which have a flux below `threshold`.

Here is an example of how to use this method:

```python
# External imports
import pandas as pd

# iMATpy imports
from imatpy.model_utils import read_model
from imatpy.imat import imat

# Read in the model
model = read_model("./tests/data/test_model.xml")

# Read in the reaction weights
rxn_weights = pd.read_csv("./tests/data/test_model_reaction_weights.csv", index_col=0).squeeze()

# Run iMAT
imat_results = imat(model=model, rxn_weights=rxn_weights, epsilon=1, threshold=0.01)

# Print the imat objective
print(f"iMAT Objective: {imat_results.objective_value}")

# Print the imat flux distribution
print(f"iMAT Flux Distribution: \n{imat_results.fluxes}")
```

```
iMAT Objective: 3.0
iMAT Flux Distribution:
R_A_e_ex    -1.0
R_B_e_ex    -1.0
R_C_e_ex    -1.0
R_F_e_ex     1.0
R_G_e_ex     1.0
R_H_e_ex    -0.0
R_A_imp      1.0
R_B_imp      1.0
R_C_imp      1.0
R_F_exp      1.0
R_G_exp      1.0
R_H_exp      0.0
r_A_B_D_E    1.0
r_C_E_F      1.0
r_C_H        0.0
r_D_G        1.0
Name: fluxes, dtype: float64
```

## Model Creation

The `model_creation` sub-module contains functions for creating new models based
on the results of iMAT. The main function is `generate_model`, which takes a
model, and a series of reaction weights, and returns a new model with
restrictions based on the iMAT results.

The available methods for creating a model based on an iMAT flux distribution
is:

- `imat_restrictions` Adds the binary variables and constraints used in the iMAT
  algorithm, as well as an additional constraint ensuring that the flux
  distribution is within tolerance of the optimal iMAT objective value. This
  method stays closest to the iMAT objective, but the included indicator
  (binary) variables mean that is unsuitable for sampling.

- `simple_bounds` Adds bounds on the reactions found to be “on”, and “off” in
  iMAT. For all the highly expressed reactions found to be “on”, the flux is
  constrained to be at least `epsilon`. For all the lowly expressed reactions
  found to be “off”, the flux is constrained to be below `threshold`.

- `subset` Removes reactions from the model which are found to be “off”. For all
  the lowly expressed reactions found to be off, they are constrained to have a
  flux below `threshold`.

- `fva` Finds bounds using an FVA like approach. A temporary model is created in
  a simmilar way to the `imat_restrictions` method above, which includes the
  imat variables, constraints, and which also constrains the flux distribution
  to be near optimal for iMAT. The maximum and minimum fluxes allowed through
  each reaction (while still maintaining the optimal iMAT objective) is found.
  These values are used as the new reaction bounds. It should be noted, that
  although the individual upper and lower bounds for the reaction are achievable
  for each reation while being consistant with the optimal iMAT objective, this
  doesn’t guarantee that the flux distribution overall is consistant with the
  optimal iMAT objective.

- `milp` Uses a set of mixed integer linear programs to find whether a reaction
  should be forced off, forward, or reverse. Each reaction in turn is forced to
  be off, active in the forward direction, and active in the reverse direction,
  and the iMAT objective is maximized. Whether a reaction should be forced off,
  or active in either the forward or reverse direction is then determined by
  which direction maximizes the iMAT objective. Again, it should be noted that
  this doesn’t guarantee that the iMAT objective is overall maximized by
  solutions to this model.

Below is an example of how to generate a model with iMAT restrictions:

```python
# External imports
import pandas as pd

# iMATpy imports
from imatpy.model_utils import read_model
from imatpy.model_creation import generate_model

# Read in the model
model = read_model("./tests/data/test_model.xml")

# Read in the reaction weights
rxn_weights = pd.read_csv("./tests/data/test_model_reaction_weights.csv", index_col=0).squeeze()

# Generate a model with iMAT restrictions
imat_model = generate_model(model=model, rxn_weights=rxn_weights, method="fva", epsilon=1, threshold=0.01)

# This model can be used for sampling, finding optimal biomass generation, finding essential genes, etc.
optimal_biomass = imat_model.slim_optimize()

print(f"Optimal Biomass: {optimal_biomass}")
```

```
Optimal Biomass: 50.0
```

## Model Utils

The `model_utils` sub-module contains several utility functions for working with
COBRApy models. Specifically, it contains functions for: - Reading and writing
models in various formats with a single function, specifically
`read_model`/`write_model`. - Determining if two models are equivalent, using
`model_eq`.

Here is an example of how to use the model IO methods:

```python
# iMATpy imports
from imatpy.model_utils import read_model, write_model, model_eq

# You can read in a model from a file
model = read_model("./tests/data/test_model.xml") # in SBML
model = read_model("./tests/data/test_model.json") # in JSON
model = read_model("./tests/data/test_model.yml") # in YAML
model = read_model("./tests/data/test_model.mat") # in Matlab

# You can also write a model to a file
write_model(model, "./tests/data/test_model.xml") # in SBML
write_model(model, "./tests/data/test_model.json") # in JSON
write_model(model, "./tests/data/test_model.yml") # in YAML
write_model(model, "./tests/data/test_model.mat") # in Matlab
```

Here is an example of using the model comparison method:

```python
# iMATpy imports
from imatpy.model_utils import read_model, model_eq

# Read a model
model = read_model("./tests/data/test_model.xml")

# Create a copy of the model
model_copy = model.copy()

# Check that the models are equivalent
print(f"Models are equivalent: {model_eq(model, model_copy)}")

# Change the copy model
model_copy.reactions.get_by_id("r_A_B_D_E").lower_bound = -314

# Check that the models are no longer equivalent
print(f"Models are equivalent: {model_eq(model, model_copy)}")
```

```
Models are equivalent: True
Models are equivalent: False
```

## License

This package itself is relased under an MIT license. It makes use of several
libraries which are listed below:

- COBRApy: Released under the LGPL-2.0-or-later license see
  [LICENSE](https://github.com/opencobra/cobrapy/blob/devel/LICENSE)
- HiGHSpy: Released under the MIT license see
  [LICENSE](https://github.com/ERGO-Code/HiGHS/blob/master/LICENSE.txt)
- Numpy: Released under the BSD-3-Clause license see
  [LICENSE](https://github.com/numpy/numpy/blob/main/LICENSE.txt)
- Optlang: Released under the Apache-2.0 license see
  [LICENSE](https://github.com/opencobra/optlang/blob/master/LICENSE)
- OSQP: Released under the Apache-2.0 license see
  [LICENSE](https://github.com/osqp/osqp/blob/master/LICENSE)
- Pandas: Released under the BSD-3-Clause license see
  [LICENSE](https://github.com/pandas-dev/pandas/blob/main/LICENSE)
- Scipy: Released under the BSD-3-Clause license see
  [LICENSE](https://github.com/scipy/scipy/blob/main/LICENSE.txt)
- Sympy: Released under the BSD-3-Clause license see
  [LICENSE](https://github.com/sympy/sympy/blob/master/LICENSE)

In addition, these libraries were used during development:

- ruff: Used for autoformatting and linting
- ty: Used for type checking/LSP
- jupyter: Used for rendering quarto documents pytest: Used for running unit
  tests
- Sphinx: Used for generating documentation
- sphinx-rtd-theme: Used for for mating the documentation
- sphinc-exec-code: Used for running code examples in documentation
- numpydoc: Used with sphinx for generating documentation
- napoleon: Used with sphinx for generating documentation
- pre-commit-hooks: Used for running linting/formatting before commits
- tox: Used for running linting and tests across different python versions, and
  during CI
- UV: Project management and build system
