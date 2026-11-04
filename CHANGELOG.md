# Changelog

<!--toc:start-->

- [Changelog](#changelog)
  - [Version 0.3.0](#version-030)
  - [Version 0.2.1](#version-021)

<!--toc:end-->

## Version 0.3.0

- Refactor of GPR parsing to directly use ASTs instead of parsing the strings
  (should improve performance and robustness)
- The gene/reaction weights will now be used to weight the objective function.
  So if a reaction has a weight of -2, it will be weighted as a low-expression
  reaction (nothing will change in terms of the constraints added for it), but
  will have a coefficient of 2 in the objective (that is, if the reaction has a
  flux below 'threshold', it will contribute 2 to the objective). Similarly for
  positive reaction weights, a weight of 2 will contribute that amount to the
  objective if the reaction has a flux above 'epsilon'.
- Allow option for which reactions are constrained in the 'milp', and 'fva'
  model construction methods through the `constrain_all_reactions` parameter. If
  this parameter is `False` (the default), only reactions with nonzero weights
  will have their bounds changed. If `constraint_all_reactions` is `True`, all
  reactions will be evaluated and potentially have their bounds altered (though
  only if the FVA or MILP problem indicates that they should be constrained).
- Update python version constraints to allow for more recent python versions
  (updating version constraint from 3.9–3.12 to 3.9–3.14, tests are run on
  CPython 3.9–3.14 during CI).
- Several other internal refactors (mostly just code reorganization, and
  updating the build and CI system, as well as the developement environment).

## Version 0.2.1

- Fix conflict when symengine is installed
