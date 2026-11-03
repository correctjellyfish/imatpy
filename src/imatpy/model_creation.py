"""
Submodule with functions for creating a context specific model using iMAT
"""

from __future__ import annotations

# Standard Library Imports
import re
import warnings

# External Imports
import cobra
import numpy as np
import pandas as pd

# Local Imports
from imatpy.imat import (
    ALMOST_ZERO,
    add_imat_constraints,
    add_imat_objective_,
    imat,
)
from imatpy.imatpy_defaults import IMAT_DEFAULTS


# region: Main Model Creation Function
def generate_model(
    model: cobra.Model,
    rxn_weights: pd.Series | dict,
    imat_solution: cobra.Solution | None = None,
    method: str = "imat_restrictions",
    epsilon: float = IMAT_DEFAULTS.epsilon,
    threshold: float = IMAT_DEFAULTS.threshold,
    constrain_all_reactions: bool = False,
    objective_tolerance: float = IMAT_DEFAULTS.objective_tolerance,
    **kwargs,
):
    """Generate a context specific model using iMAT.

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    imat_solution : cobra.Solution
        A pre-existing IMAT solution passed to the model creation
        methods, will only impact simple and subset as FVA and MILP
        solve altered versions of the IMAT problem.
    method : str
        The method to use for generating the context specific model.
        Valid methods are: 'imat_restrictions', 'simple_bounds',
        'subset', 'fva', 'milp'.
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to  be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-1). Represents
        the maximum flux for a reaction to be considered off.
    constrain_all_reactions : bool,default=False
        Whether to (potentially) constrain all the reactions in the model,
        or only those with a non-zero weight. Passed to the `fva_model` or
        `milp_model` functions if they are the selected method.
    objective_tolerance : float
        The tolerance for the objective value (used for
        imat_restrictions and fva methods). The objective will be
        restricted to be within objective_tolerance*objective_value of
        the optimal objective value. (default: 5e-2)
    kwargs
        Keyword arugments are passed to the underlying model generation
        functions (check `See Also`).

    Returns
    -------
    cobra.Model
        A context specific cobra.Model.

    See Also
    --------
    imat_constraint_model : imat_restrictions method
    simple_bounds_model : simple_bounds method
    subset_model : subset method.
    fva_model : fva based method
    milp_model : milp method.
    """
    method = _parse_method(method)
    if method == "imat_constraint":
        return imat_constraint_model(
            model=model,
            rxn_weights=rxn_weights,
            epsilon=epsilon,
            threshold=threshold,
            objective_tolerance=objective_tolerance,
            **kwargs,
        )
    elif method == "simple_bounds":
        return simple_bounds_model(
            model=model,
            rxn_weights=rxn_weights,
            epsilon=epsilon,
            threshold=threshold,
            imat_solution=imat_solution,
            **kwargs,
        )
    elif method == "subset":
        return subset_model(
            model=model,
            rxn_weights=rxn_weights,
            epsilon=epsilon,
            threshold=threshold,
            imat_solution=imat_solution,
            **kwargs,
        )
    elif method == "fva":
        return fva_model(
            model=model,
            rxn_weights=rxn_weights,
            epsilon=epsilon,
            threshold=threshold,
            objective_tolerance=objective_tolerance,
            constrain_all_reactions=constrain_all_reactions,
            **kwargs,
        )
    elif method == "milp":
        return milp_model(
            model=model,
            rxn_weights=rxn_weights,
            epsilon=epsilon,
            threshold=threshold,
            constrain_all_reactions=constrain_all_reactions,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Invalid method: {method}. Valid methods are: 'simple_bounds', \
            'imat_restrictions', "
            f"'subset', 'fva', 'milp'."
        )


# endregion: Main Model Creation Function


# region: Model Creation methods
def imat_constraint_model(
    model, rxn_weights, epsilon, threshold, objective_tolerance, **kwargs
):
    """Generate a context specific model by adding iMAT constraints, and
    ensuring iMAT objective value is near optimal.

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-1). Represents
        the maximum flux for a reaction to be considered off.
    objective_tolerance : float
        The tolerance for the objective value. The objective will be
        restricted to be within objective_tolerance*objective_value of
        the optimal objective value. (default: 5e-2)

    Returns
    -------
    cobra.Model
        A context specific cobra.Model.

    Notes
    -----
    This function first solves the iMAT problem, then adds a constraint
    to ensure that the iMAT objective value is within
    objective_tolerance*objective_value of the optimal objective value.
    This model will include integer constraints, and so won't work with
    COBRApy sampling methods, but can be used with
    `metworkpy.sampling.corner_sampling`. If you want to use the model
    with the COBRApy sampling methods, use any of the other methods.
    """
    original_objective = model.objective
    imat_model = add_imat_constraints(model, rxn_weights, epsilon, threshold)
    add_imat_objective_(imat_model, rxn_weights)
    cobra.util.fix_objective_as_constraint(
        imat_model,
        fraction=(1 - objective_tolerance),
        name="imat_objective_constraint",
    )
    imat_model.objective = original_objective
    return imat_model


def simple_bounds_model(
    model,
    rxn_weights: dict | pd.Series,
    epsilon: float,
    threshold: float,
    imat_solution: cobra.Solution | None = None,
    **kwargs,
):
    """Generate a context specific model by setting bounds on reactions based on
    iMAT solution.

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-1). Represents
        the maximum flux for a reaction to be considered off.
    imat_solution : cobra.Solution
        A preexisting IMAT solution, if not provided will be computed
        when function is called

    Returns
    -------
    cobra.Model
        A context specific cobra.Model.

    Notes
    -----
    This method first solves the iMAT solution, then for reactions found
    to be lowly expressed (weight<0), and inactive (flux<threshold),
    the reaction bounds are set to (-threshold, threshold),
    (0, threshold), or (-threshold, 0) depending on reversibility.
    For reactions found to be highly expressed and active in the
    forward direction (weight>0, flux>epsilon), the reaction bounds are
    set to (epsilon, ub), or (lb, ub) if lb>epsilon. For reactions found
    to be highly expressed and active in the reverse direction
    (weight>0, flux<-epsilon), the reaction bounds are set to
    (lb, -epsilon), or (lb, ub) if ub<-epsilon.  This model will not
    include integer constraints, and so can be used for sampling.
    """
    updated_model = model.copy()
    if not imat_solution:
        imat_solution = imat(model, rxn_weights, epsilon, threshold)
    if imat_solution.status != "optimal":
        raise ValueError("No optimal solution found for IMAT problem")
    fluxes = imat_solution.fluxes
    rl = rxn_weights[rxn_weights < 0].index.tolist()  # type: ignore
    rh = rxn_weights[rxn_weights > 0].index.tolist()  # type: ignore
    inactive_reactions = fluxes[
        (fluxes.abs() <= threshold) & (fluxes.index.isin(rl))
    ]
    forward_active_reactions = fluxes[
        (fluxes >= epsilon) & (fluxes.index.isin(rh))
    ]
    reverse_active_reactions = fluxes[
        (fluxes <= -epsilon) & (fluxes.index.isin(rh))
    ]
    for rxn in inactive_reactions.index:
        reaction = updated_model.reactions.get_by_id(rxn)
        reaction.bounds = _inactive_bounds(
            reaction.lower_bound, reaction.upper_bound, threshold
        )
    for rxn in forward_active_reactions.index:
        reaction = updated_model.reactions.get_by_id(rxn)
        reaction.bounds = _active_bounds(
            reaction.lower_bound, reaction.upper_bound, epsilon, forward=True
        )
    for rxn in reverse_active_reactions.index:
        reaction = updated_model.reactions.get_by_id(rxn)
        reaction.bounds = _active_bounds(
            reaction.lower_bound, reaction.upper_bound, epsilon, forward=False
        )
    return updated_model


def subset_model(
    model: cobra.Model,
    rxn_weights: dict | pd.Series,
    epsilon: float,
    threshold: float,
    imat_solution: cobra.Solution | None = None,
    **kwargs,
):
    """Generate a context specific model by knocking out reactions found to
    be inactive by iMAT.

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-1). Represents
        the maximum flux for a reaction to be considered off.
    imat_solution : cobra.Solution
        A preexisting IMAT solution, if not provided will be computed
        when function is called

    Returns
    -------
    cobra.Model
        A context specific cobra.Model.

    Notes
    -----
    This method first solves the iMAT solution, then for reactions found
    to be lowly expressed (weight<0), and  inactive (flux<threshold), the
    reaction bounds are set to (-threshold, threshold). This model will
    not include integer constraints, and so can be used for sampling.
    """
    updated_model = model.copy()
    if not imat_solution:
        imat_solution = imat(model, rxn_weights, epsilon, threshold)
    if imat_solution.status != "optimal":
        raise ValueError("No optimal solution found for IMAT problem")
    if not isinstance(rxn_weights, pd.Series):
        rxn_weights = pd.Series(rxn_weights)
    fluxes = imat_solution.fluxes
    rl = rxn_weights[rxn_weights < 0.0].index.tolist()  # type: ignore
    inactive_reactions = fluxes[
        (fluxes.abs() <= threshold) & (fluxes.index.isin(rl))
    ]
    for rxn in inactive_reactions.index:
        reaction = updated_model.reactions.get_by_id(rxn)
        assert isinstance(reaction, cobra.Reaction)
        # Force reaction to be below threshold
        reaction.bounds = _inactive_bounds(
            *reaction.bounds,
            threshold=threshold,
        )
    return updated_model


def fva_model(
    model,
    rxn_weights,
    epsilon,
    threshold,
    objective_tolerance,
    constrain_all_reactions: bool = False,
    warn_tolerance: bool = True,
    **kwargs,
):
    """Generate a context specific model by setting bounds on reactions based on
    FVA for an iMAT model.

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-1). Represents
        the maximum flux for a reaction to be considered off.
    constrain_all_reactions : bool,default=False
        Whether to (potentially) constrain all the reactions in the model,
        or only those with a non-zero weight. If True, then every reaction
        will be evaluated and constrained using flux variability analysis.
        If False, then only the reactions with a non-zero weight will be
        evaluated.
    objective_tolerance : float
        The tolerance for the objective value. The objective will be
        restricted to be within objective_tolerance*objective_value of
        the optimal objective value. (default: 5e-2)
    warn_tolerance
        Issue a warning if there is a problem with finding the bounds of
        a particular reaction due to the lower bound being greater than
        the upper bound. This is normally caused by the solver
        tolerance, and the values will be swapped to get valid bounds
        for the reaction.
    kwargs:
        Keyword arguments passed to `cobra.flux_analysis.flux_variability_analysis`

    Returns
    -------
    cobra.Model
        A context specific cobra.Model.

    Notes
    -----
    This method first creates a model with the iMAT constraints, and
    objective and then performs FVA to find the minimum and maximum
    flux for each reaction which allow for the objective to be within
    tolerance of optimal. These values are then set as the reaction
    bounds. This model is not guaranteed to have fluxes consistent
    with the optimal iMAT objective. This model will not include integer
    constraints, and so can be used for sampling.
    """
    updated_model = model.copy()
    imat_model = add_imat_constraints(model, rxn_weights, epsilon, threshold)
    add_imat_objective_(imat_model, rxn_weights)
    if not constrain_all_reactions:
        reactions = rxn_weights[rxn_weights.abs() > ALMOST_ZERO].index.tolist()
    else:
        reactions = model.reactions.list_attr("id")
    fva_res = cobra.flux_analysis.flux_variability_analysis(
        imat_model,
        fraction_of_optimum=(1 - objective_tolerance),
        reaction_list=reactions,
        **kwargs,
    ).dropna()
    for rxn in reactions:
        reaction = updated_model.reactions.get_by_id(rxn)
        new_lb = fva_res.loc[rxn, "minimum"]
        new_ub = fva_res.loc[rxn, "maximum"]
        if new_lb > new_ub:
            if warn_tolerance:
                warnings.warn(
                    f"Problem with bounds computed for {rxn}, calculated "
                    f"lower bound as {new_lb} and upper bound as {new_ub}."
                    f"This should only occur due to the the tolerance of "
                    f"the solver, and so the bounds should differ by at most "
                    f"that tolerance. Swapping the computed bounds to "
                    f"correct the issue."
                )
            # This should force valid bounds
            # And should only happen due to solver tolerance
            # So the new_lb should be at most solver tolerance
            # less than the previous, (similarly with the upperbound)
            new_lb, new_ub = new_ub, new_lb
        reaction.bounds = (
            new_lb,
            new_ub,
        )
    return updated_model


def milp_model(
    model,
    rxn_weights,
    epsilon,
    threshold,
    constrain_all_reactions: bool = False,
    **kwargs,
):
    """Generate a context specific model by setting bounds on reactions based on
    a set of mixed integer linear programming problems.

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-1). Represents
        the maximum flux for a reaction to be considered off.
    constrain_all_reactions : bool,default=False
        Whether to (potentially) constrain all the reactions in the model,
        or only those with a non-zero weight. If True, then every reaction
        will be evaluated for if it should be active/inactive. If False,
        then only the reactions with a non-zero weight will be evaluated.

    Returns
    -------
    cobra.Model
        A context specific cobra.Model.

    Notes
    -----
    This method first creates a model with the iMAT constraints, and
    objective and then solves a set of mixed integer linear programming
    problems, where each reaction is set to be inactive, active in the
    forward direction, and active in the reverse direction. The
    reaction bounds are then set based on the results of the MILP
    problems. This model is not guaranteed to have fluxes consistent
    with the optimal iMAT objective. This model will not include integer
    constraints, and so can be used for sampling.
    """
    updated_model = model.copy()
    imat_model = add_imat_constraints(model, rxn_weights, epsilon, threshold)
    add_imat_objective_(imat_model, rxn_weights)
    if not constrain_all_reactions:
        reactions = rxn_weights[rxn_weights.abs() > ALMOST_ZERO].index.tolist()
    else:
        reactions = model.reactions.list_attr("id")
    milp_results = pd.DataFrame(
        np.nan,
        columns=["inactive", "forward", "reverse"],
        index=reactions,
        dtype=float,
    )
    for rxn in reactions:
        with imat_model as ko_model:  # Knock out the reaction
            reaction = ko_model.reactions.get_by_id(rxn)
            assert isinstance(reaction, cobra.Reaction)
            if (
                reaction.lower_bound > threshold
                or reaction.upper_bound < -threshold
            ):
                # Reaction can't be inactive, so skip
                ko_solution = -1
            else:
                reaction.bounds = _inactive_bounds(
                    reaction.lower_bound, reaction.upper_bound, threshold
                )
                ko_solution = ko_model.slim_optimize(error_value=-1.0)
        with imat_model as forward_model:
            reaction = forward_model.reactions.get_by_id(rxn)
            assert isinstance(reaction, cobra.Reaction)
            if reaction.upper_bound < epsilon:
                # Reaction can't be forced forward, so skip
                forward_solution = -1
            else:
                reaction.bounds = _active_bounds(
                    reaction.lower_bound,
                    reaction.upper_bound,
                    epsilon,
                    forward=True,
                )
                forward_solution = forward_model.slim_optimize(
                    error_value=-1.0
                )
        with imat_model as reverse_model:
            reaction = reverse_model.reactions.get_by_id(rxn)
            assert isinstance(reaction, cobra.Reaction)
            if reaction.lower_bound > -epsilon:
                # Reaction can't be forced reverse, so skip
                reverse_solution = -1
            else:
                reaction.bounds = _active_bounds(
                    reaction.lower_bound,
                    reaction.upper_bound,
                    epsilon,
                    forward=False,
                )
                reverse_solution = reverse_model.slim_optimize(
                    error_value=-1.0
                )
        milp_results.loc[rxn, :] = [
            ko_solution,
            forward_solution,
            reverse_solution,
        ]
    final_results = milp_results.apply(_milp_eval, axis=1)
    # Now 0 is inactive, 1 is forward, -1 is reverse, nan is under determined
    for rxn in reactions:
        if pd.isna(final_results[rxn]):  # skip under-determined reactions
            continue
        reaction = updated_model.reactions.get_by_id(rxn)
        if final_results[rxn] == 0:  # inactive
            reaction.bounds = _inactive_bounds(
                reaction.lower_bound, reaction.upper_bound, threshold
            )
        elif final_results[rxn] == 1:  # forward
            reaction.bounds = _active_bounds(
                reaction.lower_bound,
                reaction.upper_bound,
                epsilon,
                forward=True,
            )
        elif final_results[rxn] == -1:  # reverse
            reaction.bounds = _active_bounds(
                reaction.lower_bound,
                reaction.upper_bound,
                epsilon,
                forward=False,
            )
    return updated_model


# endregion Model Creation methods


# region Helper Functions
# noinspection PyProtectedMember
def _parse_method(method: str) -> str:
    """Parse the method string to a valid method name.

    Parameters
    ----------
    method : str
        The method to parse.

    Returns
    -------
    str
        The parsed method name.
    """
    try:
        return _parse_str_args_dict(
            method,
            {
                "simple_bounds": [
                    "simple bounds",
                    "simple-bounds",
                    "simple_bounds",
                ],
                "imat_constraint": [
                    "imat",
                    "imat_restrictions",
                    "imat-restrictions",
                    "imat restrictions",
                    "ir",
                    "imat constraints",
                    "imat-constraints",
                    "imat_constraints",
                    "ic",
                ],
                "subset": [
                    "subset-ko",
                    "subset_ko",
                    "eliminate-below-threshold",
                    "eliminate_below_threshold",
                    "subset-model",
                    "subset_model",
                    "subset model",
                ],
                "fva": [
                    "fva",
                    "flux_variability_analysis",
                    "flux-variability-analysis",
                    "flux variability analysis",
                    "fva_model",
                    "fva-model",
                    "fva model",
                ],
                "milp": [
                    "milp",
                    "milp-model",
                    "milp_model",
                    "milp model",
                    "mixed_integer_linear_programming",
                    "mixed integer linear programming",
                    "mixed-integer-linear-programming",
                ],
            },
        )
    except ValueError as err:
        raise ValueError(
            f"Invalid method: {method}. Valid methods are: 'simple_bounds', "
            f"'imat_restrictions', 'eliminate_below_threshold', 'fva', 'milp'."
        ) from err


def _inactive_bounds(
    lb: float, ub: float, threshold: float
) -> tuple[float, float]:
    """Find the new bounds for the reaction if it is inactive."""
    if lb > threshold:
        raise ValueError(
            "Lower bound is greater than threshold, reaction can not be \
                inactive."
        )
    if ub < -threshold:
        raise ValueError(
            "Upper bound is less than negative threshold, reaction can not be \
                inactive."
        )
    if lb > ub:
        raise ValueError("Lower bound is greater than upperbound")
    new_lb = max(lb, -threshold)
    new_ub = min(ub, threshold)
    return new_lb, new_ub


def _active_bounds(
    lb: float, ub: float, epsilon: float, forward: bool
) -> tuple[float, float]:
    """Find the new bounds for the reaction if it is active."""
    if lb > ub:
        raise ValueError("Lower bound is greater than upperbound")
    if forward:
        if ub < epsilon:
            raise ValueError(
                "Upper bound is less than epsilon, reaction can not be active \
                    in forward direction."
            )
        new_lb = max(lb, epsilon)
        new_ub = ub
    else:
        if lb > -epsilon:
            raise ValueError(
                "Lower bound is greater than negative epsilon, reaction can \
                    not be active in reverse "
                "direction."
            )
        new_lb = lb
        new_ub = min(ub, -epsilon)
    return new_lb, new_ub


def _milp_eval(milp_res: pd.Series) -> float:
    """Function for evaluating the results of the MILP method, to determine if a \
        reaction should be considered active or inactive. Returns 1 if
        forced forward, -1 if forced reverse, 0 if inactive, and NaN if
        under determined.
    """
    if pd.isna(milp_res).any():
        return np.nan
    if (
        len(np.unique(milp_res)) == 3
    ):  # Find which is max, then return based on that
        max_ind = np.argmax(milp_res)
        if max_ind > 1:
            res = -1  # reverse is the max
        else:
            res = max_ind  # Index corresponds to which is max
        assert isinstance(res, (np.number, int, float)), (
            f"Result is {type(res)}, expected a number"
        )
        return res  # type: ignore
    elif (milp_res["forward"] == milp_res["reverse"]) and (
        milp_res["inactive"] > milp_res["forward"]
    ):
        # Forced forward, and reverse are the same, and inactive is
        # greater than both, so inactive
        return 0
    elif (milp_res["inactive"] == milp_res["reverse"]) and (
        milp_res["forward"] > milp_res["inactive"]
    ):
        # Forced reverse, and inactive are the same, and forward is
        # greater than both, so forward
        return 1
    elif (milp_res["inactive"] == milp_res["forward"]) and (
        milp_res["reverse"] > milp_res["inactive"]
    ):
        # Forced inactive, and forward are the same, and reverse is
        # greater than both, so reverse
        return -1
    else:
        # Under-determined case, return nan
        return np.nan


def _parse_str_args_dict(arg: str, arg_dict: dict[str, list]) -> str:
    """Function to decide which group of arguments given as values in a dictionary the
    argument matches, returns the key for that group.

    Parameters
    ----------
    arg : str
        Argument to match
    arg_dict : Dict[str, List]
        Dictionary defining argument groups

    Returns
    -------
    str
        Key from the dictionary representing the group

    Raises
    ------
    ValueError
        If more than one argument group could be matched
    """
    arg_regex = re.compile(r"^" + arg, re.IGNORECASE)
    filtered_groups = [
        key
        for key, arg_list in arg_dict.items()
        if len([arg for arg in arg_list if arg_regex.match(arg)]) >= 1
    ]
    if len(filtered_groups) == 1:
        return filtered_groups[0]
    raise ValueError(
        f"Argument could match more than one argument group in arg_dict: "
        f"{filtered_groups}"
    )


# endregion Helper Functions
