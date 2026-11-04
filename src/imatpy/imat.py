"""
Submodule with functions for adding iMAT constraints and objectives to a cobra
model, and running iMAT
"""

# Standard Library Imports
from __future__ import annotations

import hashlib
from typing import Literal

# External Imports
import cobra
import pandas as pd

# NOTE: Need to import from symengine directly if it's installed
# otherwise objective construction will fail
try:
    from symengine import Add  # ty: ignore[import-not-found]
except ModuleNotFoundError:
    from sympy import Add

# Local Imports
from imatpy.imatpy_defaults import IMAT_DEFAULTS

# Define global constaints
ALMOST_ZERO = 5e-17


# region: Main iMat Function
def imat(
    model: cobra.Model,
    rxn_weights: pd.Series | dict,
    epsilon: float = IMAT_DEFAULTS.epsilon,
    threshold: float = IMAT_DEFAULTS.threshold,
) -> cobra.Solution:
    """Function for performing iMAT analysis. Returns a cobra Solution object,
    with objective value and fluxes.

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    epsilon : float
        The epsilon value to use for iMAT (default: 1e-3).
        Represents the minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-4).
        Represents the maximum flux for a reaction to be considered off.

    Returns
    -------
    cobra.Solution
        A cobra Solution object with the objective value and fluxes.
    """
    assert epsilon > 0.0, f"Epsilon must be positive, but was {epsilon}"
    assert threshold >= 0.0, f"Threshold must be positive, but was {threshold}"
    assert epsilon > threshold, (
        f"Epsilon must be greater than threshold, but epsilon: {epsilon} < threshold: {threshold}"
    )
    imat_model = add_imat_constraints(model, rxn_weights, epsilon, threshold)
    add_imat_objective_(imat_model, rxn_weights)
    return imat_model.optimize()


# endregion: Main iMat Function


# region: iMAT extension functions
def flux_to_binary(
    fluxes: pd.Series,
    which_reactions: str = "active",
    epsilon: float = IMAT_DEFAULTS.epsilon,
    threshold: float = IMAT_DEFAULTS.threshold,
    tolerance=IMAT_DEFAULTS.solver_tolerance,
) -> pd.Series:
    """Convert a pandas series of fluxes to a pandas series of binary values.

    Parameters
    ----------
    fluxes : pandas.Series
        A pandas series of fluxes.
    which_reactions : str
        Which reactions should be the binary values?
        Options are "active", "inactive", "forward", "reverse", or their first
        letters. Default is "active". "active" will return 1 for reactions
        with absolute value of flux greater than epsilon, and 0 for reactions
        with flux less than epsilon. "inactive" will return 1 for reactions
        with absolute value of flux less than threshold, and 0 for reactions
        with flux greater than threshold. "forward" will return 1 for reactions
        with flux greater than epsilon, and 0 for reactions with flux less than
        epsilon. "reverse" will return 1 for reactions with flux less than
        -epsilon, and 0 for reactions with flux greater than -epsilon.
    epsilon : float
        The epsilon value to use for iMAT (default: 1e-3).
        Represents the minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-4).
        Represents the maximum flux for a reaction to be considered off.
    tolerance : float
        The tolerance of the solver. Default from cobra is 1e-7.
    Returns
    -------
    pandas.Series
        A pandas series of binary values.

    Notes
    -----
    This doesn't account for gene expression level, to determine highly expressed reactions which are
    considered on/off, the output of this function will need to be compared with reaction weights.
    """
    assert epsilon > 0.0, f"Epsilon must be positive, but was {epsilon}"
    assert threshold >= 0.0, f"Threshold must be positive, but was {threshold}"
    assert epsilon > threshold, (
        f"Epsilon must be greater than threshold, but epsilon: {epsilon} < threshold: {threshold}"
    )
    which_reactions = _parse_which_reactions(which_reactions)
    if which_reactions == "forward":
        return (fluxes >= (epsilon - tolerance)).astype(int)
    elif which_reactions == "reverse":
        return (fluxes <= (-epsilon + tolerance)).astype(int)
    elif which_reactions == "active":
        return (
            (fluxes >= epsilon - tolerance) | (fluxes <= -epsilon + tolerance)
        ).astype(int)
    elif which_reactions == "inactive":
        return (
            (fluxes <= threshold + tolerance)
            & (fluxes >= -threshold - tolerance)
        ).astype(int)
    else:
        raise ValueError(
            "Couldn't Parse which_reactions, should be one of: \
                         active, inactive, forward, reverse"
        )


def compute_imat_objective(
    fluxes: pd.Series,
    rxn_weights: dict[str, float] | pd.Series[float],
    epsilon: float = IMAT_DEFAULTS.epsilon,
    threshold: float = IMAT_DEFAULTS.threshold,
) -> float:
    """
    Compute the iMAT objective value for a given set of fluxes.

    Parameters
    ----------
    fluxes : pandas.Series
        A pandas series of fluxes.
    rxn_weights : dict of str to float or pandas.Series of floats
        A dictionary or pandas series of reaction weights.
    epsilon : float
        The epsilon value to use for iMAT (default: 1e-3).
        Represents the minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-4).
        Represents the maximum flux for a reaction to be considered off.

    Returns
    -------
    float
        The iMAT objective value.
    """
    if isinstance(rxn_weights, dict):
        rxn_weights = pd.Series(rxn_weights)
    rh = rxn_weights[rxn_weights > ALMOST_ZERO]
    rl = rxn_weights[rxn_weights < -ALMOST_ZERO]
    # Get the fluxes greater than epsilon which are highly expressed
    rh_pos = fluxes[rh.index].ge(epsilon).sum()
    # Get the fluxes less than -epsilon which are highly expressed
    rh_neg = fluxes[rh.index].le(-epsilon).sum()
    # Get the fluxes whose absolute value is less than threshold which are
    # lowly expressed
    rl_pos = fluxes[rl.index].abs().le(threshold).sum()
    return rh_pos + rh_neg + rl_pos


# endregion: iMAT extension functions


# region: iMAT Helper Functions
def add_imat_constraints_(
    model: cobra.Model,
    rxn_weights: pd.Series[float] | dict[str, float],
    epsilon: float = IMAT_DEFAULTS.epsilon,
    threshold: float = IMAT_DEFAULTS.threshold,
) -> cobra.Model:
    """Add the IMAT constraints to the model (updates the model in place).

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to update with iMAT constraints.
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    epsilon : float
        The epsilon value to use for iMAT (default: 1e-3).
        Represents the minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-4).
        Represents the maximum flux for a reaction to be considered off.

    Returns
    -------
    cobra.Model
        The updated model.

    """
    assert epsilon > 0.0, f"Epsilon must be positive, but was {epsilon}"
    assert threshold >= 0.0, f"Threshold must be positive, but was {threshold}"
    assert epsilon > threshold, (
        f"Epsilon must be greater than threshold, but epsilon: {epsilon} < threshold: {threshold}"
    )
    for rxn, weight in rxn_weights.items():
        assert isinstance(rxn, str)
        # Don't add any restrictions for 0 weight reactions
        if -ALMOST_ZERO < weight < ALMOST_ZERO:
            continue
        if weight > ALMOST_ZERO:  # Add highly expressed constraint
            _imat_pos_weight_(model=model, rxn=rxn, epsilon=epsilon)
        elif weight < -ALMOST_ZERO:  # Add lowly expressed constraint
            _imat_neg_weight_(model=model, rxn=rxn, threshold=threshold)
    return model


def add_imat_constraints(
    model, rxn_weights, epsilon: float = 1e-3, threshold: float = 1e-4
) -> cobra.Model:
    """Add the IMAT constraints to the model (returns new model, doesn't
    update model in place).

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to update with iMAT constraints.
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    epsilon : float
        The epsilon value to use for iMAT (default: 1e-3).
        Represents the minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-4).
        Represents the maximum flux for a reaction to be considered off.
    Returns
    -------
    cobra.Model
        The updated model.
    """
    imat_model = model.copy()
    add_imat_constraints_(imat_model, rxn_weights, epsilon, threshold)
    return imat_model


def add_imat_objective_(
    model: cobra.Model, rxn_weights: pd.Series[float] | dict[str, float]
) -> None:
    """
    Add the IMAT objective to the model (updates the model in place).
    Model must already have iMAT constraints added.

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to update with iMAT constraints.
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    """
    if isinstance(rxn_weights, dict):
        rxn_weights = pd.Series(rxn_weights)
    rh = rxn_weights[rxn_weights > ALMOST_ZERO]
    rl = rxn_weights[rxn_weights < -ALMOST_ZERO]
    obj = []
    for rxn, weight in rh.items():  # For each highly expressed reaction
        # Get the forward and reverse variables from the model
        forward_variable = model.solver.variables[
            _get_rxn_imat_binary_variable_name(
                rxn,  # type: ignore
                expression_weight="high",
                which="positive",
            )
        ]
        reverse_variable = model.solver.variables[
            _get_rxn_imat_binary_variable_name(
                rxn,  # type: ignore
                expression_weight="high",
                which="negative",
            )
        ]
        # Adds the two variables to the rh list which will be used for sum
        obj += [forward_variable * weight, reverse_variable * weight]
    for rxn, weight in rl.items():  # For each lowly expressed reaction
        variable = model.solver.variables[
            _get_rxn_imat_binary_variable_name(
                rxn,  # type: ignore
                expression_weight="low",
                which="positive",
            )
        ]
        # Note: Only one variable for lowly expressed reactions
        obj += [variable * abs(weight)]
    imat_obj = model.solver.interface.Objective(Add(*obj), direction="max")
    model.objective = imat_obj


def add_imat_objective(
    model: cobra.Model, rxn_weights: pd.Series[float] | dict[str, float]
) -> cobra.Model:
    """
    Add the IMAT objective to the model (doesn't change passed model).
    Model must already have iMAT constraints added.

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to update with iMAT constraints.
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.

    Returns
    -------
    updated_model : cobra.Model
        New model with the iMAT objective added
    """
    imat_model = model.copy()
    add_imat_objective_(imat_model, rxn_weights)
    return imat_model


# endregion: iMAT Helper Functions


# region: Internal Methods


def _imat_pos_weight_(model: cobra.Model, rxn: str, epsilon: float) -> None:
    """Internal method for adding positive weight constraints to the model.

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to update with iMAT constraints.
    rxn : str
        The reaction ID to add the constraint to.
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.

    Returns
    -------
    unknown
        None
    """
    assert epsilon > 0.0, f"Epsilon must be positive, but was {epsilon}"
    reaction = model.reactions.get_by_id(rxn)
    assert isinstance(reaction, cobra.Reaction)
    lb = reaction.lower_bound
    ub = reaction.upper_bound
    assert reaction.forward_variable is not None
    assert reaction.reverse_variable is not None
    reaction_flux = reaction.forward_variable - reaction.reverse_variable
    y_pos = model.solver.interface.Variable(
        _get_rxn_imat_binary_variable_name(
            reaction.id, expression_weight="high", which="positive"
        ),
        type="binary",
    )
    model.solver.add(y_pos)
    forward_constraint = model.solver.interface.Constraint(
        reaction_flux + (y_pos * (lb - epsilon)),
        lb=lb,
        name=_get_rxn_imat_constraint_name(
            reaction.id, expression_weight="high", which="forward"
        ),
    )
    model.solver.add(forward_constraint)
    y_neg = model.solver.interface.Variable(
        _get_rxn_imat_binary_variable_name(
            reaction.id, expression_weight="high", which="negative"
        ),
        type="binary",
    )
    model.solver.add(y_neg)
    reverse_constraint = model.solver.interface.Constraint(
        reaction_flux + y_neg * (ub + epsilon),
        ub=ub,
        name=_get_rxn_imat_constraint_name(
            reaction.id, expression_weight="high", which="reverse"
        ),
    )
    model.solver.add(reverse_constraint)


def _imat_neg_weight_(model: cobra.Model, rxn: str, threshold: float) -> None:
    """Internal method for adding negative weight constraints to the model.

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to update with iMAT constraints.
    rxn : str
        The reaction ID to add the constraint to.
    threshold : float
        The threshold value to use for iMAT (default: 1e-2). Represents
        the maximum flux for a reaction to be considered off.

    Returns
    -------
    unknown
        None
    """
    reaction = model.reactions.get_by_id(rxn)
    assert isinstance(reaction, cobra.Reaction)
    lb = reaction.lower_bound
    ub = reaction.upper_bound
    assert reaction.forward_variable is not None
    assert reaction.reverse_variable is not None
    reaction_flux = reaction.forward_variable - reaction.reverse_variable
    y_pos = model.solver.interface.Variable(
        _get_rxn_imat_binary_variable_name(
            reaction.id, expression_weight="low", which="positive"
        ),
        type="binary",
    )
    model.solver.add(y_pos)
    forward_constraint = model.solver.interface.Constraint(
        reaction_flux - ub * (1 - y_pos) - threshold * y_pos,
        ub=0,
        name=_get_rxn_imat_constraint_name(
            reaction.id, expression_weight="low", which="forward"
        ),
    )
    model.solver.add(forward_constraint)
    reverse_constraint = model.solver.interface.Constraint(
        reaction_flux - lb * (1 - y_pos) + threshold * y_pos,
        lb=0,
        name=_get_rxn_imat_constraint_name(
            reaction.id, expression_weight="low", which="reverse"
        ),
    )
    model.solver.add(reverse_constraint)


def _parse_which_reactions(which_reactions: str) -> str:
    if which_reactions.lower() in ["active", "on"]:
        return "active"
    elif which_reactions.lower() in ["inactive", "off"]:
        return "inactive"
    elif which_reactions.lower() in ["forward", "f"]:
        return "forward"
    elif which_reactions.lower() in ["reverse", "r"]:
        return "reverse"
    else:
        raise ValueError(
            "Couldn't Parse which_reactions, should be one of: \
                         active, inactive, forward, reverse"
        )


def _get_rxn_imat_binary_variable_name(
    rxn_id: str,
    expression_weight: Literal["high", "low"],
    which: Literal["positive", "negative"],
):
    """
    Get the name for the binary variable associated with the reaction in the imat problem

    Parameters
    ----------
    rxn_id : str
        ID of the reaction to get the binary variable name for
    expression_weight : "high" or "low"
        Whether the variable is for a high expression, or low expression reaction
    which : "positive" or "negative
        Which of the associated variables to get the name for (only used for high expression reactions),
        the positive (associated with the reaction being active in the forward direction) or negative (
        associated with the reaction being active in the reverse direction)

    Raises
    ------
    ValueError
        If expression_weight is not 'high' or 'low', or which is not 'positive' or 'negative
    """
    if expression_weight == "high":
        if (which != "positive") and (which != "negative"):
            raise ValueError(
                f"Version must be either 'positive' or 'negative' but received {which}"
            )
        initial_name = f"y_{expression_weight}_{which[:3]}_{rxn_id}"
    elif expression_weight == "low":
        initial_name = f"y_{expression_weight}_{rxn_id}"
    else:
        raise ValueError(
            f"Expression must be 'high' or 'low' but received {expression_weight}"
        )
    # Get a hash of the initial name to "ensure" no name collisions
    if initial_name is not None:
        name_hash = hashlib.md5(initial_name.encode("utf-8")).hexdigest()[-8:]
        return f"{initial_name}_{name_hash}"
    else:
        raise ValueError(
            f"Expression must be 'high' or 'low' but received {expression_weight}"
        )


def _get_rxn_imat_constraint_name(
    rxn_id: str,
    expression_weight: Literal["high", "low"],
    which: Literal["forward", "reverse"],
):
    """
    Get the name for the constraint associated with the reaction in the imat problem

    Parameters
    ----------
    rxn_id : str
        ID of the reaction to get the constraint name for
    expression_weight : "high" or "low"
        Whether the constraint is for a high expression, or low expression reaction
    which : "forward" or "reverse"
        Which direction of constraint to get the name for

    Raises
    ------
    ValueError
        If expression is not 'high' or 'low', or which is not 'forward', or 'reverse'
    """
    if (expression_weight != "high") and (expression_weight != "low"):
        raise ValueError(
            f"Expression must be either 'high' or 'low', but received {expression_weight}"
        )
    if (which != "forward") and (which != "reverse"):
        raise ValueError(
            f"Which must be either 'forward' or 'reverse', but received {which}"
        )
    initial_name = f"imat_{expression_weight}_{which}_{rxn_id}_constraint"
    name_hash = hashlib.md5(initial_name.encode("utf-8")).hexdigest()[-8:]
    return f"{initial_name}_{name_hash}"


# endregion: Internal Methods
