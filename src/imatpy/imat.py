"""
Submodule with functions for adding iMAT constraints and objectives to a cobra
model, and running iMAT
"""

# Standard Library Imports
from __future__ import annotations
import re
from typing import Union

# External Imports
import cobra
from cobra.core.configuration import Configuration
import numpy as np
import pandas as pd
import sympy as sym

# Local Imports

# define defaults for the iMAT functions
DEFAULTS = {
    "epsilon": 1e-2,
    "threshold": 1e-3,
    "tolerance": Configuration().tolerance,
}

BINARY_REGEX = re.compile(r"y_(pos|neg)_(.+)")


# region: Main iMat Function
def imat(
    model: cobra.Model,
    rxn_weights: Union[pd.Series, dict],
    epsilon: float = DEFAULTS["epsilon"],
    threshold: float = DEFAULTS["threshold"],
) -> cobra.Solution:
    """
    Function for performing iMAT analysis. Returns a cobra Solution object,
    with objective value and fluxes.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3).
        Represents the minimum flux for a reaction to be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4).
        Represents the maximum flux for a reaction to be considered off.
    :type threshold: float
    :return: A cobra Solution object with the objective value and fluxes.
    :rtype: cobra.Solution
    """
    imat_model = add_imat_constraints(model, rxn_weights, epsilon, threshold)
    add_imat_objective_(imat_model, rxn_weights)
    return imat_model.optimize()


# endregion: Main iMat Function


# region: iMAT extension functions
def flux_to_binary(
    fluxes: pd.Series,
    which_reactions: str = "active",
    epsilon: float = DEFAULTS["epsilon"],
    threshold: float = DEFAULTS["threshold"],
    tolerance=DEFAULTS["tolerance"],
) -> pd.Series:
    """
    Convert a pandas series of fluxes to a pandas series of binary values.

    :param fluxes: A pandas series of fluxes.
    :type fluxes: pandas.Series
    :param which_reactions: Which reactions should be the binary values?
        Options are "active", "inactive", "forward", "reverse", or their first
        letters. Default is "active". "active" will return 1 for reactions
        with absolute value of flux greater than epsilon, and 0 for reactions
        with flux less than epsilon. "inactive" will return 1 for reactions
        with absolute value of flux less than threshold, and 0 for reactions
        with flux greater than threshold. "forward" will return 1 for reactions
        with flux greater than epsilon, and 0 for reactions with flux less than
        epsilon. "reverse" will return 1 for reactions with flux less than
        -epsilon, and 0 for reactions with flux greater than -epsilon.
    :type which_reactions: str
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3).
        Represents the minimum flux for a reaction to be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4).
        Represents the maximum flux for a reaction to be considered off.
    :type threshold: float
    :param tolerance: The tolerance of the solver. Default from cobra is 1e-7.
    :type tolerance: float
    :return: A pandas series of binary values.
    :rtype: pandas.Series
    """
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
            (fluxes <= threshold + tolerance) & (fluxes >= -threshold - tolerance)
        ).astype(int)
    else:
        raise ValueError(
            "Couldn't Parse which_reactions, should be one of: \
                         active, inactive, forward, reverse"
        )


def compute_imat_objective(
    fluxes: pd.Series,
    rxn_weights,
    epsilon: float = DEFAULTS["epsilon"],
    threshold: float = DEFAULTS["threshold"],
):
    """
    Compute the iMAT objective value for a given set of fluxes.

    :param fluxes: A pandas series of fluxes.
    :type fluxes: pandas.Series
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3).
        Represents the minimum flux for a reaction to be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4).
        Represents the maximum flux for a reaction to be considered off.
    :type threshold: float
    :return: The iMAT objective value.
    """
    if isinstance(rxn_weights, dict):
        rxn_weights = pd.Series(rxn_weights)
    rh = rxn_weights[rxn_weights > 0]
    rl = rxn_weights[rxn_weights < 0]
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
    rxn_weights: Union[pd.Series, dict],
    epsilon: float = DEFAULTS["epsilon"],
    threshold: float = DEFAULTS["threshold"],
) -> cobra.Model:
    """
    Add the IMAT constraints to the model (updates the model in place).

    :param model: A cobra.Model object to update with iMAT constraints.
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3).
        Represents the minimum flux for a reaction to be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4).
        Represents the maximum flux for a reaction to be considered off.
    :type threshold: float
    :return: The updated model.
    :rtype: cobra.Model
    """
    for rxn, weight in rxn_weights.items():
        # Don't add any restrictions for 0 weight reactions
        if np.isclose(weight, 0):
            continue
        if weight > 0:  # Add highly expressed constraint
            _imat_pos_weight_(model=model, rxn=rxn, epsilon=epsilon)
        elif weight < 0:  # Add lowly expressed constraint
            _imat_neg_weight_(model=model, rxn=rxn, threshold=threshold)
    return model


def add_imat_constraints(
    model, rxn_weights, epsilon: float = 1e-3, threshold: float = 1e-4
) -> cobra.Model:
    """
    Add the IMAT constraints to the model (returns new model, doesn't
    update model in place).

    :param model: A cobra.Model object to update with iMAT constraints.
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3).
        Represents the minimum flux for a reaction to be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-4).
        Represents the maximum flux for a reaction to be considered off.
    :type threshold: float
    :return: The updated model.
    :rtype: cobra.Model
    """
    imat_model = model.copy()
    add_imat_constraints_(imat_model, rxn_weights, epsilon, threshold)
    return imat_model


def add_imat_objective_(
    model: cobra.Model, rxn_weights: Union[pd.Series, dict]
) -> None:
    """
    Add the IMAT objective to the model (updates the model in place).
    Model must already have iMAT constraints added.

    :param model: A cobra.Model object to update with iMAT constraints.
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :return: None
    """
    if isinstance(rxn_weights, dict):
        rxn_weights = pd.Series(rxn_weights)
    rh = rxn_weights[rxn_weights > 0].index.tolist()
    rl = rxn_weights[rxn_weights < 0].index.tolist()
    rh_obj = []
    rl_obj = []
    for rxn in rh:  # For each highly expressed reaction
        # Get the forward and reverse variables from the model
        forward_variable = model.solver.variables[f"y_pos_{rxn}"]
        reverse_variable = model.solver.variables[f"y_neg_{rxn}"]
        # Adds the two variables to the rh list which will be used for sum
        rh_obj += [forward_variable, reverse_variable]
    for rxn in rl:  # For each lowly expressed reaction
        variable = model.solver.variables[f"y_pos_{rxn}"]
        # Note: Only one variable for lowly expressed reactions
        rl_obj += [variable]
    imat_obj = model.solver.interface.Objective(
        sym.Add(*rh_obj) + sym.Add(*rl_obj), direction="max"
    )
    model.objective = imat_obj


def add_imat_objective(
    model: cobra.Model, rxn_weights: Union[pd.Series, dict]
) -> cobra.Model:
    """
    Add the IMAT objective to the model (doesn't change passed model).
    Model must already have iMAT constraints added.

    :param model: A cobra.Model object to update with iMAT constraints.
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :return: None
    """
    imat_model = model.copy()
    _enforce_binary(imat_model)
    add_imat_objective_(imat_model, rxn_weights)
    return imat_model


# endregion: iMAT Helper Functions


# region: Internal Methods
def _imat_pos_weight_(model: cobra.Model, rxn: str, epsilon: float) -> None:
    """
    Internal method for adding positive weight constraints to the model.

    :param model: A cobra.Model object to update with iMAT constraints.
    :type model: cobra.Model
    :param rxn: The reaction ID to add the constraint to.
    :type rxn: str
    :param epsilon: The epsilon value to use for iMAT (default: 1e-3).
        Represents the minimum flux for a reaction to be considered on.
    :type epsilon: float
    :return: None
    """
    reaction = model.reactions.get_by_id(rxn)
    lb = reaction.lower_bound
    ub = reaction.upper_bound
    reaction_flux = reaction.forward_variable - reaction.reverse_variable
    y_pos = model.solver.interface.Variable(f"y_pos_{reaction.id}", type="binary")
    model.solver.add(y_pos)
    forward_constraint = model.solver.interface.Constraint(
        reaction_flux + y_pos * (lb - epsilon),
        lb=lb,
        name=f"forward_constraint_{reaction.id}",
    )
    model.solver.add(forward_constraint)
    y_neg = model.solver.interface.Variable(f"y_neg_{reaction.id}", type="binary")
    model.solver.add(y_neg)
    reverse_constraint = model.solver.interface.Constraint(
        reaction_flux + y_neg * (ub + epsilon),
        ub=ub,
        name=f"reverse_constraint_{reaction.id}",
    )
    model.solver.add(reverse_constraint)


def _imat_neg_weight_(model: cobra.Model, rxn: str, threshold: float) -> None:
    """
    Internal method for adding negative weight constraints to the model.

    :param model: A cobra.Model object to update with iMAT constraints.
    :type model: cobra.Model
    :param rxn: The reaction ID to add the constraint to.
    :type rxn: str
    :param threshold: The threshold value to use for iMAT (default: 1e-4).
        Represents the maximum flux for a reaction to be considered off.
    :type threshold: float
    :return: None
    """
    reaction = model.reactions.get_by_id(rxn)
    lb = reaction.lower_bound
    ub = reaction.upper_bound
    reaction_flux = reaction.forward_variable - reaction.reverse_variable
    y_pos = model.solver.interface.Variable(f"y_pos_{reaction.id}", type="binary")
    model.solver.add(y_pos)
    forward_constraint = model.solver.interface.Constraint(
        reaction_flux - ub * (1 - y_pos) - threshold * y_pos,
        ub=0,
        name=f"forward_constraint_{reaction.id}",
    )
    model.solver.add(forward_constraint)
    reverse_constraint = model.solver.interface.Constraint(
        reaction_flux - lb * (1 - y_pos) + threshold * y_pos,
        lb=0,
        name=f"reverse_constraint_{reaction.id}",
    )
    model.solver.add(reverse_constraint)


def _enforce_binary(model: cobra.Model):
    """
    Internal method for enforcing binary type for added binary variables
    """
    for var in model.solver.variables:
        if BINARY_REGEX.search(var.name):
            model.solver.variables[var.name].type = "binary"


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


# endregion: Internal Methods
