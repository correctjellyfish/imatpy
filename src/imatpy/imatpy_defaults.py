"""
Submodules for some shared default values
"""

from dataclasses import dataclass

from cobra.core import Configuration


@dataclass
class ImatDefaults:
    """
    Default values for iMAT (and related functionality) parameters

    Attributes
    ----------
    epsilon : float
        Flux cutoff, above which a reaction is considered active
    threshold : float
        Flux cutoff, below which a reaction is considered inactive
    solver_tolerance : float
        Tolerance for the solver
    objective_tolerance : float
        For FVA method, the tolerance allowed in the objective function. A value
        of 0.1 would require that the iMAT objective would be within 10% of the
        optimum iMAT objective when finding reaction bounds using FVA.
    max_iter : int
        For iMAT Iterator methods, the maximum number of iterations to perform
    """

    epsilon: float = 1.0
    threshold: float = 1e-2
    solver_tolerance: float = Configuration().tolerance
    objective_tolerance: float = 5e-2  # For FVA method
    max_iter: int = 20  # For iMAT Iter


# The way that python imports works makes this work like a singleton
IMAT_DEFAULTS = ImatDefaults()
