"""
Module for model utilities
"""

# External Imports
from __future__ import annotations
import cobra
import optlang.container
from sympy import parse_expr


# region Model IO
def read_model(model_path, file_type=None):
    """
    Read a model from a file

    :param model_path: Path to the model file
    :type model_path: str | pathlib.Path
    :param file_type: Type of the file
    :type file_type: str
    :return: The model
    """
    if file_type is None:
        model_path = str(model_path)
        file_type = model_path.split(".")[-1]
    file_type = _parse_file_type(file_type)
    if file_type == "joblib":
        from joblib import load

        model = load(model_path)
    elif file_type == "pickle":
        import pickle

        with open(model_path, "rb") as f:
            model = pickle.load(f)
    elif file_type == "sbml":
        from cobra.io import read_sbml_model

        model = read_sbml_model(model_path)
    elif file_type == "yaml":
        from cobra.io import load_yaml_model

        model = load_yaml_model(model_path)
    elif file_type == "json":
        from cobra.io import load_json_model

        model = load_json_model(model_path)
    elif file_type == "mat":
        from cobra.io import load_matlab_model

        model = load_matlab_model(model_path)
    else:
        raise ValueError("File type not supported")
    return model


def write_model(model, model_path, file_type=None):
    """
    Write a model to a file

    :param model: Model to write
    :type model: cobra.Model
    :param model_path: Path to the model file
    :type model_path: str
    :param file_type: Type of the file
    :type file_type: str
    :return: Nothing
    """
    if file_type is None:
        model_path = str(model_path)
        file_type = model_path.split(".")[-1]
    file_type = _parse_file_type(file_type)
    if file_type == "joblib":
        from joblib import dump

        dump(model, model_path)
    elif file_type == "pickle":
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    elif file_type == "sbml":
        from cobra.io import write_sbml_model

        write_sbml_model(model, model_path)
    elif file_type == "yaml":
        from cobra.io import save_yaml_model

        save_yaml_model(model, model_path)
    elif file_type == "json":
        from cobra.io import save_json_model

        save_json_model(model, model_path)
    elif file_type == "mat":
        from cobra.io import save_matlab_model

        save_matlab_model(model, model_path)
    else:
        raise ValueError("File type not supported")


def _parse_file_type(file_type):
    """
    Parse the file type
    :param file_type: File type to parse
    :type file_type: str
    :return: Parsed file type
    :rtype: str
    """
    if file_type.lower() in ["json", "jsn"]:
        return "json"
    elif file_type.lower() in ["yaml", "yml"]:
        return "yaml"
    elif file_type.lower() in ["sbml", "xml"]:
        return "sbml"
    elif file_type.lower() in ["mat", "m", "matlab"]:
        return "mat"
    elif file_type.lower() in ["joblib", "jl", "jlb"]:
        return "joblib"
    elif file_type.lower() in ["pickle", "pkl"]:
        return "pickle"
    else:
        raise ValueError("File type not supported")


# endregion: Model IO


# region Model Comparison
def model_eq(model1: cobra.Model, model2: cobra.Model, verbose: bool = False) -> bool:
    """
    Check if two cobra models are equal.

    :param model1: The first model to compare.
    :type model1: cobra.Model
    :param model2: The second model to compare.
    :type model2: cobra.Model
    :param verbose: Whether to print where the models differ (default: False).
    :type verbose: bool
    :return: True if the models are equal, False otherwise.
    :rtype: bool
    """
    if verbose:
        print("Verbose model comparison")
    # Check metabolites, reactions, and genes
    if not _check_dictlist_eq(model1.metabolites, model2.metabolites):
        if verbose:
            print("Models have different metabolites")
        return False
    if not _check_dictlist_eq(model1.reactions, model2.reactions):
        if verbose:
            print("Models have different reactions")
        return False
    if not _check_dictlist_eq(model1.genes, model2.genes):
        if verbose:
            print("Models have different genes")
        return False
    # Check reaction equality (guaranteed they have the same reactions)
    for reaction1 in model1.reactions:
        reaction2 = model2.reactions.get_by_id(reaction1.id)
        if not _check_reaction_eq(reaction1, reaction2, verbose=verbose):
            return False
    # Check objective
    if not _check_objective_eq(model1.objective, model2.objective, verbose=verbose):
        if verbose:
            print("Models have different objectives")
            print(f"Model 1 objective: {model1.objective}")
            print(f"Model 2 objective: {model2.objective}")
        return False
    # Check the underlying constraint model
    if not _check_optlang_container_eq(
        model1.solver.constraints, model2.solver.constraints
    ):
        if verbose:
            print("Models have different constraints")
        return False
    if not _check_optlang_container_eq(
        model1.solver.variables, model2.solver.variables
    ):
        if verbose:
            print("Models have different variables")
        return False

    # Checking more specifics for the variables
    for var1 in model1.variables:
        var2 = model2.variables[var1.name]
        if not _check_variable_eq(var1, var2, verbose=verbose):
            return False

    # Checking more specifics for the constraints
    for const1 in model1.constraints:
        const2 = model2.constraints[const1.name]
        if not _check_constraint_eq(const1, const2, verbose=verbose):
            return False
    return True


def _check_dictlist_subset(
    dictlist1: cobra.DictList, dictlist2: cobra.DictList
) -> bool:
    """
    Check if dictlist1 is a subset of dictlist2.

    :param dictlist1: The first dictlist to compare.
    :type dictlist1: cobra.DictList
    :param dictlist2: The second dictlist to compare.
    :type dictlist2: cobra.DictList
    :return: True if dictlist1 is a subset of dictlist2, False otherwise.
    :rtype: bool
    """
    for val in dictlist1:
        if val not in dictlist2:
            return False
    return True


def _check_dictlist_eq(dictlist1: cobra.DictList, dictlist2: cobra.DictList) -> bool:
    """
    Check if two dictlists are equal.

    :param dictlist1: The first dictlist to compare.
    :type dictlist1: cobra.DictList
    :param dictlist2: The second dictlist to compare.
    :type dictlist2: cobra.DictList
    :return: True if the dictlists are equal, False otherwise.
    :rtype: bool
    """
    if not _check_dictlist_subset(dictlist1, dictlist2):
        return False
    if not _check_dictlist_subset(dictlist2, dictlist1):
        return False
    return True


def _check_optlang_container_subset(
    cont1: optlang.container.Container, cont2: optlang.container.Container
) -> bool:
    """
    Check if cont1 is a subset of cont2.

    :param cont1: The first container to compare.
    :type cont1: optlang.container.Container
    :param cont2: The second container to compare.
    :type cont2: optlang.container.Container
    :return: True if cont1 is a subset of cont2, False otherwise.
    :rtype: bool
    """
    for val in cont1:
        if val.name not in cont2:
            return False
    return True


def _check_optlang_container_eq(
    cont1: optlang.container.Container, cont2: optlang.container.Container
) -> bool:
    """
    Check if two optlang containers are equal.

    :param cont1: The first container to compare.
    :type cont1: optlang.container.Container
    :param cont2: The second container to compare.
    :type cont2: optlang.container.Container
    :return: True if the containers are equal, False otherwise.
    :rtype: bool
    """
    if not _check_optlang_container_subset(cont1, cont2):
        return False
    if not _check_optlang_container_subset(cont2, cont1):
        return False
    return True


def _check_reaction_eq(
    rxn1: cobra.Reaction, rxn2: cobra.Reaction, verbose: bool = False
) -> bool:
    """
    Check if two reactions are equal.

    :param rxn1: The first reaction to compare.
    :type rxn1: cobra.Reaction
    :param rxn2: The second reaction to compare.
    :type rxn2: cobra.Reaction
    :param verbose: Whether to print where the reactions differ
    (default: False).
    :type verbose: bool
    :return: True if the reactions are equal, False otherwise.
    :rtype: bool
    """
    if rxn1.lower_bound != rxn2.lower_bound:
        if verbose:
            print(f"Reaction {rxn1.id} has different lower bounds")
        return False
    if rxn1.upper_bound != rxn2.upper_bound:
        if verbose:
            print(f"Reaction {rxn1.id} has different upper bounds")
        return False
    if rxn1.gene_reaction_rule != rxn2.gene_reaction_rule:
        if verbose:
            print(f"Reaction {rxn1.id} has different GPR")
        return False
    if rxn1.name != rxn2.name:
        if verbose:
            print(f"Reaction {rxn1.id} has different names")
        return False
    if rxn1.subsystem != rxn2.subsystem:
        if verbose:
            print(f"Reaction {rxn1.id} has different subsystems")
        return False
    if rxn1.objective_coefficient != rxn2.objective_coefficient:
        if verbose:
            print(f"Reaction {rxn1.id} has different objective coefficients")
        return False
    return True


def _check_expression_eq(expr1, expr2, verbose=False) -> bool:
    """
    Check if two sympy or optlang expressions are equal.

    :param expr1: The first expression to compare.
    :type expr1: sympy.Expr or optlang.Expression
    :param expr2: The second expression to compare.
    :type expr2: sympy.Expr or optlang.Expression
    :param verbose: Whether to print where the expressions differ
        (default: False).
    :type verbose: bool
    :return: True if the expressions are equal, False otherwise.
    :rtype: bool
    """
    if parse_expr(str(expr1)) - parse_expr(str(expr2)) != 0:
        print("ENTERED IF STATEMENT")
        if verbose:
            print(f"Expressions {expr1} and {expr2} are not equal")
        return False
    return True


def _check_objective_eq(objective1, objective2, verbose=False) -> bool:
    """
    Check if two objectives are equal.

    :param objective1: The first objective to compare.
    :type objective1: cobra.core.objective.Objective
    :param objective2: The second objective to compare.
    :type objective2: cobra.core.objective.Objective
    :param verbose: Whether to print where the objectives differ
        (default: False).
    :type verbose: bool
    :return: True if the objectives are equal, False otherwise.
    :rtype: bool
    """
    expr1 = objective1.expression
    expr2 = objective2.expression
    if not _check_expression_eq(expr1, expr2, verbose=verbose):
        if verbose:
            print("Expressions of the objectives are different")
        return False
    if objective1.direction != objective2.direction:
        if verbose:
            print("Directions of the objectives are different")
        return False
    return True


def _check_variable_eq(var1, var2, verbose: bool = False) -> bool:
    if var1.lb != var2.lb:
        if verbose:
            print(f"Variable {var1.name} has different lower bounds")
        return False
    if var1.ub != var2.ub:
        if verbose:
            print(f"Variable {var1.name} has different upper bounds")
        return False
    if var1.type != var2.type:
        if verbose:
            print(f"Variable {var1.name} has different types")
        return False
    return True


def _check_constraint_eq(constraint1, constraint2, verbose: bool = False) -> bool:
    if constraint1.lb != constraint2.lb:
        if verbose:
            print(f"Constraint {constraint1.name} has different lower bounds")
        return False
    if constraint1.ub != constraint2.ub:
        if verbose:
            print(f"Constraint {constraint1.name} has different upper bounds")
        return False
    if not _check_expression_eq(
        constraint1.expression, constraint2.expression, verbose=verbose
    ):
        if verbose:
            print(f"Constraint {constraint1.name} has different expressions")
        return False
    return True


# endregion: Model Comparison
