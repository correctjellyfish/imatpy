# Standard Library Imports
from __future__ import annotations

import re
from typing import Any

# External Imports
import cobra
import pandas as pd


def gene_to_rxn_weights(
    model: cobra.Model,
    gene_weights: pd.Series,
    fn_dict: dict | None = None,
    fill_val: Any = 0,
) -> pd.Series:
    """
    Convert a gene weights series to a reaction weights series using the
    provided function dictionary.

    :param model: cobra.Model: A cobra model
    :type model: cobra.Model
    :param gene_weights: pd.Series: A series of gene weights
    :type gene_weights: pd.Series
    :param fn_dict: dict: A dictionary of functions to use for each operator
    :type fn_dict: dict
    :param fill_val: Any: The value to fill missing values with
    :type fill_val: Any
    :return: A series of reaction weights
    :rtype: pd.Series
    """
    rxn_weights = pd.Series(0, index=[rxn.id for rxn in model.reactions])
    for rxn in model.reactions:
        gpr = rxn.gene_reaction_rule
        rxn_weights[rxn.id] = eval_gpr(gpr, gene_weights, fn_dict)
    rxn_weights.fillna(fill_val, inplace=True)
    return rxn_weights


def eval_gpr(
    gpr: str, gene_weights: pd.Series, fn_dict: dict | None = None
) -> Any | None:
    """
    Evaluate a single GPR string using the provided gene weights and
    function dictionary.

    :param gpr: str: A single GPR string
    :type gpr: str
    :param gene_weights: pd.Series: A series of gene weights
    :type gene_weights: pd.Series
    :param fn_dict: dict: A dictionary of functions to use for each operator
    :type fn_dict: dict
    :return: The GPR score
    :rtype: float | None
    """
    if not gpr:  # If GPR is empty string, return None
        return None
    if fn_dict is None:
        fn_dict = {"AND": min, "OR": max}
    gpr_expr = str_to_list(gpr)
    gpr_expr = to_postfix(gpr_expr)
    eval_stack = []
    for token in gpr_expr:
        if token not in fn_dict:
            eval_stack.append(gene_weights[token])
            continue
        val1 = eval_stack.pop()
        val2 = eval_stack.pop()
        eval_stack.append(fn_dict[token](val1, val2))
    if len(eval_stack) != 1:
        raise ValueError(f"Failed to parse GPR: {gpr}")
    return eval_stack.pop()


def str_to_list(in_string: str, replacements: dict | None = None) -> list[str]:
    """
    Convert a string to a list of strings, splitting on whitespace and
    parentheses.

    :param in_string: str: Specify the input string
    :type in_string: str
    :param replacements: dict: Replace certain strings with other strings
        before splitting, uses regex
    :type replacements: dict
    :return: A list of strings
    :rtype: list[str]
    """
    if not replacements:
        replacements = {
            "\\b[Aa][Nn][Dd]\\b": "AND",
            "\\b[Oo][Rr]\\b": "OR",
            "&&?": " AND ",
            r"\|\|?": " OR ",
        }
    in_string = in_string.replace("(", " ( ").replace(")", " ) ")
    for key, value in replacements.items():
        in_string = re.sub(key, value, in_string)
    return in_string.split()


def process_token(token, postfix, operator_stack, precedence):
    """
    The process_token function takes in a token, the postfix list, the
    operator stack and precedence dictionary. It performs the shunting
    yard algorithm for a the single provided token.

    :param token: Current token
    :type token: str
    :param postfix: Current state of output
    :type postfix: list[str]
    :param operator_stack: Current operator stack
    :type operator_stack: list[str]
    :param precedence: Determines the operators precedence
    :type precedence: dict[str:int]
    :return: Nothing
    :rtype: None
    """
    # If token is not operator, add it to postfix
    if (token not in precedence) and (token != "(") and (token != ")"):
        postfix.append(token)
        return
    # If token is operator, move higher priority operators from stack to
    # output, then add the operator itself to the postfix expression
    if token in precedence:
        while (
            (len(operator_stack) > 0)
            and (operator_stack[-1] != "(")
            and (precedence[operator_stack[-1]] >= precedence[token])
        ):
            op = operator_stack.pop()
            postfix.append(op)
        operator_stack.append(token)
        return
    # For left parenthesis add to operator stack
    if token == "(":
        operator_stack.append(token)
        return
    # For right parenthesis pop operator stack until reach
    # matching left parenthesis
    if token == ")":
        if len(operator_stack) == 0:  # Check for mismatch in parentheses
            raise ValueError("Mismatched Parenthesis in Expression")
        while len(operator_stack) > 0 and operator_stack[-1] != "(":
            op = operator_stack.pop()
            postfix.append(op)
        if (
            len(operator_stack) == 0 or operator_stack[-1] != "("
        ):  # Check for mismatch in parentheses
            raise ValueError("Mismatched Parenthesis in Expression")
        _ = operator_stack.pop()  # Remove left paren from stack
        return


def to_postfix(infix: list[str], precedence: dict | None = None) -> list[str]:
    """
    Convert an infix expression to postfix notation.
    :param infix: list[str]: A list of strings representing an infix expression
    :type infix: list[str]
    :param precedence: Dictionary of operators determining precedence
    :type precedence: dict[str:int]
    :return: A list of strings representing the postfix expression
    :rtype: list[str]
    """
    # Set default precedence
    if precedence is None:
        precedence = {"AND": 1, "OR": 1}
    postfix = []
    operator_stack = []
    # For each token, use shunting yard algorithm to process it
    for token in infix:
        process_token(token, postfix, operator_stack, precedence)
    # Empty the operator stack
    while len(operator_stack) > 0:
        op = operator_stack.pop()
        if op == "(":
            raise ValueError("Mismatched Parenthesis in Expression")
        postfix.append(op)
    return postfix
