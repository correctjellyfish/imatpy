# Standard Library Imports
from __future__ import annotations

import ast
import functools
import warnings
from collections.abc import Iterable
from typing import Any

# External Imports
import cobra
import pandas as pd

# Local imports

# Global function dictionary declarations
METCHANGE_FUNC_DICT = {"AND": max, "OR": min}
IMAT_FUNC_DICT = {"AND": min, "OR": max}


def gene_to_rxn_weights(
    model: cobra.Model,
    gene_weights: pd.Series,
    fn_dict: dict | None = None,
    fill_val: Any = 0,
) -> pd.Series:
    """Convert a gene weights series to a reaction weights series using the
    provided function dictionary.

    Parameters
    ----------
    model : cobra.Model
        cobra.Model: A cobra model
    gene_weights : pd.Series or dict of str to Value
        pd.Series: A series of gene weights
    fn_dict : dict
        dict: A dictionary of functions to use for each operator
    fill_val : Any
        Any: The value to fill missing values with

    Returns
    -------
    pd.Series
        A series of reaction weights

    Notes
    -----
    The fill value is applied to fill NaN values after the GPR rules have been
    applied.
    If there are genes missing from the expression data, they will silently be
    assigned a value of 0 before the GPR processing is performed.
    """
    # Check that all genes in the model are in the gene expression data,
    # and if not add them with a weight of 0
    model_genes = set(model.genes.list_attr("id"))
    expr_genes = set(gene_weights.index)
    missing_genes = list(model_genes - expr_genes)
    if missing_genes:
        warnings.warn(
            f"Genes {missing_genes} are in model but not in gene weights, "
            f"setting their weight to {fill_val}."
        )
        missing_genes_series = pd.Series(0, index=missing_genes)
        gene_weights = pd.concat([gene_weights, missing_genes_series])

    # Convert the fill_val into the same type as the gene_weights
    fill_val = pd.Series(fill_val, dtype=gene_weights.dtype).iloc[0]

    # Create the rxn_weight series, filled with all 0
    rxn_weights = pd.Series(
        fill_val, index=pd.Index(model.reactions.list_attr("id"))
    )

    if fn_dict is None:
        fn_dict = IMAT_FUNC_DICT
    # For each reaction, trinarize it based on the expression data
    for rxn in model.reactions:
        rxn_weights[rxn.id] = eval_gpr(
            rxn.gpr, gene_weights, fn_dict, fill_val
        )
    return rxn_weights


def eval_gpr(
    gpr: cobra.core.GPR | ast.Expression | list | ast.BoolOp | ast.Name | None,
    gene_weights: pd.Series | dict,
    fn_dict: dict,
    fill_val: Any = 0,
) -> Any:
    """
    Evaluate a cobra GPR with specified functions for the Boolean operations

    Parameters
    ----------
    gpr : GPR or Expression or list or BoolOp or Name optional
        The GPR to evaluate
    gene_weights: pd.Series or dict
        Weights to assign to each gene
    fn_dict : dict
        Dict of 'AND' and 'OR' (strings) to functions which can
        take two gene weights and return a single value
    fill_val : Any
        Value to replace any missing weights with
    """
    if isinstance(gpr, (ast.Expression, cobra.core.GPR)):
        if not gpr.body:
            return fill_val
        return eval_gpr(
            gpr=gpr.body,  # type:ignore
            gene_weights=gene_weights,
            fn_dict=fn_dict,
            fill_val=fill_val,
        )
    elif isinstance(gpr, ast.Name):
        return gene_weights.get(gpr.id, fill_val)
    elif isinstance(gpr, ast.BoolOp):
        op = gpr.op
        if isinstance(op, ast.Or):
            return fn_dict["OR"](
                *[
                    eval_gpr(e, gene_weights, fn_dict, fill_val)  # type: ignore
                    for e in gpr.values
                ]
            )
        elif isinstance(op, ast.And):
            return fn_dict["AND"](
                *[
                    eval_gpr(e, gene_weights, fn_dict, fill_val)  # type: ignore
                    for e in gpr.values
                ]
            )
        else:
            raise TypeError(
                f"Unsupported Boolean Operation: {op.__class__.__name__}"
            )
    elif gpr is None:
        return fill_val
    raise TypeError(f"Unsupported GPR type: {type(gpr)}")


def gene_to_reaction_ids(
    model: cobra.Model, gene: str, essential: bool = False
) -> set[str]:
    """
    Convert a gene id into associated reaction ids

    Parameters
    ----------
    model : cobra.Model
        Model to use for performing the translation
    gene : str
        id for gene to translate into reaciton ids
    essential : bool, default=False
        Whether the reactions should only be those for
        which the gene is required

    Returns
    -------
    reaction_set : set[str]
        Set of reactions associated with a particular gene
    """
    gene_obj = model.genes.get_by_id(gene)
    rxn_set = gene_obj.reactions
    if len(rxn_set) == 0:
        return set()
    if not essential:
        return {r.id for r in rxn_set}
    associated_reaction_set = set()
    for rxn in rxn_set:
        if not rxn.gpr.eval(gene):
            associated_reaction_set.add(rxn.id)
    return associated_reaction_set


def gene_to_reaction_list(
    model: cobra.Model, gene_list: Iterable[str], essential: bool = False
):
    """
    Convert a list (or other Iterable) of gene ids into a list of associated reaction ids

    Parameters
    ----------
    model : cobra.Model
        Model to use for performing the translation
    gene_list : list of str
        list of gene ids to translate
    essential : bool, default=False
        Whether the reactions should only be those for
        which the genes are required

    Returns
    -------
    reaction_list : list[str]
        list of reactions associated with the genes in `gene_list`

    See Also
    --------
    gene_group_to_reaction_list : Translate genes as a group

    Notes
    -----
    The order of the genes is not perseved in the order of the reactions,
    and each gene is translated individually (even if both genes in an OR
    relationship are included `gene_list`, that reaction won't be included
    if `essential` is True). For treating the genes as a group,
    see `gene_group_to_reaction_list`.
    """
    return list(
        functools.reduce(
            lambda x, y: x | y,  # ty: ignore[unsupported-operator]
            (
                gene_to_reaction_ids(model=model, gene=g, essential=essential)
                for g in gene_list
            ),
            set(),
        )
    )


def gene_group_to_reaction_list(
    model: cobra.Model, gene_list: Iterable[str], essential: bool = False
):
    """
    Convert a list (or other Iterable) of gene ids into a list of
    associated reaction ids

    Parameters
    ----------
    model : cobra.Model
        Model to use for performing the translation
    gene_list : list of str
        list of gene ids to translate
    essential : bool, default=False
        Whether the reactions should only be those for
        which the genes are required

    Returns
    -------
    reaction_list : list[str]
        list of reactions associated with the genes in `gene_list`

    See Also
    --------
    gene_to_reaction_list : Individually translate genes to reactions

    Notes
    -----
    Genes are translated as a group, so that when essential is True,
    if a reaction requires an OR between genes to be True, if both
    of the genes in the OR expression are in gene_list, then
    the reaction is included. This contrasts with
    `gene_to_reaction_list` which translates genes individually.
    """
    if essential is False:
        return gene_to_reaction_list(
            model=model, gene_list=gene_list, essential=False
        )
    gene_series = pd.Series(-1, index=pd.Index(gene_list))
    rxn_series = gene_to_rxn_weights(
        model=model,
        gene_weights=gene_series,
        fn_dict=IMAT_FUNC_DICT,
        fill_val=0,
    )
    return list(rxn_series[rxn_series == -1].index)
