# Standard Library Imports
import pathlib
import random
import unittest

# External Imports
import cobra
import pandas as pd
from cobra.core.configuration import Configuration

# Local Imports
from imatpy.model_utils import read_model
from imatpy.parse_gpr import (
    IMAT_FUNC_DICT,
    eval_gpr,
    gene_to_rxn_weights,
)


class TestEvalGpr(unittest.TestCase):
    def test_single_gene(self):
        gpr = cobra.core.gene.GPR.from_string("g001")
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series([1.0], index=["g001"]),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 1.0)

    def test_or(self):
        gpr = cobra.core.gene.GPR.from_string("g001 or g002")
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series([1.0, 0.0], index=["g001", "g002"]),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 1.0)
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series([0.0, 0.0], index=["g001", "g002"]),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 0.0)
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series([0.0, 1.0], index=["g001", "g002"]),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 1.0)
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series([1.0, 1.0], index=["g001", "g002"]),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 1.0)

    def test_and(self):
        gpr = cobra.core.gene.GPR.from_string("g001 and g002")
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series([1.0, 0.0], index=["g001", "g002"]),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 0.0)
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series([0.0, 0.0], index=["g001", "g002"]),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 0.0)
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series([0.0, 1.0], index=["g001", "g002"]),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 0.0)
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series([1.0, 1.0], index=["g001", "g002"]),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 1.0)

    def test_arity_3(self):
        gpr = cobra.core.gene.GPR.from_string("g001 and g002 or g003 ")
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series(
                [1.0, 0.0, 1.0], index=["g001", "g002", "g003"]
            ),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 1.0)
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series(
                [1.0, 1.0, 0.0], index=["g001", "g002", "g003"]
            ),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 1.0)
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series(
                [0.0, 1.0, 0.0], index=["g001", "g002", "g003"]
            ),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 0.0)

    def test_repeated(self):
        gpr = cobra.core.gene.GPR.from_string("g001 or g002 or g003 or g004")
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series(
                [1.0, 0.0, 1.0, 0.0], index=["g001", "g002", "g003", "g004"]
            ),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 1.0)
        gpr_res = eval_gpr(
            gpr,
            gene_weights=pd.Series(
                [0.0, 0.0, 0.0, 0.0], index=["g001", "g002", "g003", "g004"]
            ),
            fn_dict=IMAT_FUNC_DICT,
            fill_val=0,
        )
        self.assertAlmostEqual(gpr_res, 0.0)


class TestGeneToRxnWeights(unittest.TestCase):
    test_model = None

    @classmethod
    def setUpClass(cls):
        Configuration().solver = "glpk"
        data_path = pathlib.Path(__file__).parent.joinpath("data")
        test_model_path = data_path / "test_model.json"
        cls.test_model = read_model(test_model_path)
        textbook_model_path = data_path / "textbook_model.json"
        cls.textbook_model = read_model(textbook_model_path)
        cls.test_model_weights = pd.Series(
            {
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
            }
        )
        cls.test_model_rxn_weights = pd.Series(
            {
                "R_A_e_ex": 0.0,
                "R_B_e_ex": 0.0,
                "R_C_e_ex": 0.0,
                "R_F_e_ex": 0.0,
                "R_G_e_ex": 0.0,
                "R_H_e_ex": 0.0,
                "R_A_imp": 1,
                "R_B_imp": -1,
                "R_C_imp": -1,
                "R_F_exp": 0,
                "R_G_exp": -1,
                "R_H_exp": 0,
                "r_A_B_D_E": 0,
                "r_C_E_F": -1,
                "r_C_H": 0,
                "r_D_G": 1,
            }
        )

    def test_simple_model(self):
        assert self.test_model is not None, "Failed to correctly setup test"
        rxn_weights = gene_to_rxn_weights(
            self.test_model, self.test_model_weights
        )
        self.assertTrue(rxn_weights.__eq__(self.test_model_rxn_weights).all())

    def test_larger_model(self):
        gene_weights = pd.Series(
            0.0, index=[gene.id for gene in self.textbook_model.genes]
        )
        for gene in gene_weights.index:
            gene_weights[gene] = random.choice([-1.0, 0.0, 1.0])
        rxn_weights = gene_to_rxn_weights(
            self.textbook_model, gene_weights, fill_val=0
        )
        self.assertIsInstance(rxn_weights, pd.Series)
        self.assertEqual(len(rxn_weights), len(self.textbook_model.reactions))
        self.assertEqual(rxn_weights.dtype, float)


if __name__ == "__main__":
    unittest.main()
