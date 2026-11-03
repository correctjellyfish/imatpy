# Standard Library Imports
import random
import os
import pathlib
import unittest

# External Imports
from cobra.core.configuration import Configuration
import pandas as pd

# Local Imports
from imatpy.parse_gpr import (
    str_to_list,
    to_postfix,
    eval_gpr,
    gene_to_rxn_weights,
)
from imatpy.model_utils import read_model


class TestStrToList(unittest.TestCase):
    def test_token_parse(self):
        self.assertEqual(str_to_list("Rv0031"), ["Rv0031"])
        self.assertEqual(str_to_list(""), [])

    def test_parenthesis_parse(self):
        self.assertEqual(str_to_list("(Rv0031)"), ["(", "Rv0031", ")"])
        self.assertEqual(str_to_list("(Rv0031"), ["(", "Rv0031"])
        self.assertEqual(str_to_list("Rv0031)"), ["Rv0031", ")"])
        self.assertEqual(
            str_to_list("(Rv0031)AND"), ["(", "Rv0031", ")", "AND"]
        )

    def test_operator_replacements(self):
        self.assertEqual(
            str_to_list("Rv0031 and Rv0098"), ["Rv0031", "AND", "Rv0098"]
        )
        self.assertEqual(
            str_to_list("Rv0031 anD Rv0098"), ["Rv0031", "AND", "Rv0098"]
        )
        self.assertEqual(
            str_to_list("Rv0031 And Rv0098"), ["Rv0031", "AND", "Rv0098"]
        )
        self.assertEqual(
            str_to_list("Rv0031 or Rv0098"), ["Rv0031", "OR", "Rv0098"]
        )
        self.assertEqual(
            str_to_list("Rv0031 | Rv0098"), ["Rv0031", "OR", "Rv0098"]
        )
        self.assertEqual(
            str_to_list("Rv0031 || Rv0098"), ["Rv0031", "OR", "Rv0098"]
        )
        self.assertEqual(
            str_to_list("Rv0031|Rv0098"), ["Rv0031", "OR", "Rv0098"]
        )
        self.assertEqual(
            str_to_list("Rv0031||Rv0098"), ["Rv0031", "OR", "Rv0098"]
        )
        self.assertEqual(
            str_to_list("Rv0031 & Rv0098"), ["Rv0031", "AND", "Rv0098"]
        )
        self.assertEqual(
            str_to_list("Rv0031 && Rv0098"), ["Rv0031", "AND", "Rv0098"]
        )
        self.assertEqual(
            str_to_list("Rv0031&Rv0098"), ["Rv0031", "AND", "Rv0098"]
        )
        self.assertEqual(
            str_to_list("Rv0031&&Rv0098"), ["Rv0031", "AND", "Rv0098"]
        )

    def test_replacement_dict(self):
        self.assertEqual(
            str_to_list("Rv0031 n Rv0098", {"n": "NOT"}),
            ["Rv0031", "NOT", "Rv0098"],
        )
        self.assertEqual(
            str_to_list("Rv0031 n Rv0098", {"n": "NOT", "Rv": "rv"}),
            ["rv0031", "NOT", "rv0098"],
        )
        self.assertEqual(
            str_to_list("~Rv0031", {"~": "NOT "}), ["NOT", "Rv0031"]
        )


class TestToPostfix(unittest.TestCase):
    def test_single_token(self):
        self.assertEqual(to_postfix(["Rv0031"]), ["Rv0031"])

    def test_single_infix(self):
        self.assertEqual(
            to_postfix(["Rv0031", "AND", "Rv0098"]),
            ["Rv0031", "Rv0098", "AND"],
        )

    def test_parenthesis(self):
        self.assertEqual(
            to_postfix(["(", "Rv0031", "AND", "Rv0098", ")", "OR", "Rv1234"]),
            ["Rv0031", "Rv0098", "AND", "Rv1234", "OR"],
        )

    def test_known_postfix_precedence(self):
        input_expr = ["5", "*", "2", "-", "1"]
        precedence = {"*": 2, "-": 1}
        output_expr = ["5", "2", "*", "1", "-"]
        self.assertEqual(to_postfix(input_expr, precedence), output_expr)

    def test_known_postfix_parenthesis(self):
        input_expr = ["5", "*", "(", "2", "-", "3", ")"]
        precedence = {"*": 2, "-": 1}
        output_expr = ["5", "2", "3", "-", "*"]
        self.assertEqual(to_postfix(input_expr, precedence), output_expr)


class TestEvalGpr(unittest.TestCase):
    def test_single_gene(self):
        gpr_str = "Rv0031"
        gene_weights = pd.Series({"Rv0031": 1})
        self.assertEqual(eval_gpr(gpr_str, gene_weights), 1)

    def test_two_gene(self):
        gpr_str = "Rv0031 AND Rv0098"
        gene_weights = pd.Series({"Rv0031": 1, "Rv0098": -1})
        self.assertEqual(eval_gpr(gpr_str, gene_weights), -1)
        gpr_str = "Rv0031 OR Rv0098"
        self.assertEqual(eval_gpr(gpr_str, gene_weights), 1)

    def test_parenthesis(self):
        gpr_string = "(Rv0031 AND Rv0098) OR Rv1234"
        gene_weights = pd.Series({"Rv0031": 1, "Rv0098": -1, "Rv1234": 1})
        self.assertEqual(eval_gpr(gpr_string, gene_weights), 1)
        gpr_string = "(Rv0031 AND Rv0098) OR (Rv1234 AND Rv0098)"
        self.assertEqual(eval_gpr(gpr_string, gene_weights), -1)


class TestGeneToRxnWeights(unittest.TestCase):
    test_model = None

    @classmethod
    def setUpClass(cls):
        Configuration().solver = "glpk"
        data_path = pathlib.Path(__file__).parent.joinpath("data")
        test_model_path = os.path.join(data_path, "test_model.json")
        cls.test_model = read_model(test_model_path)
        textbook_model_path = os.path.join(data_path, "textbook_model.json")
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
        rxn_weights = gene_to_rxn_weights(self.textbook_model, gene_weights)
        self.assertIsInstance(rxn_weights, pd.Series)
        self.assertEqual(len(rxn_weights), len(self.textbook_model.reactions))
        self.assertEqual(rxn_weights.dtype, float)


if __name__ == "__main__":
    unittest.main()
