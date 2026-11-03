# Standard Library Imports
import copy
import importlib.util
import pathlib
import unittest
from unittest import skipIf

# External Imports
from cobra.core.configuration import Configuration
import pandas as pd

# Local Imports
import imatpy.imat
from imatpy.model_utils import read_model, model_eq, _check_expression_eq


def setup(cls):
    Configuration().solver = "glpk"  # Use GLPK solver for testing
    cls.data_path = pathlib.Path(__file__).parent.absolute() / "data"
    cls.model = read_model(cls.data_path / "test_model.xml")
    cls.rxn_weights = pd.read_csv(
        cls.data_path / "test_model_reaction_weights.csv",
        index_col=0,
        header=None,
    ).squeeze("columns")
    cls.epsilon = 1
    cls.threshold = 1e-2


class TestAddSingleConstraints(unittest.TestCase):
    model = None
    data_path = None
    rxn_weights = None
    epsilon = None
    threshold = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_imat_neg_weight(self):
        test_model = self.model.copy()
        imatpy.imat._imat_neg_weight_(
            model=test_model, rxn="r_C_H", threshold=self.threshold
        )  # Add constraint
        # Check that the binary variable was added
        self.assertTrue("y_pos_r_C_H" in test_model.solver.variables)
        # Check the type of the added variable
        self.assertEqual(
            test_model.solver.variables["y_pos_r_C_H"].type, "binary"
        )
        # Check that the forward constraint was added
        self.assertTrue(
            "forward_constraint_r_C_H" in test_model.solver.constraints
        )
        # Check that the reverse constraint was added
        self.assertTrue(
            "reverse_constraint_r_C_H" in test_model.solver.constraints
        )
        # TODO: Read expression into sympy to check that it behaves
        # equivalently, instead of checking the bounds
        # For above, see
        # https://docs.sympy.org/latest/modules/solvers/inequalities.html

    def test_imat_pos_weight(self):
        test_model = self.model.copy()
        imatpy.imat._imat_pos_weight_(
            model=test_model, rxn="r_C_H", epsilon=self.epsilon
        )
        # CHeck that the positive binary variable was added
        self.assertTrue("y_pos_r_C_H" in test_model.solver.variables)
        # Check that the negative binary variable was added
        self.assertTrue("y_neg_r_C_H" in test_model.solver.variables)
        # Check that the forward constraint was added
        self.assertTrue(
            "forward_constraint_r_C_H" in test_model.solver.constraints
        )
        # Check that the reverse constraint was added
        self.assertTrue(
            "reverse_constraint_r_C_H" in test_model.solver.constraints
        )
        # TODO: Add checks for the behavior of the constraints


class TestAddImatConstraints(unittest.TestCase):
    model = None
    data_path = None
    rxn_weights = None
    epsilon = None
    threshold = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_add_imat_constraints_inplace(self):
        test_model = self.model.copy()
        copy_model = self.model.copy()
        imatpy.imat.add_imat_constraints_(
            model=test_model,
            rxn_weights=self.rxn_weights,
            epsilon=self.epsilon,
            threshold=self.threshold,
        )
        # Check that the model was modified in place
        self.assertFalse(model_eq(test_model, copy_model))
        # Now, update the copy model according to the known weights
        imatpy.imat._imat_neg_weight_(
            model=copy_model, rxn="r_C_H", threshold=self.threshold
        )
        imatpy.imat._imat_neg_weight_(
            model=copy_model, rxn="r_C_E_F", threshold=self.threshold
        )
        imatpy.imat._imat_pos_weight_(
            model=copy_model, rxn="r_A_B_D_E", epsilon=self.epsilon
        )
        imatpy.imat._imat_pos_weight_(
            model=copy_model, rxn="r_D_G", epsilon=self.epsilon
        )
        self.assertTrue(model_eq(test_model, copy_model))

    def test_add_imat_constraints_not_inplace(self):
        test_model = self.model.copy()
        copy_model = test_model.copy()
        updated_model = imatpy.imat.add_imat_constraints(
            model=test_model,
            rxn_weights=self.rxn_weights,
            epsilon=self.epsilon,
            threshold=self.threshold,
        )
        # Check that the model was not modified in place
        self.assertTrue(model_eq(test_model, copy_model))
        # Now, update the copy model according to the known weights
        imatpy.imat._imat_neg_weight_(
            model=copy_model, rxn="r_C_H", threshold=self.threshold
        )
        imatpy.imat._imat_neg_weight_(
            model=copy_model, rxn="r_C_E_F", threshold=self.threshold
        )
        imatpy.imat._imat_pos_weight_(
            model=copy_model, rxn="r_A_B_D_E", epsilon=self.epsilon
        )
        imatpy.imat._imat_pos_weight_(
            model=copy_model, rxn="r_D_G", epsilon=self.epsilon
        )
        self.assertTrue(model_eq(updated_model, copy_model))


class TestAddImatObjective(unittest.TestCase):
    model = None
    data_path = None
    rxn_weights = None
    epsilon = None
    threshold = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_add_objective_inplace(self):
        test_model = self.model.copy()
        # add imat constraints
        imatpy.imat.add_imat_constraints_(
            model=test_model,
            rxn_weights=self.rxn_weights,
            epsilon=self.epsilon,
            threshold=self.threshold,
        )
        copy_model = test_model.copy()

        # add imat objective
        imatpy.imat.add_imat_objective_(
            model=test_model, rxn_weights=self.rxn_weights
        )
        # Check that the model was modified in place
        self.assertFalse(model_eq(test_model, copy_model))
        # Check that the objective was changed
        self.assertFalse(
            _check_expression_eq(
                test_model.objective.expression,
                copy_model.objective.expression,
            )
        )
        # TODO: Check that the objective is actually correct

    def test_add_objective_not_inplace(self):
        test_model = self.model.copy()
        # add imat constraints
        imatpy.imat.add_imat_constraints_(
            model=test_model,
            rxn_weights=self.rxn_weights,
            epsilon=self.epsilon,
            threshold=self.threshold,
        )
        copy_model = copy.deepcopy(test_model)
        imatpy.imat._enforce_binary(model=copy_model)
        # Check that the copy creates an identical model
        self.assertTrue(model_eq(test_model, copy_model, verbose=True))
        # add imat objective
        updated_model = imatpy.imat.add_imat_objective(
            model=test_model, rxn_weights=self.rxn_weights
        )
        # Test that model wasn't modified in place
        self.assertTrue(model_eq(test_model, copy_model))
        # Test that updated model has different objective
        self.assertNotEqual(updated_model.objective, copy_model.objective)
        # Test that updated model has different objective expression
        self.assertFalse(
            _check_expression_eq(
                updated_model.objective.expression,
                copy_model.objective.expression,
            )
        )
        # TODO: Check that the objective is actually correct


class TestImat(unittest.TestCase):
    model = None
    data_path = None
    rxn_weights = None
    epsilon = None
    threshold = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def imat_helper(self, test_model):
        copy_model = test_model.copy()
        # Perform iMAT
        imat_res = imatpy.imat.imat(
            model=test_model,
            rxn_weights=self.rxn_weights,
            epsilon=self.epsilon,
            threshold=self.threshold,
        )
        # Check that the model was not modified
        self.assertTrue(model_eq(test_model, copy_model))
        # Get the binary solution
        bin_sol_active = imatpy.imat.flux_to_binary(
            fluxes=imat_res.fluxes,
            which_reactions="active",
            epsilon=self.epsilon,
            threshold=self.threshold,
        )
        bin_sol_inactive = imatpy.imat.flux_to_binary(
            fluxes=imat_res.fluxes,
            which_reactions="inactive",
            epsilon=self.epsilon,
            threshold=self.threshold,
        )
        # Check that the binary solution is correct
        # Check that r_A_B_D_E is active
        self.assertTrue(bin_sol_active["r_A_B_D_E"])
        # Check that r_D_G is active
        self.assertTrue(bin_sol_active["r_D_G"])
        # Check that r_C_H is inactive
        self.assertTrue(bin_sol_inactive["r_C_H"])

    @skipIf(
        importlib.util.find_spec("cplex") is None, "cplex is not installed"
    )
    def test_imat_cplex(self):
        test_model = self.model.copy()
        test_model.solver = "cplex"
        self.imat_helper(test_model)

    @skipIf(
        importlib.util.find_spec("gurobipy") is None,
        "gurobipy is not installed",
    )
    def test_imat_gurobi(self):
        test_model = self.model.copy()
        test_model.solver = "gurobi"
        self.imat_helper(test_model)

    @skipIf(
        importlib.util.find_spec("swiglpk") is None, "glpk is not installed"
    )
    def test_imat_glpk(self):
        test_model = self.model.copy()
        test_model.solver = "glpk"
        self.imat_helper(test_model)


class TestImatExtensionFunctions(unittest.TestCase):
    model = None
    data_path = None
    rxn_weights = None
    epsilon = None
    threshold = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_compute_imat_objective(self):
        test_model = self.model.copy()
        # Solve imat problem
        imat_sol = imatpy.imat.imat(
            model=test_model,
            rxn_weights=self.rxn_weights,
            epsilon=self.epsilon,
            threshold=self.threshold,
        )
        imat_objective = imat_sol.objective_value
        # Compute objective using extension function
        computed_objective = imatpy.imat.compute_imat_objective(
            fluxes=imat_sol.fluxes,
            rxn_weights=self.rxn_weights,
            epsilon=self.epsilon,
            threshold=self.threshold,
        )
        # Check that the objectives are equal
        self.assertEqual(imat_objective, computed_objective)


if __name__ == "__main__":
    unittest.main()
