# Standard Library Imports
import pathlib
import unittest

import numpy as np
import pandas as pd

# External Imports
from cobra.core.configuration import Configuration

import imatpy.imat
from imatpy.imat import compute_imat_objective
from imatpy.model_creation import (
    _active_bounds,
    _inactive_bounds,
    _milp_eval,
    _parse_method,
    fva_model,
    imat_constraint_model,
    milp_model,
    simple_bounds_model,
    subset_model,
)

# Local Imports
from imatpy.model_utils import _check_objective_eq, model_eq, read_model


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
    cls.objective_tolerance = 5e-2


class TestGenerateModel(unittest.TestCase):
    pass


class TestModelCreation(unittest.TestCase):
    data_path = None
    model = None
    rxn_weights = None
    epsilon = None
    threshold = None
    objective_tolerance = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_imat_model(self):
        test_model = self.model.copy()
        imat_sol = imatpy.imat.imat(
            model=test_model,
            rxn_weights=self.rxn_weights,
            epsilon=self.epsilon,
            threshold=self.threshold,
        )
        imat_model = imat_constraint_model(
            test_model,
            self.rxn_weights,
            self.epsilon,
            self.threshold,
            self.objective_tolerance,
        )
        # Check that the binary variables were added
        self.assertTrue("y_pos_r_C_H" in imat_model.solver.variables)
        self.assertTrue("y_pos_r_C_E_F" in imat_model.solver.variables)
        self.assertTrue("y_pos_r_D_G" in imat_model.solver.variables)
        self.assertTrue("y_pos_r_A_B_D_E" in imat_model.solver.variables)
        self.assertTrue("y_neg_r_A_B_D_E" in imat_model.solver.variables)
        self.assertTrue("y_neg_r_D_G" in imat_model.solver.variables)
        # Check that the imat constraint was added
        self.assertTrue("imat_obj_constraint" in imat_model.solver.constraints)
        # Check that the objective is the same as before
        self.assertTrue(
            _check_objective_eq(test_model.objective, imat_model.objective)
        )
        # Check that the test_model hasn't been modified
        self.assertTrue(model_eq(test_model, self.model))
        # Check that optimization respected imat constraint
        solution = imat_model.optimize()
        imat_objective = compute_imat_objective(
            fluxes=solution.fluxes,
            rxn_weights=self.rxn_weights,
            epsilon=self.epsilon,
            threshold=self.threshold,
        )
        self.assertTrue(
            np.abs(imat_objective - imat_sol.objective_value)
            < self.objective_tolerance * imat_sol.objective_value
        )

    def test_simple_bounds_model(self):
        test_model = self.model.copy()
        updated_model = simple_bounds_model(
            test_model, self.rxn_weights, self.epsilon, self.threshold
        )
        # Check that the objective is the same as before
        self.assertTrue(
            _check_objective_eq(test_model.objective, updated_model.objective)
        )
        # Check that the test_model hasn't been modified
        self.assertTrue(model_eq(test_model, self.model))
        # Check that the binary vaiables weren't added
        self.assertTrue("y_pos_r_C_H" not in updated_model.solver.variables)
        self.assertTrue("y_pos_r_C_E_F" not in updated_model.solver.variables)
        self.assertTrue("y_pos_r_D_G" not in updated_model.solver.variables)
        self.assertTrue(
            "y_pos_r_A_B_D_E" not in updated_model.solver.variables
        )
        self.assertTrue(
            "y_neg_r_A_B_D_E" not in updated_model.solver.variables
        )
        self.assertTrue("y_neg_r_D_G" not in updated_model.solver.variables)
        # Check that the model has been changed
        self.assertFalse(model_eq(test_model, updated_model))
        # Check that the model can be optimized
        _ = updated_model.optimize()

    def test_subset_model(self):
        test_model = self.model.copy()
        updated_model = subset_model(
            test_model, self.rxn_weights, self.epsilon, self.threshold
        )
        # Check that the objective is the same as before
        self.assertTrue(
            _check_objective_eq(test_model.objective, updated_model.objective)
        )
        # Check that the test_model hasn't been modified
        self.assertTrue(model_eq(test_model, self.model))
        # Check that the binary variables weren't added
        self.assertTrue("y_pos_r_C_H" not in updated_model.solver.variables)
        self.assertTrue("y_pos_r_C_E_F" not in updated_model.solver.variables)
        self.assertTrue("y_pos_r_D_G" not in updated_model.solver.variables)
        self.assertTrue(
            "y_pos_r_A_B_D_E" not in updated_model.solver.variables
        )
        self.assertTrue(
            "y_neg_r_A_B_D_E" not in updated_model.solver.variables
        )
        self.assertTrue("y_neg_r_D_G" not in updated_model.solver.variables)
        # Check that the model has been changed
        self.assertFalse(model_eq(test_model, updated_model))
        # Check that the model can be optimized
        _ = updated_model.optimize()

    def test_fva_model(self):
        test_model = self.model.copy()
        updated_model = fva_model(
            test_model,
            self.rxn_weights,
            self.epsilon,
            self.threshold,
            self.objective_tolerance,
        )
        # Check that the objective is the same as before
        self.assertTrue(
            _check_objective_eq(test_model.objective, updated_model.objective)
        )
        # Check that the test_model hasn't been modified
        self.assertTrue(model_eq(test_model, self.model))
        # Check that the binary variables weren't added
        self.assertTrue("y_pos_r_C_H" not in updated_model.solver.variables)
        self.assertTrue("y_pos_r_C_E_F" not in updated_model.solver.variables)
        self.assertTrue("y_pos_r_D_G" not in updated_model.solver.variables)
        self.assertTrue(
            "y_pos_r_A_B_D_E" not in updated_model.solver.variables
        )
        self.assertTrue(
            "y_neg_r_A_B_D_E" not in updated_model.solver.variables
        )
        self.assertTrue("y_neg_r_D_G" not in updated_model.solver.variables)
        # Check that the model has been changed
        self.assertFalse(model_eq(test_model, updated_model))
        # Check that the model can be optimized
        _ = updated_model.optimize()

    def test_milp_model(self):
        test_model = self.model.copy()
        updated_model = milp_model(
            test_model, self.rxn_weights, self.epsilon, self.threshold
        )
        # Check that the objective is the same as before
        self.assertTrue(
            _check_objective_eq(test_model.objective, updated_model.objective)
        )
        # Check that the test_model hasn't been modified
        self.assertTrue(model_eq(test_model, self.model))
        # Check that the binary variables weren't added
        self.assertTrue("y_pos_r_C_H" not in updated_model.solver.variables)
        self.assertTrue("y_pos_r_C_E_F" not in updated_model.solver.variables)
        self.assertTrue("y_pos_r_D_G" not in updated_model.solver.variables)
        self.assertTrue(
            "y_pos_r_A_B_D_E" not in updated_model.solver.variables
        )
        self.assertTrue(
            "y_neg_r_A_B_D_E" not in updated_model.solver.variables
        )
        self.assertTrue("y_neg_r_D_G" not in updated_model.solver.variables)
        # Check that the model has been changed
        self.assertFalse(model_eq(test_model, updated_model))
        # Check that the model can be optimized
        _ = updated_model.optimize()


class TestHelperFunctions(unittest.TestCase):
    def test_parse_method(self):
        self.assertEqual(_parse_method("simple"), "simple_bounds")
        self.assertEqual(_parse_method("imat_restrictions"), "imat_constraint")
        self.assertEqual(_parse_method("subset-ko"), "subset")
        self.assertEqual(_parse_method("flux variability analysis"), "fva")
        self.assertEqual(_parse_method("m"), "milp")
        with self.assertRaises(ValueError):
            _parse_method("invalid")

    def test_inactive_bounds(self):
        self.assertEqual(_inactive_bounds(-1, 1, 0.5), (-0.5, 0.5))
        self.assertEqual(_inactive_bounds(-0.25, 1, 0.5), (-0.25, 0.5))
        self.assertEqual(_inactive_bounds(0, 1, 0.5), (0, 0.5))
        self.assertEqual(_inactive_bounds(0.25, 1, 0.5), (0.25, 0.5))
        with self.assertRaises(ValueError):
            _inactive_bounds(1, 2, 0.5)

    def test_active_bounds(self):
        self.assertEqual(_active_bounds(-1, 1, 0.5, True), (0.5, 1))
        self.assertEqual(_active_bounds(-1, 1, 0.5, False), (-1, -0.5))
        with self.assertRaises(ValueError):
            _active_bounds(0, 0.25, 0.5, True)
        with self.assertRaises(ValueError):
            _active_bounds(-0.25, 0.25, 0.5, False)
        with self.assertRaises(ValueError):
            _active_bounds(2, 1, 0.5, True)

    def test_milp_eval(self):
        test_df = pd.DataFrame(
            {
                "inactive": [np.nan, 1, 3, 3, 2, 1],
                "forward": [1, 2, 3, 2, 3, 1],
                "reverse": [2, 3, 2, 2, 2, 2],
            },
            index=["A", "B", "C", "D", "E", "F"],
        )
        actual_results = test_df.apply(_milp_eval, axis=1).replace(
            {np.nan: -1}
        )
        expected_results = pd.Series(
            [-1, 2, -1, 0, 1, 2], index=["A", "B", "C", "D", "E", "F"]
        )
        self.assertTrue(np.all(actual_results == expected_results))


if __name__ == "__main__":
    unittest.main()
