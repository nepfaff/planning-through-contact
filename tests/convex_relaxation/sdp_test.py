from typing import NamedTuple, Optional

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
import pytest
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.sdp import (
    create_sdp_relaxation,
    eliminate_equality_constraints,
    find_solution,
    get_nullspace_matrix,
)


class LinearSystem(NamedTuple):
    A: npt.NDArray[np.float64]
    b: npt.NDArray[np.float64]
    x_sol: Optional[npt.NDArray[np.float64]] = None


@pytest.fixture
def fully_determined_linear_system() -> LinearSystem:
    A = np.array([[1, 0, 1], [0, -3, 1], [2, 1, 3]])
    b = np.array([6, 7, 15]).reshape(-1, 1)
    x = np.array([2, -1, 4]).reshape(-1, 1)
    return LinearSystem(A, b, x)


@pytest.fixture
def underdetermined_linear_system() -> LinearSystem:
    A = np.array([[1, 1, 1], [1, 1, 2]])
    b = np.array([1, 3]).reshape(-1, 1)
    return LinearSystem(A, b)


def test_find_solution_fully_determined(
    fully_determined_linear_system: LinearSystem,
) -> None:
    A, b, x_sol = fully_determined_linear_system
    assert x_sol is not None
    x_hat = find_solution(A, b)
    assert np.allclose(x_sol, x_hat)


def test_find_solution_underdetermined(
    underdetermined_linear_system: LinearSystem,
) -> None:
    A, b, _ = underdetermined_linear_system
    x_hat = find_solution(A, b)
    assert np.allclose(A.dot(x_hat), b)


def test_get_nullspace_matrix(underdetermined_linear_system: LinearSystem) -> None:
    A, b, _ = underdetermined_linear_system
    F = get_nullspace_matrix(A)

    assert np.allclose(A.dot(F), 0)


@pytest.fixture
def so_2_true_sol() -> npt.NDArray[np.float64]:
    return np.array([-0.70710678, -0.70710678])


@pytest.fixture
def so_2_prog() -> MathematicalProgram:
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x")[0]
    y = prog.NewContinuousVariables(1, "y")[0]

    prog.AddQuadraticConstraint(x**2 + y**2 - 1, 0, 0)
    prog.AddLinearConstraint(x == y)

    prog.AddQuadraticCost(
        x * y + x + y
    )  # add a cost with a linear term to make the relaxation tight

    return prog


# def test_equality_elimination_with_initial_guess(
#     so_2_prog: MathematicalProgram, so_2_true_sol: npt.NDArray[np.float64]
# ) -> None:
#     prog = so_2_prog
#     true_sol = so_2_true_sol
#
#     prog.SetInitialGuess(prog.decision_variables(), np.array([0.7, 0.7]))
#
#     result = Solve(prog)
#     assert result.is_success()
#
#     sol = result.GetSolution(prog.decision_variables())
#
#     smaller_prog, retrieve_x, F, x_hat = eliminate_equality_constraints(prog)
#     z = smaller_prog.decision_variables()[0]
#     smaller_prog.SetInitialGuess(z, 0.7)
#     smaller_result = Solve(smaller_prog)
#     assert smaller_result.is_success()
#
#     sol = retrieve_x(smaller_result.GetSolution(z))
#     breakpoint()
#     assert np.all(sol == true_sol)
#
#
# def test_equality_elimination_with_sdp_relaxation(
#     so_2_test_data: Tuple[MathematicalProgram, npt.NDArray[np.float64]]
# ):
#     prog, true_sol = so_2_test_data
#
#     # TODO: Fix problem where sdp relaxation doesn't pick up quadratic constraints
#     prog = MathematicalProgram()
#     x = prog.NewContinuousVariables(1, "x")[0]
#     y = prog.NewContinuousVariables(1, "y")[0]
#
#     prog.AddQuadraticConstraint(x**2 + y**2 - 1, 0, 0)
#     prog.AddLinearConstraint(x == y)
#
#     # Solve with SDP relaxation
#     relaxed_prog, X, _ = create_sdp_relaxation(prog)
#     x = X[1:, 0]
#
#     result = Solve(relaxed_prog)
#     assert result.is_success()
#
#     sol_relaxation = result.GetSolution(x)
#
#     # Eliminate linear equality constraints, then solve relaxation
#     smaller_prog, retrieve_x = eliminate_equality_constraints(prog)
#     relaxed_prog, Z, _ = create_sdp_relaxation(smaller_prog)
#     relaxed_result = Solve(relaxed_prog)
#     assert relaxed_result.is_success()
#
#     z_sol = relaxed_result.GetSolution(Z[1:, 0])
#     eliminated_sol = retrieve_x(z_sol)
#     assert np.allclose(sol_relaxation, eliminated_sol, atol=1e-3)
#
#     assert np.allclose(eliminated_sol, TRUE_SOL, atol=1e-3)


if __name__ == "__main__":
    # test_equality_elimination_with_initial_guess(so_2_test_data())
    # test_equality_elimination_with_sdp_relaxation()

    test_find_solution_fully_determined(fully_determined_linear_system())
