import matplotlib.pyplot as plt
import cdd

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import List, Literal, Union, Optional

import math
from pydrake.math import le, ge, eq
import pydrake.symbolic as sym
import pydrake.geometry.optimization as opt

from pydrake.geometry.optimization import GraphOfConvexSets
from pydrake.solvers import MathematicalProgram, Solve, MathematicalProgramResult

# TODO: Replace with VPolytope
class Polyhedron(opt.HPolyhedron):
    def get_vertices(self) -> npt.NDArray[np.float64]:  # [N, 2]
        A = self.A()
        b = self.b().reshape((-1, 1))
        dim = A.shape[0]
        cdd_matrix = cdd.Matrix(np.hstack((b, -A)))
        cdd_matrix.rep_type = cdd.RepType.INEQUALITY
        cdd_poly = cdd.Polyhedron(cdd_matrix)
        generators = np.array(cdd_poly.get_generators())
        # cdd specific, see https://pycddlib.readthedocs.io/en/latest/polyhedron.html
        vertices = generators[:, 1 : 1 + dim]
        return vertices

    @classmethod
    def from_vertices(cls, vertices: npt.NDArray[np.float64]) -> "Polyhedron":
        ones = np.ones((vertices.shape[0], 1))
        cdd_matrix = cdd.Matrix(np.hstack((ones, vertices)))
        cdd_matrix.rep_type = cdd.RepType.GENERATOR
        cdd_poly = cdd.Polyhedron(cdd_matrix)
        inequalities = np.array(cdd_poly.get_inequalities())
        b = inequalities[:, 0:1]
        A = -inequalities[:, 1:]

        return cls(A, b)


class PolyhedronFormulator:
    def __init__(
        self,
        constraints: List[sym.Expression],
    ):
        flattened_constraints = [c.flatten() for c in constraints]
        self.constraints = np.concatenate(flattened_constraints)

    def formulate_polyhedron(
        self,
        variables: npt.NDArray[sym.Variable],
        make_bounded: bool = True,
        remove_constraints_not_in_vars: bool = False,
        BOUND: float = 999,  # TODO tune?
    ) -> opt.HPolyhedron:
        if make_bounded:
            ub = np.ones(variables.shape) * BOUND
            upper_limits = le(variables, ub)
            lower_limits = le(-ub, variables)
            limits = np.concatenate((upper_limits, lower_limits))
            self.constraints = np.append(self.constraints, limits)

        expressions = []
        for formula in self.constraints:
            kind = formula.get_kind()
            lhs, rhs = formula.Unapply()[1]
            if kind == sym.FormulaKind.Eq:
                # Eq constraint ax = b is
                # implemented as ax <= b, -ax <= -b
                expressions.append(lhs - rhs)
                expressions.append(rhs - lhs)
            elif kind == sym.FormulaKind.Geq:
                # lhs >= rhs
                # ==> rhs - lhs <= 0
                expressions.append(rhs - lhs)
            elif kind == sym.FormulaKind.Leq:
                # lhs <= rhs
                # ==> lhs - rhs <= 0
                expressions.append(lhs - rhs)

        if remove_constraints_not_in_vars:

            def check_all_vars_are_relevant(exp):
                return all(
                    [
                        exp_var in sym.Variables(variables) # Need to convert this to Variables to check contents
                        for exp_var in exp.GetVariables()
                    ]
                )

            expressions = list(filter(check_all_vars_are_relevant, expressions))

        # We now have expr <= 0 for all expressions
        # ==> we get Ax - b <= 0
        A, b_neg = sym.DecomposeAffineExpressions(expressions, variables)

        # Polyhedrons are of the form: Ax <= b
        b = -b_neg
        convex_set_as_polyhedron = opt.HPolyhedron(A, b)
        return convex_set_as_polyhedron
