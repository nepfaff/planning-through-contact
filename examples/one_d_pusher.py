from geometry.contact import CollisionGeometry
from geometry.bezier import BezierVariable
from geometry.polyhedron import PolyhedronFormulator

from pydrake.math import le, ge, eq
from pydrake.geometry.optimization import GraphOfConvexSets
import pydrake.symbolic as sym
from pydrake.solvers import LinearConstraint, Binding, L1NormCost, Cost

import numpy as np

import itertools


def plan_for_one_d_pusher():
    # Bezier curve params
    dim = 2
    order = 2

    # Physical params
    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    l = 2
    friction_coeff = 0.5

    finger = CollisionGeometry(dim=dim, order=order, name="finger")
    box = CollisionGeometry(dim=dim, order=order, name="box")

    x_f = finger.pos.x
    v_f = finger.vel.x

    x_b = box.pos.x
    v_b = box.vel.x

    lam_n = BezierVariable(dim=dim, order=order, name="lam_n").x
    lam_f = BezierVariable(dim=dim, order=order, name="lam_f").x

    sdf = x_b - x_f - l

    # "No contact" vertex
    no_contact = []
    no_contact.append(ge(sdf, 0))
    no_contact.append(eq(lam_n, 0))
    no_contact.append(eq(v_b, 0))
    no_contact.append(eq(lam_f, -lam_n))

    # "Touching" vertex
    touching = []
    touching.append(eq(sdf, 0))
    touching.append(ge(lam_n, 0))
    touching.append(eq(v_b, 0))
    touching.append(eq(lam_f, -lam_n))

    # "Pushing right" vertex
    pushing_right = []
    pushing_right.append(eq(sdf, 0))
    pushing_right.append(ge(lam_n, 0))
    pushing_right.append(ge(v_b, 0))
    pushing_right.append(eq(lam_f, -friction_coeff * mg))
    pushing_right.append(eq(lam_f, -lam_n))

    # Create the convex sets
    all_variables = np.concatenate([var.flatten() for var in [x_f, x_b, lam_n, lam_f]])
    contact_modes = [no_contact, touching, pushing_right]
    mode_names = ["no_contact", "touching", "pushing_right"]
    polyhedrons = [
        PolyhedronFormulator(mode).formulate_polyhedron(
            variables=all_variables, make_bounded=True
        )
        for mode in contact_modes
    ]

    # Add Vertices
    gcs = GraphOfConvexSets()
    for name, poly in zip(mode_names, polyhedrons):
        gcs.AddVertex(poly, name)

    # Add edges between all vertices
    for u, v in itertools.permutations(gcs.Vertices(), 2):
        gcs.AddEdge(u, v, name=f"({u.name()},{v.name()})")

    # Create position continuity constraints
    pos_vars = np.vstack((x_f, x_b))
    first_pos_vars = pos_vars[:, 0]
    last_pos_vars = pos_vars[:, -1]
    A_first = sym.DecomposeLinearExpressions(first_pos_vars, all_variables)
    A_last = sym.DecomposeLinearExpressions(last_pos_vars, all_variables)
    for e in gcs.Edges():
        xu, xv = e.xu(), e.xv()
        constraints = eq(A_last.dot(xu), A_first.dot(xv))
        for c in constraints:
            e.AddConstraint(c)

    # Create L1 norm cost
    diffs = pos_vars[:, 1:] - pos_vars[:, :-1]
    A = sym.DecomposeLinearExpressions(diffs.flatten(), all_variables)
    b = np.zeros((A.shape[0], 1))
    l1_norm_cost = L1NormCost(A, b)
    for v in gcs.Vertices():
        cost = Binding[Cost](l1_norm_cost, v.x())
        v.AddCost(cost)

    breakpoint()
    return
