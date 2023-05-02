import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from functools import reduce

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import ConvexSet
from pydrake.math import eq, ge, le

from planning_through_contact.geometry.contact_mode import (
    ContactMode,
    ContactModeConfig,
    ContactModeType,
    PositionModeType,
    ForceVariablePair,
    calc_intersection_of_contact_modes,
)
from planning_through_contact.geometry.polyhedron import PolyhedronFormulator
from planning_through_contact.geometry.rigid_body import RigidBody


@dataclass
class ContactPair:
    body_a: RigidBody
    body_b: RigidBody
    friction_coeff: float
    position_mode: PositionModeType
    position_type_force_variable_pairs: Dict[PositionModeType, ForceVariablePair]
    force_curve_order: int = 1  # TODO remove?
    allowable_contact_mode_types: List[ContactModeType] = field(
        default_factory=lambda: []
    )
    transition_eps: float = 0
    center_contact_buffer: float = 0

    def __post_init__(self):
        self.sdf = self._create_signed_distance_func(
            self.body_a, self.body_b, self.position_mode
        )
        self.n_hat = self._create_normal_vec(
            self.body_a, self.body_b, self.position_mode
        )  # Contact normal from body A to body B
        self.d_hat = self._create_tangential_vec(self.n_hat)

        # Forces between body_a and body_b
        self.lam_n = self.position_type_force_variable_pairs[self.position_mode].lam_n
        self.lam_f = self.position_type_force_variable_pairs[self.position_mode].lam_f

        self.additional_constraints = []
        self.contact_modes_formulated = False

    def _create_position_mode_constraints(
        self, body_a, body_b, position_mode: PositionModeType
    ) -> npt.NDArray[sym.Formula]:
        if body_a.geometry == "point" and body_b.geometry == "point":
            raise ValueError("Point with point contact not allowed")
        elif body_a.geometry == "box" and body_b.geometry == "box":
            if (
                position_mode == PositionModeType.LEFT
                or position_mode == PositionModeType.RIGHT
            ):
                y_constraint_top = ge(
                    body_a.pos_y + body_a.height, body_b.pos_y - body_b.height
                )
                y_constraint_bottom = le(
                    body_a.pos_y - body_a.height, body_b.pos_y + body_b.height
                )
                return np.array([y_constraint_top, y_constraint_bottom])
            elif (
                position_mode == PositionModeType.TOP
                or position_mode == PositionModeType.BOTTOM
            ):
                x_constraint_left = ge(
                    body_a.pos_x + body_a.width, body_b.pos_x - body_b.width
                )
                x_constraint_right = le(
                    body_a.pos_x - body_a.width, body_b.pos_x + body_b.width
                )
                return np.array([x_constraint_left, x_constraint_right])
            elif position_mode == PositionModeType.FRONT:
                # Above floor
                z_constraint = ge(
                    body_a.pos_z - body_a.depth, body_b.pos_z + body_b.depth
                )

                # Within floor
                x_constraint_left = ge(
                    body_a.pos_x + body_a.width, body_b.pos_x - body_b.width
                )
                x_constraint_right = le(
                    body_a.pos_x - body_a.width, body_b.pos_x + body_b.width
                )
                y_constraint_top = ge(
                    body_a.pos_y + body_a.height, body_b.pos_y - body_b.height
                )
                y_constraint_bottom = le(
                    body_a.pos_y - body_a.height, body_b.pos_y + body_b.height
                )
                return np.array(
                    [
                        z_constraint,
                        x_constraint_left,
                        x_constraint_right,
                        y_constraint_top,
                        y_constraint_bottom,
                    ]
                )
        elif body_a.geometry == "point" and body_b.geometry == "box":
            box = body_b
            point = body_a

            if (
                position_mode == PositionModeType.LEFT
            ):
                y_constraint_top = le(sphere.pos_y - sphere.radius, box.pos_y + box.height) # top of the sphere is below top of the box
                y_constraint_bottom = ge(sphere.pos_y + sphere.radius, box.pos_y - box.height) # bottom of the sphere is above the bottom of the box
                x_constraint_left = le(sphere.pos_x, box.pos_x - box.width - self.transition_eps) # right side of the sphere is to the left of the box with transition_eps buffer
                return np.array([y_constraint_top, y_constraint_bottom, x_constraint_left])
            if (
                position_mode == PositionModeType.RIGHT
            ):
                y_constraint_top = le(sphere.pos_y - sphere.radius, box.pos_y + box.height)
                y_constraint_bottom = ge(sphere.pos_y + sphere.radius, box.pos_y - box.height)
                x_constraint_right = ge(sphere.pos_x, box.pos_x + box.width + self.transition_eps)
                return np.array([y_constraint_top, y_constraint_bottom, x_constraint_right])
            elif position_mode == PositionModeType.TOP:
                x_constraint_left = ge(sphere.pos_x + sphere.radius, box.pos_x - box.width)
                x_constraint_right = le(sphere.pos_x - sphere.radius, box.pos_x + box.width)
                y_constraint_top = ge(sphere.pos_y, box.pos_y + box.height + self.transition_eps)
                return np.array([x_constraint_left, x_constraint_right, y_constraint_top])
            elif position_mode == PositionModeType.BOTTOM:
                x_constraint_left = ge(sphere.pos_x + sphere.radius, box.pos_x - box.width)
                x_constraint_right = le(sphere.pos_x - sphere.radius, box.pos_x + box.width)
                y_constraint_bottom = le(sphere.pos_y, box.pos_y - box.height - self.transition_eps)
                return np.array([x_constraint_left, x_constraint_right, y_constraint_bottom])
            elif position_mode == PositionModeType.LEFT_TRANSITION:
                y_constraint_top = le(sphere.pos_y + sphere.radius, box.pos_y + sphere.radius + self.center_contact_buffer)
                y_constraint_bottom = ge(sphere.pos_y - sphere.radius, box.pos_y - sphere.radius - self.center_contact_buffer)
                x_constraint_left = le(sphere.pos_x + sphere.radius, box.pos_x - box.width)
                x_constraint_right = ge(sphere.pos_x, box.pos_x - box.width - self.transition_eps)
                return np.array([y_constraint_top, y_constraint_bottom, x_constraint_left, x_constraint_right])
            elif position_mode == PositionModeType.RIGHT_TRANSITION:
                y_constraint_top = le(sphere.pos_y + sphere.radius, box.pos_y + sphere.radius + self.center_contact_buffer)
                y_constraint_bottom = ge(sphere.pos_y - sphere.radius, box.pos_y - sphere.radius - self.center_contact_buffer)
                x_constraint_left = le(sphere.pos_x, box.pos_x + box.width + self.transition_eps) # right of the sphere is to the left of the right position mode
                x_constraint_right = ge(sphere.pos_x - sphere.radius, box.pos_x + box.width) # left of the sphere is to the right of the box
                return np.array([y_constraint_top, y_constraint_bottom, x_constraint_left, x_constraint_right])
            elif position_mode == PositionModeType.TOP_TRANSITION:
                x_constraint_left = le(sphere.pos_x + sphere.radius, box.pos_x + sphere.radius + self.center_contact_buffer)
                x_constraint_right = ge(sphere.pos_x - sphere.radius, box.pos_x - sphere.radius - self.center_contact_buffer)
                y_constraint_top = ge(sphere.pos_y - sphere.radius, box.pos_y + box.height) # bottom of the sphere is above the box 
                y_constraint_bottom = le(sphere.pos_y, box.pos_y + box.height + self.transition_eps) # top of the sphere is below the top position mode
                return np.array([y_constraint_top, y_constraint_bottom, x_constraint_left, x_constraint_right])
            elif position_mode == PositionModeType.BOTTOM_TRANSITION:
                x_constraint_left = le(sphere.pos_x + sphere.radius, box.pos_x + sphere.radius + self.center_contact_buffer)
                x_constraint_right = ge(sphere.pos_x - sphere.radius, box.pos_x - sphere.radius - self.center_contact_buffer)
                y_constraint_top = ge(sphere.pos_y, box.pos_y - box.height - self.transition_eps) # bottom of the sphere is above the bottom position mode
                y_constraint_bottom = le(sphere.pos_y + sphere.radius, box.pos_y - box.height) # top of the sphere is below the bottom of the box
                return np.array([y_constraint_top, y_constraint_bottom, x_constraint_left, x_constraint_right])
            elif position_mode == PositionModeType.TOP_LEFT:
                x_constraint = le(sphere.pos_x + sphere.radius, box.pos_x - box.width)
                y_constraint = ge(sphere.pos_y - sphere.radius, box.pos_y + box.height)
                return np.array([x_constraint, y_constraint])
            elif position_mode == PositionModeType.TOP_RIGHT:
                x_constraint = ge(sphere.pos_x - sphere.radius, box.pos_x + box.width)
                y_constraint = ge(sphere.pos_y - sphere.radius, box.pos_y + box.height)
                return np.array([x_constraint, y_constraint])
            elif position_mode == PositionModeType.BOTTOM_LEFT:
                x_constraint = le(sphere.pos_x + sphere.radius, box.pos_x - box.width)
                y_constraint = le(sphere.pos_y + sphere.radius, box.pos_y - box.height)
                return np.array([x_constraint, y_constraint])
            elif position_mode == PositionModeType.BOTTOM_RIGHT:
                x_constraint = ge(sphere.pos_x - sphere.radius, box.pos_x + box.width)
                y_constraint = le(sphere.pos_y + sphere.radius, box.pos_y - box.height)
                return np.array([x_constraint, y_constraint])
            elif position_mode == PositionModeType.FRONT:
                raise NotImplementedError()
            else:
                raise NotImplementedError(
                    f"Position mode not implemented: {position_mode}"
                )
        else:
            raise NotImplementedError(f"Position mode not implemented for body_a: {body_a.geometry} and body_b: {body_b.geometry}")

    def _create_signed_distance_func(
        self, body_a, body_b, position_mode: PositionModeType
    ) -> sym.Expression:
        if self.dim == 2:
            if body_a.geometry == "point" and body_b.geometry == "point":
                raise ValueError("Point with point contact not allowed")
            elif body_a.geometry == "box" and body_b.geometry == "box":
                x_offset = body_a.width + body_b.width
                y_offset = body_a.height + body_b.height
            else:
                box = body_a if body_a.geometry == "box" else body_b
                x_offset = box.width
                y_offset = box.height

            if (
                position_mode == PositionModeType.LEFT
            ):  # body_a is on left side of body_b
                dx = body_b.pos_x - body_a.pos_x - x_offset
                dy = 0
            elif position_mode == PositionModeType.RIGHT:
                dx = body_a.pos_x - body_b.pos_x - x_offset
                dy = 0
            elif position_mode == PositionModeType.TOP:  # body_a on top of body_b
                dx = 0
                dy = body_a.pos_y - body_b.pos_y - y_offset
            elif position_mode == PositionModeType.BOTTOM:
                dx = 0
                dy = body_b.pos_y - body_a.pos_y - y_offset
            else:
                raise NotImplementedError(
                    f"Position mode not implemented: {position_mode}"
                )

            return dx + dy  # NOTE convex relaxation
        else:
            if body_a.geometry == "point" and body_b.geometry == "point":
                raise ValueError("Point with point contact not allowed")
            elif body_a.geometry == "box" and body_b.geometry == "box":
                x_offset = body_a.width + body_b.width
                y_offset = body_a.height + body_b.height
                z_offset = body_a.depth + body_b.depth
            elif (body_a.geometry == "sphere" and body_b.geometry == "box") or (body_a.geometry == "box" and body_b.geometry == "sphere"):
                box = body_a if body_a.geometry == "box" else body_b
                sphere = body_a if body_a.geometry == "sphere" else body_b
                x_offset = box.width + sphere.radius
                y_offset = box.height + sphere.radius
                z_offset = box.depth + sphere.radius
            else:
                box = body_a if body_a.geometry == "box" else body_b
                x_offset = box.width
                y_offset = box.height
                z_offset = box.depth

            if (
                position_mode == PositionModeType.LEFT
                or position_mode == PositionModeType.LEFT_TRANSITION
            ):  # body_a is on left side of body_b
                dx = body_b.pos_x - body_a.pos_x - x_offset
                dy = 0
                dz = 0
            elif (
                position_mode == PositionModeType.RIGHT
                or position_mode == PositionModeType.RIGHT_TRANSITION
            ):
                dx = body_a.pos_x - body_b.pos_x - x_offset
                dy = 0
                dz = 0
            elif (
                position_mode == PositionModeType.TOP
                or position_mode == PositionModeType.TOP_TRANSITION
            ): # body_a on top of body_b
                dx = 0
                dy = body_a.pos_y - body_b.pos_y - y_offset
                dz = 0
            elif (
                position_mode == PositionModeType.BOTTOM
                or position_mode == PositionModeType.BOTTOM_TRANSITION
            ):
                dx = 0
                dy = body_b.pos_y - body_a.pos_y - y_offset
                dz = 0
            elif position_mode == PositionModeType.FRONT:
                dx = 0
                dy = 0
                dz = body_a.pos_z - body_b.pos_z - z_offset
            elif position_mode == PositionModeType.TOP_LEFT:
                dx = body_b.pos_x - body_a.pos_x - x_offset
                dy = body_a.pos_y - body_b.pos_y - y_offset
                dz = 0
            elif position_mode == PositionModeType.TOP_RIGHT:
                dx = body_a.pos_x - body_b.pos_x - x_offset
                dy = body_a.pos_y - body_b.pos_y - y_offset
                dz = 0
            elif position_mode == PositionModeType.BOTTOM_LEFT:
                dx = body_b.pos_x - body_a.pos_x - x_offset
                dy = body_b.pos_y - body_a.pos_y - y_offset
                dz = 0
            elif position_mode == PositionModeType.BOTTOM_RIGHT:
                dx = body_a.pos_x - body_b.pos_x - x_offset
                dy = body_b.pos_y - body_a.pos_y - y_offset
                dz = 0
            else:
                raise NotImplementedError(
                    f"Position mode not implemented: {position_mode}"
                )

            return dx + dy + dz  # NOTE convex relaxation

    def _create_normal_vec(
        self, body_a, body_b, position_mode: PositionModeType
    ) -> npt.NDArray[np.float64]:
        if body_a.geometry == "point" and body_b.geometry == "point":
            raise ValueError("Point with point contact not allowed")

        # Normal vector: from body_a to body_b
        if self.dim == 2:
            if position_mode == PositionModeType.LEFT:  # body_a left side of body_b
                n_hat = np.array([[1, 0]]).T
            elif position_mode == PositionModeType.RIGHT:
                n_hat = np.array([[-1, 0]]).T
            elif position_mode == PositionModeType.TOP:
                n_hat = np.array([[0, -1]]).T
            elif position_mode == PositionModeType.BOTTOM:
                n_hat = np.array([[0, 1]]).T
            else:
                raise NotImplementedError(
                    f"2D position mode not implemented: {position_mode}"
                )
        else:
            if (position_mode == PositionModeType.LEFT) or (position_mode == PositionModeType.LEFT_TRANSITION):  # body_a left side of body_b
                n_hat = np.array([[1, 0, 0]]).T
            elif (position_mode == PositionModeType.RIGHT) or (position_mode == PositionModeType.RIGHT_TRANSITION):
                n_hat = np.array([[-1, 0, 0]]).T
            elif (position_mode == PositionModeType.TOP) or (position_mode == PositionModeType.TOP_TRANSITION):
                n_hat = np.array([[0, -1, 0]]).T
            elif (position_mode == PositionModeType.BOTTOM) or (position_mode == PositionModeType.BOTTOM_TRANSITION):
                n_hat = np.array([[0, 1, 0]]).T
            elif position_mode == PositionModeType.FRONT:
                n_hat = np.array([[0, 0, -1]]).T
            elif position_mode == PositionModeType.TOP_LEFT:
                # NOTE: This is incorrect (should not be used)
                n_hat = np.array([[0, 0, 0]]).T
            elif position_mode == PositionModeType.TOP_RIGHT:
                # NOTE: This is incorrect (should not be used)
                n_hat = np.array([[0, 0, 0]]).T
            elif position_mode == PositionModeType.BOTTOM_LEFT:
                # NOTE: This is incorrect (should not be used)
                n_hat = np.array([[0, 0, 0]]).T
            elif position_mode == PositionModeType.BOTTOM_RIGHT:
                # NOTE: This is incorrect (should not be used)
                n_hat = np.array([[0, 0, 0]]).T
            else:
                raise NotImplementedError(
                    f"3D position mode not implemented: {position_mode}"
                )

        return n_hat

    def _create_tangential_vec(self, n_hat: npt.NDArray[np.float64]):
        if self.dim == 2:
            d_hat = np.array([[-n_hat[1, 0]], [n_hat[0, 0]]])
        else:
            if n_hat[0, 0] != 0:
                d_hat = n_hat[0, 0] * np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).T
            elif n_hat[1, 0] != 0:
                d_hat = n_hat[1, 0] * np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T
            elif n_hat[2, 0] != 0:
                d_hat = n_hat[2, 0] * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T
            else:
                # NOTE: This is incorrect and should not be used
                d_hat = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).T
        return d_hat

    @property
    def name(self) -> str:
        return f"({self.body_a.name},{self.body_b.name},{self.position_mode.name})"

    @property
    def dim(self) -> int:
        return self.body_a.dim

    @property
    def contact_jacobian(self) -> npt.NDArray[np.float64]:
        # See http://underactuated.mit.edu/multibody.html#contact
        # v_rel = v_body_b - v_body_a = J (v_body_a, v_body_b)^T
        return np.hstack((-np.eye(self.dim), np.eye(self.dim)))

    @property
    def normal_jacobian(self) -> npt.NDArray[np.float64]:
        # Direction of normal force acting on each body
        return self.n_hat.T.dot(self.contact_jacobian)

    @property
    def tangential_jacobian(self) -> npt.NDArray[np.float64]:
        # Direction of friction force acting on each body
        return self.d_hat.T.dot(self.contact_jacobian)

    @property
    def rel_tangential_sliding_vel(self) -> npt.NDArray[sym.Expression]:
        # Project sliding velocity along friction force direction
        return self.tangential_jacobian.dot(
            np.vstack((self.body_a.vel.x, self.body_b.vel.x))
        )

    @property
    def allowed_contact_modes(self) -> List[ContactModeType]:
        assert self.contact_modes is not None
        return list(self.contact_modes.keys())

    def _get_jacobian_for_bodies(
        self, bodies: List[RigidBody], local_jacobian: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # (1, num_bodies * num_dims)
        jacobian_for_all_bodies = np.zeros(
            (len(local_jacobian), len(bodies) * self.dim)
        )

        body_a_idx_in_J = bodies.index(self.body_a) * self.dim
        body_b_idx_in_J = bodies.index(self.body_b) * self.dim
        body_a_cols_in_J = np.arange(body_a_idx_in_J, body_a_idx_in_J + self.dim)
        body_b_cols_in_J = np.arange(body_b_idx_in_J, body_b_idx_in_J + self.dim)
        body_a_cols_in_local_J = np.arange(0, self.dim)
        body_b_cols_in_local_J = np.arange(self.dim, 2 * self.dim)

        jacobian_for_all_bodies[:, body_a_cols_in_J] = local_jacobian[
            :, body_a_cols_in_local_J
        ]
        jacobian_for_all_bodies[:, body_b_cols_in_J] = local_jacobian[
            :, body_b_cols_in_local_J
        ]
        return jacobian_for_all_bodies

    def get_tangential_jacobian_for_bodies(
        self, bodies: List[RigidBody]
    ) -> npt.NDArray[np.float64]:
        return self._get_jacobian_for_bodies(bodies, self.tangential_jacobian)

    def get_normal_jacobian_for_bodies(
        self, bodies: List[RigidBody]
    ) -> npt.NDArray[np.float64]:
        return self._get_jacobian_for_bodies(bodies, self.normal_jacobian)

    def add_constraint_to_all_modes(self, constraints) -> None:
        self.additional_constraints = sum(
            [self.additional_constraints, constraints], []
        )

    def add_force_balance(self, force_balance):
        self.force_balance = force_balance

    def get_contact_mode(self, contact_mode: ContactModeType) -> ContactMode:
        if not self.contact_modes_formulated:
            raise RuntimeError("Contact modes not formulated for {self.name}")
        return self.contact_modes[contact_mode]

    def formulate_contact_modes(
        self,
        all_variables: npt.NDArray[sym.Variable],
        allow_sliding: bool = False,
    ):
        if self.contact_modes_formulated:
            raise ValueError(f"Contact modes already formulated for {self.name}")

        if self.force_balance is None:
            raise ValueError(
                "Force balance must be set before formulating contact modes"
            )

        position_mode_constraints = self._create_position_mode_constraints(
            self.body_a, self.body_b, self.position_mode
        )

        modes_constraints = {}

        if ContactModeType.NO_CONTACT in self.allowable_contact_mode_types:
            modes_constraints[ContactModeType.NO_CONTACT] = [
                ge(self.sdf, 0),
                *[
                    eq(p.lam_n, 0)
                    for p in self.position_type_force_variable_pairs.values()
                ],
                *[
                    le(p.lam_f, self.friction_coeff * p.lam_n)
                    for p in self.position_type_force_variable_pairs.values()
                ],
                *[
                    ge(p.lam_f, -self.friction_coeff * p.lam_n)
                    for p in self.position_type_force_variable_pairs.values()
                ],
                *position_mode_constraints,
                *self.force_balance,
                *self.additional_constraints,
            ]

        no_contact_position_modes = [
            PositionModeType.TOP_LEFT,
            PositionModeType.TOP_RIGHT,
            PositionModeType.BOTTOM_LEFT,
            PositionModeType.BOTTOM_RIGHT,
        ]
        if self.position_mode not in no_contact_position_modes:
            if ContactModeType.ROLLING in self.allowable_contact_mode_types:
                modes_constraints[ContactModeType.ROLLING] = [
                    eq(self.sdf, 0),
                    ge(self.lam_n, 0),
                    *[
                        eq(p.lam_n, 0)
                        for mode, p in self.position_type_force_variable_pairs.items()
                        if mode != self.position_mode
                    ],
                    eq(self.rel_tangential_sliding_vel, 0),
                    *[
                        le(p.lam_f, self.friction_coeff * p.lam_n)
                        for p in self.position_type_force_variable_pairs.values()
                    ],
                    *[
                        ge(p.lam_f, -self.friction_coeff * p.lam_n)
                        for p in self.position_type_force_variable_pairs.values()
                    ],
                    *position_mode_constraints,
                    *self.force_balance,
                    *self.additional_constraints,
                ]

            if allow_sliding:
                # Outside horizontal friction cone (max negative friction force as sliding
                # right) and inside vertical friction cone
                if ContactModeType.SLIDING_RIGHT in self.allowable_contact_mode_types:
                    modes_constraints[ContactModeType.SLIDING_RIGHT] = [
                        eq(self.sdf, 0),
                        ge(self.lam_n, 0),
                        *[
                            eq(p.lam_n, 0)
                            for mode, p in self.position_type_force_variable_pairs.items()
                            if mode != self.position_mode
                        ],
                        ge(self.rel_tangential_sliding_vel[0], 0),
                        eq(self.rel_tangential_sliding_vel[1], 0),
                        eq(self.lam_f[0], -self.friction_coeff * self.lam_n),
                        le(self.lam_f[1], self.friction_coeff * self.lam_n),
                        ge(self.lam_f[1], -self.friction_coeff * self.lam_n),
                        *[
                            eq(p.lam_f, 0)
                            for mode, p in self.position_type_force_variable_pairs.items()
                            if mode != self.position_mode
                        ],
                        *position_mode_constraints,
                        *self.force_balance,
                        *self.additional_constraints,
                    ]
                if ContactModeType.SLIDING_LEFT in self.allowable_contact_mode_types:
                    modes_constraints[ContactModeType.SLIDING_LEFT] = [
                        eq(self.sdf, 0),
                        ge(self.lam_n, 0),
                        *[
                            eq(p.lam_n, 0)
                            for mode, p in self.position_type_force_variable_pairs.items()
                            if mode != self.position_mode
                        ],
                        le(self.rel_tangential_sliding_vel[0], 0),
                        eq(self.rel_tangential_sliding_vel[1], 0),
                        eq(self.lam_f[0], self.friction_coeff * self.lam_n),
                        le(self.lam_f[1], self.friction_coeff * self.lam_n),
                        ge(self.lam_f[1], -self.friction_coeff * self.lam_n),
                        *[
                            eq(p.lam_f, 0)
                            for mode, p in self.position_type_force_variable_pairs.items()
                            if mode != self.position_mode
                        ],
                        *position_mode_constraints,
                        *self.force_balance,
                        *self.additional_constraints,
                    ]

                # Inside horizontal friction cone, outside vertical friction cone
                if ContactModeType.SLIDING_UP in self.allowable_contact_mode_types:
                    modes_constraints[ContactModeType.SLIDING_UP] = [
                        eq(self.sdf, 0),
                        ge(self.lam_n, 0),
                        *[
                            eq(p.lam_n, 0)
                            for mode, p in self.position_type_force_variable_pairs.items()
                            if mode != self.position_mode
                        ],
                        ge(self.rel_tangential_sliding_vel[1], 0),
                        eq(self.rel_tangential_sliding_vel[0], 0),
                        eq(self.lam_f[1], -self.friction_coeff * self.lam_n),
                        le(self.lam_f[0], self.friction_coeff * self.lam_n),
                        ge(self.lam_f[0], -self.friction_coeff * self.lam_n),
                        *[
                            eq(p.lam_f, 0)
                            for mode, p in self.position_type_force_variable_pairs.items()
                            if mode != self.position_mode
                        ],
                        *position_mode_constraints,
                        *self.force_balance,
                        *self.additional_constraints,
                    ]
                if ContactModeType.SLIDING_DOWN in self.allowable_contact_mode_types:
                    modes_constraints[ContactModeType.SLIDING_DOWN] = [
                        eq(self.sdf, 0),
                        ge(self.lam_n, 0),
                        *[
                            eq(p.lam_n, 0)
                            for mode, p in self.position_type_force_variable_pairs.items()
                            if mode != self.position_mode
                        ],
                        le(self.rel_tangential_sliding_vel[1], 0),
                        eq(self.rel_tangential_sliding_vel[0], 0),
                        eq(self.lam_f[1], self.friction_coeff * self.lam_n),
                        le(self.lam_f[0], self.friction_coeff * self.lam_n),
                        ge(self.lam_f[0], -self.friction_coeff * self.lam_n),
                        *[
                            eq(p.lam_f, 0)
                            for mode, p in self.position_type_force_variable_pairs.items()
                            if mode != self.position_mode
                        ],
                        *position_mode_constraints,
                        *self.force_balance,
                        *self.additional_constraints,
                    ]

        self.contact_modes = {
            mode_type: ContactMode(
                f"{self.name}",
                contact_constraints,
                all_variables,
                mode_type,
            )
            for mode_type, contact_constraints in modes_constraints.items()
        }
        # TODO I HATE this, very against functional programming principles. Find an alternative?
        self.contact_modes_formulated = True


class ObjectPairHandler:
    def __init__(
        self,
        all_decision_vars: List[sym.Variable],  # TODO this should not be here
        rigid_bodies: List[RigidBody],
        object_pairs: List["ObjectPair"],  # TODO Will be removed
        external_forces: List[sym.Expression],
        additional_constraints: Optional[List[sym.Formula]],
        allow_sliding: bool = False,
    ) -> None:
        self.all_decision_vars = all_decision_vars
        self.rigid_bodies = rigid_bodies
        self.object_pairs = object_pairs
        unactuated_dofs = self._get_unactuated_dofs(
            self.rigid_bodies, self.position_dim
        )

        contact_pairs_nested = [
            object_pair.contact_pairs for object_pair in self.object_pairs
        ]
        self.contact_pairs = reduce(lambda a, b: a + b, contact_pairs_nested)

        force_balance_constraints = self.construct_force_balance(
            self.contact_pairs,
            self.rigid_bodies,
            external_forces,
            unactuated_dofs,
        )
        for p in self.contact_pairs:
            p.add_force_balance(force_balance_constraints)
        for p in self.contact_pairs:
            p.add_constraint_to_all_modes(additional_constraints)

        for p in self.contact_pairs:
            p.formulate_contact_modes(self.all_decision_vars, allow_sliding)

    @property
    def position_dim(self) -> int:
        return self.rigid_bodies[0].dim

    @property
    def collision_pairs_by_name(self) -> Dict[str, ContactPair]:
        return {
            contact_pair.name: contact_pair
            for object_pair in self.object_pairs
            for contact_pair in object_pair.contact_pairs
        }

    def _get_unactuated_dofs(
        self, rigid_bodies: List[RigidBody], dim: int
    ) -> npt.NDArray[np.int32]:
        unactuated_idxs = [i for i, b in enumerate(rigid_bodies) if not b.actuated]
        unactuated_dofs = np.concatenate(
            [np.arange(idx * dim, (idx + 1) * dim) for idx in unactuated_idxs]
        )
        return unactuated_dofs

    @staticmethod
    def construct_force_balance(
        contact_pairs: List[ContactPair],
        bodies: List[RigidBody],
        external_forces: npt.NDArray[sym.Expression],
        unactuated_dofs: npt.NDArray[np.int64],
    ) -> List[sym.Formula]:
        # Enforce force balance at Bezier control points
        normal_jacobians = np.vstack(
            [p.get_normal_jacobian_for_bodies(bodies) for p in contact_pairs]
        )
        tangential_jacobians = np.vstack(
            [p.get_tangential_jacobian_for_bodies(bodies) for p in contact_pairs]
        )

        normal_forces = np.concatenate([p.lam_n for p in contact_pairs])
        friction_forces = np.concatenate([p.lam_f for p in contact_pairs])

        # Projection of force is scalar as project onto single dimension
        all_force_balances = eq(
            normal_jacobians.T.dot(
                normal_forces
            )  # Projection of normal force along normal force direction
            + tangential_jacobians.T.dot(
                friction_forces
            )  # Projection of friction force along friction force direction
            + external_forces,
            0,
        )
        force_balance = all_force_balances[unactuated_dofs, :]
        return force_balance

    def all_possible_contact_cfg_perms(self) -> List[ContactModeConfig]:
        # [(n_m), (n_m), ... (n_m)] n_p times --> n_m * n_p
        all_allowed_contact_modes = [
            [
                (contact_pair.name, mode)
                for contact_pair in object_pair.contact_pairs
                for mode in contact_pair.allowed_contact_modes
            ]
            for object_pair in self.object_pairs
        ]
        # Cartesian product:
        # S = P_1 X P_2 X ... X P_n_p
        # |S| = |P_1| * |P_2| * ... * |P_n_p|
        #     = n_m * n_m * ... * n_m
        #     = n_m^n_p
        # All possible permuations between object pairs
        all_possible_permutations = [
            ContactModeConfig({name: mode for name, mode in perm})
            for perm in itertools.product(*all_allowed_contact_modes)
        ]
        return all_possible_permutations

    def create_convex_set_from_mode_config(
        self,
        config: "ContactModeConfig",
        name: Optional[str] = None,
    ) -> Optional[Tuple[ConvexSet, str]]:
        contact_modes = [
            self.collision_pairs_by_name[pair].get_contact_mode(mode)
            for pair, mode in config.modes.items()
        ]
        intersects, (
            calculated_name,
            intersection,
        ) = calc_intersection_of_contact_modes(contact_modes)

        # breakpoint()
        if not intersects:
            return None

        if config.additional_constraints is not None:
            additional_set = PolyhedronFormulator(
                config.additional_constraints
            ).formulate_polyhedron(self.all_decision_vars)
            intersects = intersects and intersection.IntersectsWith(additional_set)
            if not intersects:
                return None

            intersection = intersection.Intersection(additional_set)

        name = f"{name}: {calculated_name}" if name is not None else calculated_name
        return intersection, name
