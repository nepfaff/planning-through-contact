from dataclasses import dataclass, field
from typing import List

from planning_through_contact.geometry.contact_mode import (
    PositionModeType,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.geometry.contact_pair import ContactPair
from planning_through_contact.geometry.contact_mode import (
    ContactModeType,
    ForceVariablePair,
)
from planning_through_contact.geometry.bezier import BezierVariable


@dataclass
class ObjectPair:
    body_a: RigidBody
    body_b: RigidBody
    friction_coeff: float
    allowed_position_modes: List[PositionModeType]
    force_curve_order: int = 1
    allowable_contact_mode_types: List[ContactModeType] = field(
        default_factory=lambda: []
    )

    def __post_init__(self):
        position_type_force_variable_pairs = {}
        for position_mode in self.allowed_position_modes:
            base_name = f"({self.body_a.name},{self.body_b.name},{position_mode.name})"
            lam_n = BezierVariable(
                dim=1, order=self.force_curve_order, name=f"{base_name}_lam_n"
            ).x  # Normal force
            lam_f = BezierVariable(
                dim=2, order=self.force_curve_order, name=f"{base_name}_lam_f"
            ).x  # Friction force
            position_type_force_variable_pairs[position_mode] = ForceVariablePair(
                lam_n=lam_n, lam_f=lam_f
            )

        self.contact_pairs = [
            ContactPair(
                self.body_a,
                self.body_b,
                self.friction_coeff,
                position_mode,
                position_type_force_variable_pairs,
                self.force_curve_order,
                self.allowable_contact_mode_types,
            )
            for position_mode in self.allowed_position_modes
        ]
