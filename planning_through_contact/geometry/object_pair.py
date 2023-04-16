from dataclasses import dataclass
from typing import List

from planning_through_contact.geometry.contact_mode import (
    PositionModeType,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.geometry.contact_pair import ContactPair


@dataclass
class ObjectPair:
    body_a: RigidBody
    body_b: RigidBody
    friction_coeff: float
    allowed_position_modes: List[PositionModeType]
    force_curve_order: int = 1

    def __post_init__(self):
        self.contact_pairs = [
            ContactPair(
                self.body_a,
                self.body_b,
                self.friction_coeff,
                position_mode,
                self.force_curve_order,
            )
            for position_mode in self.allowed_position_modes
        ]
