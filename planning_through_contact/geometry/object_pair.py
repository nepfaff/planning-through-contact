from dataclasses import dataclass, field
from typing import List

from planning_through_contact.geometry.contact_mode import (
    PositionModeType,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.geometry.contact_pair import ContactPair
from planning_through_contact.geometry.contact_mode import ContactModeType


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
        self.contact_pairs = [
            ContactPair(
                self.body_a,
                self.body_b,
                self.friction_coeff,
                position_mode,
                self.force_curve_order,
                self.allowable_contact_mode_types,
            )
            for position_mode in self.allowed_position_modes
        ]
