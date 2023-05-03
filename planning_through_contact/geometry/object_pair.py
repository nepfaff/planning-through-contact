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
    transition_eps: float = 0
    center_contact_buffer: float = 0

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
                self.transition_eps,
                self.center_contact_buffer,
            )
            for position_mode in self.allowed_position_modes
        ]
    
    def get_contact_pair_for_positions(self, body_a_pos, body_b_pos) -> ContactPair:
        # Renaming to make the conditions more readable
        body_a_pos_x = body_a_pos[0]
        body_a_pos_y = body_a_pos[1]
        body_a_pos_z = body_a_pos[2]
        body_b_pos_x = body_b_pos[0]
        body_b_pos_y = body_b_pos[1]
        body_b_pos_z = body_b_pos[2]

        if self.body_a.geometry == "box" and self.body_b.geometry == "box":
            z_constraint_above = body_a_pos_z - self.body_a.depth >= body_b_pos_z + self.body_b.depth
            x_constraint_left = body_a_pos_x + self.body_a.width >= body_b_pos_x - self.body_b.width
            x_constraint_right = body_a_pos_x - self.body_a.width <= body_b_pos_x + self.body_b.width
            y_constraint_left = body_a_pos_y + self.body_a.height >= body_b_pos_y - self.body_b.height
            y_constraint_right = body_a_pos_y - self.body_a.height <= body_b_pos_y + self.body_b.height
            if z_constraint_above and x_constraint_left and x_constraint_right and y_constraint_left and y_constraint_right:
                pos_mode = PositionModeType.FRONT  
            else:
                raise NotImplementedError(f"get_contact_pair_for_positions for {self.body_a.geometry} and {self.body_b.geometry} that is not FRONT not implemented")
        elif self.body_a.geometry == "point" and self.body_b.geometry == "box":
            box = self.body_b
            point = self.body_a
            raise NotImplementedError("Not implemented for transition modes")
            y_le_top = body_a_pos_y <= body_b_pos_y + box.height
            y_ge_top = body_a_pos_y >= body_b_pos_y + box.height
            y_le_bottom = body_a_pos_y <= body_b_pos_y - box.height
            y_ge_bottom = body_a_pos_y >= body_b_pos_y - box.height
            x_le_left = body_a_pos_x <= body_b_pos_x - box.width
            x_ge_left = body_a_pos_x >= body_b_pos_x - box.width
            x_le_right = body_a_pos_x <= body_b_pos_x + box.width
            x_ge_right = body_a_pos_x >= body_b_pos_x + box.width

            if y_le_top and y_ge_bottom:
                if x_le_left:
                    pos_mode = PositionModeType.LEFT
                elif x_ge_right:
                    pos_mode = PositionModeType.RIGHT
            elif x_ge_left and x_le_right:
                if y_ge_top:
                    pos_mode = PositionModeType.TOP
                elif y_le_bottom:
                    pos_mode = PositionModeType.BOTTOM
            elif y_ge_top and x_le_left:
                pos_mode = PositionModeType.TOP_LEFT
            elif y_ge_top and x_ge_right:
                pos_mode = PositionModeType.TOP_RIGHT
            elif y_le_bottom and x_le_left:
                pos_mode = PositionModeType.BOTTOM_LEFT
            elif y_le_bottom and x_ge_right:
                pos_mode = PositionModeType.BOTTOM_RIGHT
            else:
                raise NotImplementedError(f"get_contact_pair_for_positions for {self.body_a.geometry}: {body_a_pos} and {self.body_b.geometry}: {body_b_pos} not implemented")
        elif self.body_a.geometry == "sphere" and self.body_b.geometry == "box":
            box = self.body_b
            sphere = self.body_a

            y_le_top = body_a_pos_y <= body_b_pos_y + box.height
            y_ge_top = body_a_pos_y >= body_b_pos_y + box.height
            y_le_bottom = body_a_pos_y <= body_b_pos_y - box.height
            y_ge_bottom = body_a_pos_y >= body_b_pos_y - box.height
            x_le_left = body_a_pos_x <= body_b_pos_x - box.width
            x_ge_left = body_a_pos_x >= body_b_pos_x - box.width
            x_le_right = body_a_pos_x <= body_b_pos_x + box.width
            x_ge_right = body_a_pos_x >= body_b_pos_x + box.width


        else:
            raise NotImplementedError(f"get_contact_pair_for_positions for {self.body_a.geometry} and {self.body_b.geometry} not implemented")
        for pair in self.contact_pairs:
            if pair.position_mode == pos_mode:
                return pair
        raise ValueError("body_a is {pos_mode} relative to body_b, but no contact pair with position mode {pos_mode} found")