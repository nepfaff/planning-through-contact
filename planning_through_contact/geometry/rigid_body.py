from dataclasses import dataclass
from typing import Literal

import numpy.typing as npt
import pydrake.symbolic as sym

from planning_through_contact.geometry.bezier import BezierVariable


@dataclass
class RigidBody:
    name: str
    dim: int
    geometry: Literal["point", "box", "sphere"]
    width: float = 0  # TODO generalize
    height: float = 0
    depth: float = 0
    radius: float = 0
    position_curve_order: int = 1
    actuated: bool = False

    def __post_init__(self) -> None:
        self.pos = BezierVariable(
            self.dim, self.position_curve_order, name=f"{self.name}_pos"
        )

    @property
    def vel(self) -> BezierVariable:
        return self.pos.get_derivative()

    @property
    def pos_x(self) -> npt.NDArray[sym.Expression]:
        return self.pos.x[0, :]

    @property
    def pos_y(self) -> npt.NDArray[sym.Expression]:
        return self.pos.x[1, :]

    @property
    def pos_z(self) -> npt.NDArray[sym.Expression]:
        return self.pos.x[2, :]
