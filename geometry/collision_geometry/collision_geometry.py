from abc import ABC, abstractmethod
from enum import Enum
from typing import List, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.geometry import Shape as DrakeShape

from geometry.hyperplane import Hyperplane


class ContactLocation(Enum):
    FACE = 1
    VERTEX = 2


# TODO: move this?
class PolytopeContactLocation(NamedTuple):
    pos: ContactLocation
    idx: int


class CollisionGeometry(ABC):
    """
    Abstract class for all of the collision geometries supported by the contact planner,
    with all of the helper functions required by the planner.
    """

    @abstractmethod
    def get_proximate_vertices_from_location(
        self, location: PolytopeContactLocation
    ) -> List[npt.NDArray[np.float64]]:
        pass

    @abstractmethod
    def get_neighbouring_vertices(
        self, location: PolytopeContactLocation
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pass

    @abstractmethod
    def get_hyperplane_from_location(
        self, location: PolytopeContactLocation
    ) -> Hyperplane:
        pass

    @abstractmethod
    def get_norm_and_tang_vecs_from_location(
        self, location: PolytopeContactLocation
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pass

    @property
    @abstractmethod
    def vertices_for_plotting(self) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_face_length(self, location: PolytopeContactLocation) -> float:
        pass

    def get_shortest_vec_from_com_to_face(
        self, location: PolytopeContactLocation
    ) -> npt.NDArray[np.float64]:
        v1, v2 = self.get_proximate_vertices_from_location(location)
        vec = (v1 + v2) / 2
        return vec

    @classmethod
    @abstractmethod
    def from_drake(cls, drake_shape: DrakeShape):
        pass
