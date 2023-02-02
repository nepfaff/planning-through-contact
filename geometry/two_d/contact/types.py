from enum import Enum
from typing import NamedTuple


class ContactType(Enum):
    FACE = 1
    VERTEX = 2


class ContactMode(Enum):
    ROLLING = 1
    SLIDING = 2


class ContactLocation(NamedTuple):
    type: ContactType
    idx: int

