from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PriorityListElement:
    module: Tuple[str, int]
    module_actions: int
    action: int
