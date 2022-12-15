from dataclasses import dataclass
from functools import total_ordering
from typing import Tuple, Optional


@total_ordering
@dataclass(frozen=True, eq=False)
class PriorityListElement:
    module: Tuple[str, int]
    module_actions: int
    action: int
    marginal_cost: Optional[float] = None

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented

        return (
                self.module == other.module and
                self.module_actions == other.module_actions and
                self.action == other.action and
                self.marginal_cost == other.marginal_cost
        )

    def __lt__(self, other):
        if type(self) != type(other) or self.marginal_cost is None or other.marginal_cost is None:
            return NotImplemented

        return (
                self.marginal_cost < other.marginal_cost or
                (self.marginal_cost == other.marginal_cost and self.action > other.action)
        )
