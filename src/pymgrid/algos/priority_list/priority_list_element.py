from dataclasses import dataclass
from functools import total_ordering
from typing import Tuple, Optional


@total_ordering
@dataclass(frozen=True, eq=False)
class PriorityListElement:
    """
    A dataclass that acts as a representation of a module's position in a deployment order.

    Used in :class:`.RuleBasedControl` and :class:`.DiscreteMicrogridEnv` to define actions given a specific order
    of module deployment.

    In :class:`.RuleBasedControl`, one list of these elements is used at every step to define actions.

    In :class:`.DiscreteMicrogridEnv`, each action in the environment's action space corresponds to a distinct priority
    list.

    """

    module: Tuple[str, int]
    """
    Module name.
    """

    module_actions: int
    """
    Number of possible action classes.
    
    * If one, actions generated with this element will be scalar.
    
    * If greater than one, actions generated this element will be a length-2 array. The first element
      is the action number (defined by `action`) and the second is a float corresponding to the scalar case.

    """

    action: int
    """
    Action number.
    
    Value is in {0, ..., `module_actions`-1}.
    
    """

    marginal_cost: Optional[float] = None
    """
    Marginal cost of using the module.
    
    Defines an ordering on elements.
    """

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
