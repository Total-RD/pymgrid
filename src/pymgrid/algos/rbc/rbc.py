from copy import deepcopy

from pymgrid import Microgrid
from pymgrid.algos.priority_list import PriorityListAlgo


class RuleBasedControl(PriorityListAlgo):
    """
    Parameters
    ----------
    microgrid : pymgrid.Microgrid

    """
    def __init__(self, microgrid):
        super().__init__()
        self._microgrid = deepcopy(microgrid)
        self._priority_list = self._get_priority_list()

    def _get_priority_list(self):
        """
        Given a microgrid, return the optimal order of module deployment.
        Returns
        -------

        """
        return sorted(self._get_priority_lists()[0])

    def _get_action(self):
        """
        Given the priority list, define an action.
        Returns
        -------

        """
        return self._populate_action(self._priority_list)

    def reset(self):
        return self._microgrid.reset()

    def run(self, max_steps=None):
        """
        Get the priority list and then deploy on the microgrid for some number of steps.

        Parameters
        ---------
        max_steps : int or None, default None
            Maximum number of RBC steps. If None, run until the microgrid terminates.

        Returns
        -------

        """
        pass

    def get_empty_action(self):
        return self._microgrid.get_empty_action()

    @property
    def microgrid(self):
        """
        View of the microgrid.

        Returns
        -------
        microgrid : :class:`pymgrid.Microgrid`
            The microgrid that RBC is being run on.

        """
        return self._microgrid

    @property
    def fixed(self):
        return self._microgrid.fixed

    @property
    def flex(self):
        return self._microgrid.flex

    @property
    def modules(self):
        return self._microgrid.modules

    @property
    def priority_list(self):
        return self._priority_list
