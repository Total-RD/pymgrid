from copy import deepcopy

from pymgrid import Microgrid


class RuleBasedControl:
    def __init__(self, microgrid):
        self._microgrid = deepcopy(microgrid)

    def _get_priority_list(self):
        """
        Given a microgrid, return the optimal order of module deployment.
        Returns
        -------

        """
        pass

    def _get_action(self):
        """
        Given the priority list, define an action.
        Returns
        -------

        """
        pass

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
