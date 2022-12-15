from copy import deepcopy
from tqdm import tqdm

from pymgrid.algos.priority_list import PriorityListAlgo


class RuleBasedControl(PriorityListAlgo):
    """

    Run a rule-based (heuristic) control algorithm on a microgrid.

    In rule-based control, modules are deployed in a preset order. You can either define this order by passing a
    priority list or the order will be defined automatically from the module with the lowest marginal cost to the
    highest.

    Parameters
    ----------
    microgrid : :class:`pymgrid.Microgrid`
        Microgrid on which to run rule-based control.

    priority_list : list of :class:`.PriorityListElement` or None, default None.
        Priority list to use. If None, the order will be defined automatically from the module with the lowest marginal
        cost to the highest.

    """
    def __init__(self, microgrid, priority_list=None):
        super().__init__()
        self._microgrid = deepcopy(microgrid)
        self._priority_list = self._get_priority_list(priority_list)

    def _get_priority_list(self, priority_list):
        """
        Given a microgrid, return the optimal order of module deployment.
        """
        priority_lists = self.get_priority_lists()

        if priority_list is None:
            return sorted(priority_lists[0])

        if priority_list not in priority_lists:
            raise ValueError('Invalid priority list. Use RuleBasedControl.get_priority_lists to view all '
                             'valid priority lists.')

        return priority_list

    def _get_action(self):
        """
        Given the priority list, define an action.
        """
        return self._populate_action(self._priority_list)

    def reset(self):
        """
        Reset the underlying microgrid.

        Returns
        -------
        obs : dict[str, list[float]]
            Observations from resetting the modules as well as the flushed balance log.

        """
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
        log : pd.DataFrame
            Results of running the rule-based control algorithm.

        """
        if max_steps is None:
            max_steps = len(self.microgrid)

        self.reset()

        for _ in tqdm(range(max_steps)):
            action = self._get_action()
            _, _, done, _ = self._microgrid.run(action, normalized=False)
            if done:
                break

        return self._microgrid.get_log(as_frame=True)

    def get_empty_action(self):
        """
        :meta private:
        """
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
        """:meta private:"""
        return self._microgrid.fixed

    @property
    def flex(self):
        """:meta private:"""
        return self._microgrid.flex

    @property
    def modules(self):
        """:meta private:"""
        return self._microgrid.modules

    @property
    def priority_list(self):
        """
        Order in which to deploy controllable modules.

        Returns
        -------
        priority_list: list of :class:`.PriorityListElement`
            Priority list.

        """
        return self._priority_list
