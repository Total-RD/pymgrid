import yaml

from gym.spaces import Discrete
from warnings import warn

from pymgrid.algos.priority_list import PriorityListAlgo
from pymgrid.envs.base import BaseMicrogridEnv


class DiscreteMicrogridEnv(BaseMicrogridEnv, PriorityListAlgo):
    """
    A discrete env that implements priority lists as actions on a microgrid.

    The env assumes that you need to meet the consumption in fixed sink (e.g. load) modules and that
    you would like to use as much of your flex source modules (e.g. PV) as possible.

    The env assumes that all module actions are either singletons or have length two. In the latter case, assumes that
    the first value is boolean. This

    Attributes
    -----------------
    actions_list: List[Tuple[Tuple]]
        List of priority tuples. Each priority tuple defines the order in which to prioritize fixed source modules.
            Each tuple contains three elements (module_name, total_actions_for_{module_name}, action_num).
            For example: (('genset', 0), 2, 1) is a tuple defining the first element (of two) for ('genset', 0).
    """

    yaml_tag = u"!DiscreteMicrogridEnv"
    yaml_loader = yaml.SafeLoader
    yaml_dumper = yaml.SafeDumper

    def __init__(self,
                 modules,
                 add_unbalanced_module=True,
                 loss_load_cost=10,
                 overgeneration_cost=2
                 ):
        super().__init__(modules,
                         add_unbalanced_module=add_unbalanced_module,
                         loss_load_cost=loss_load_cost,
                         overgeneration_cost=overgeneration_cost)

        self.action_space, self.actions_list = self._get_action_space()

    def _get_action_space(self):
        """
        An action here is a priority list - in what order to deploy fixed source modules.
        Compute the total expected load
        It exhausts the entire production of the 0th source module before moving on to the second and the third, etc.

        If there are n fixed source modules, then there are 2^{n-1} actions.
        :return:
        """

        priority_lists = self.get_priority_lists()

        n_actions = len(priority_lists)

        if n_actions > 1000:
            warn(f'Microgrid with {len(priority_lists[0])} fixed source modules defines large action space with '
                 f'{n_actions} elements.')

        space = Discrete(n_actions)

        return space, priority_lists

    def _get_action(self, action_num):
        if action_num not in self.action_space:
            raise ValueError(f" Action {action_num} not in action space {self.action_space}")

        priority_list = list(self.actions_list[action_num])

        return self._populate_action(priority_list)


    def step(self, action):
        self._microgrid_logger.log(action=action)
        microgrid_action = self._get_action(action)
        return super().step(microgrid_action, normalized=False)

    def sample_action(self, strict_bound=False, sample_flex_modules=False):
        return self.action_space.sample()

    def __repr__(self):
        return f"DiscreteMicrogridEnv({super().__repr__()}"

    def __str__(self):
        return self.__repr__()
