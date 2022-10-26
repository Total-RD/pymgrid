from itertools import permutations
from warnings import warn
import numpy as np
from gym.spaces import Discrete
from math import isclose

from pymgrid.microgrid.envs.base.base import BaseMicrogridEnv
from pymgrid.microgrid.utils.logger import ModularLogger


class DiscreteMicrogridEnv(BaseMicrogridEnv):
    """
    A discrete env that implements priority lists on a microgrid.
    The env assumes that you need to meet the consumption in fixed sink (e.g. load) modules and that
    you would like to use as much of your flex source modules (e.g. PV) as possible.

    The Env assumes that all module actions are either singletons or have length 2. In the latter case, assumes that
        the first value is boolean.

    Attributes
    -----------------
    actions_list: List[Tuple[Tuple]]
        List of priority tuples. Each priority tuple defines the order in which to prioritize fixed source modules.
            Each tuple contains three elements (module_name, total_actions_for_{module_name}, action_num).
            For example: (('genset', 0), 2, 1) is a tuple defining the first element (of two) for ('genset', 0).
    """
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
        self.log_dict = ModularLogger()

    def _get_action_space(self):
        """
        An action here is a priority list - in what order to deploy fixed source modules.
        Compute the total expected load
        It exhausts the entire production of the 0th source module before moving on to the second and the third, etc.

        If there are n fixed source modules, then there are 2^{n-1} actions.
        :return:
        """
        # n_actions = 2**(self.modules.fixed.)
        fixed_sources = [(module.name, module.action_spaces['unnormalized'].shape[0], n_actions)
            for module in self.fixed_modules.sources.iterlist()
            for n_actions in range(module.action_spaces['unnormalized'].shape[0])]

        fixed_sources.extend([ (module.name, module.action_spaces['unnormalized'].shape[0], n_actions)
            for module in self.fixed_modules.source_and_sinks.iterlist()
            for n_actions in range(module.action_spaces['unnormalized'].shape[0])])

        priority_lists = list(permutations(fixed_sources))
        n_actions = len(priority_lists)

        if n_actions > 1000:
            warn(f'Microgrid with {len(fixed_sources)} fixed source modules defines large action space with '
                 f'{n_actions} elements.')

        space = Discrete(n_actions)

        return space, priority_lists

    def _get_action(self, action_num):
        if action_num not in self.action_space:
            raise ValueError(f" Action {action_num} not in action space {self.action_space}")

        action = self.get_empty_action()
        loads, total_load = self._get_load()
        for load_module, load in loads.items():
            module_name, module_num = load_module
            action[module_name][module_num] = -1.0 * load

        renewable = self._get_renewable()
        assert total_load >= 0 and renewable >= 0

        remaining_load = (total_load-renewable).item()
        priority_list = list(self.actions_list[action_num])

        while priority_list:
            (module_name, element_number), total_module_actions, module_action_number = priority_list.pop(0)
            module_to_deploy = self.modules[module_name][element_number]

            if total_module_actions > 1:
                if action[module_name][element_number] is not None: # Already hit this module in the priority list (has multiple elements)
                    continue
                else:
                    action[module_name][element_number] = [module_action_number]

            if isclose(remaining_load, 0.0, abs_tol=1e-4): # Don't need to do anything
                try:
                    action[module_name][element_number].append(0.0)
                except AttributeError:
                    action[module_name][element_number] = 0.0

            elif remaining_load > 0: # Need to produce
                try:
                    max_production = module_to_deploy.next_max_production(module_action_number)
                    min_production = module_to_deploy.next_min_production(module_action_number)
                except AttributeError:
                    max_production, min_production = module_to_deploy.max_production, module_to_deploy.min_production
                if min_production < remaining_load < max_production:
                    # Module can meet demand
                    module_production = remaining_load
                elif remaining_load < min_production:          # Module production too much
                    module_production = min_production
                else:                                                           # Module production not enough
                    module_production = max_production
                remaining_load -= module_production

                try:
                    action[module_name][element_number].append(module_production)
                except AttributeError:
                    action[module_name][element_number] = module_production

            else:                   # Need to consume. These are sources and sources_and_sinks, so need to only use sources_and_sinks.
                if module_to_deploy.is_sink:
                    if remaining_load < -1.0 * module_to_deploy.max_consumption: # Can't consume it all
                        module_consumption = -1.0 * module_to_deploy.max_consumption
                    else:                                           # Can consume
                        module_consumption = remaining_load
                    assert module_consumption <= 0
                    # action[module_name][element_number] = module_consumption
                else:                                           # Not a sink
                    module_consumption = 0.0

                remaining_load += module_consumption
                try:
                    action[module_name][element_number].append(module_consumption)
                except AttributeError:
                    action[module_name][element_number] = module_consumption

            if total_module_actions > 1:
                # If we have, e.g. a genset (with two actions)
                action[module_name][element_number] = np.array(action[module_name][element_number])

        return action

    def step(self, action):
        self.log_dict.log(action=action)
        microgrid_action = self._get_action(action)
        return super().step(microgrid_action, normalized=False)

    def _get_load(self):
        loads = dict()
        total_load = 0.0
        for fixed_sink in self.fixed.sinks.iterlist():
            loads[fixed_sink.name] = fixed_sink.max_consumption.item()
            total_load += fixed_sink.max_consumption

        return loads, total_load

    def _get_renewable(self):
        return np.sum([flex_source.max_production for flex_source in self.flex.sources.iterlist()])

    def sample_action(self, strict_bound=False, sample_flex_modules=False):
        return self.action_space.sample()

    def __repr__(self):
        return f"DiscreteMicrogridEnv({super().__repr__()}"

    def __str__(self):
        return self.__repr__()
