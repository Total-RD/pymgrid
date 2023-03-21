import numpy as np
import pandas as pd

from abc import abstractmethod
from itertools import permutations


from gym.spaces import Discrete

from pymgrid.algos.priority_list import PriorityListElement as Element
from pymgrid.modules import GensetModule


class PriorityListAlgo:
    def get_priority_lists(self, remove_redundant_gensets):
        """
        Get all of the priority lists for the microgrid.

        A priority list is an order in which to deploy all of the controllable modules of the microgrid.

        Returns
        -------
        priority_lists : list of list of :class:`.PriorityListElement`
            List of all priority lists.

        """
        controllable_sources = [Element(module.name, module.action_space.shape[0], n_actions, module.marginal_cost)
                                for module in self.modules.controllable.sources.iterlist()
                                for n_actions in range(module.action_space.shape[0])]

        controllable_sources.extend([Element(module.name, module.action_space.shape[0], n_actions, module.marginal_cost)
                                     for module in self.modules.controllable.source_and_sinks.iterlist()
                                     for n_actions in range(module.action_space.shape[0])])

        all_permutations = permutations(controllable_sources)
        priority_lists = self._remove_redundant_actions(all_permutations, gensets=remove_redundant_gensets)

        return priority_lists

    def _remove_redundant_actions(self, priority_lists, gensets=False):
        pls = []
        for pl in priority_lists:
            is_redundant = pd.DataFrame(el.module for el in pl).duplicated()
            pls.append(tuple(el for j, el in enumerate(pl) if not is_redundant.iloc[j]))

        unique_pls = list(dict.fromkeys(pls))

        if gensets:
            unique_pls = self._remove_redundant_gensets(unique_pls)

        return unique_pls

    def _remove_redundant_gensets(self, priority_lists):
        redundant_genset_actions = []
        for module_name, module_list in self.modules.iterdict():
            for module_n, module in enumerate(module_list):
                if isinstance(module, GensetModule):
                    if module.running_min_production == 0:
                        removable_element = Element(
                            module=(module_name, module_n),
                            module_actions=2,
                            action=0,
                            marginal_cost=module.marginal_cost
                        )
                        redundant_genset_actions.append(removable_element)

        return [el for el in priority_lists if not any(redundant in el for redundant in redundant_genset_actions)]

    def _populate_action(self, priority_list):
        action = self.get_empty_action()
        loads, total_load = self._get_load()
        renewable = self._get_renewable()
        assert total_load >= 0 and renewable >= 0

        remaining_load = (total_load-renewable).item()

        for element in priority_list:
            module_name, module_number = element.module
            total_module_actions = element.module_actions
            module_action_number = element.action

            module_to_deploy = self.modules[module_name][module_number]

            if total_module_actions > 1:
                if action[module_name][module_number] is not None:
                    # Already hit this module in the priority list (as it has multiple elements)
                    continue
                else:
                    action[module_name][module_number] = [module_action_number]

            if np.isclose(remaining_load, 0.0, atol=1e-4):
                # Don't need to do anything
                module_energy = 0.0
            elif remaining_load > 0:
                # Need to produce
                module_energy = self._produce_from_module(module_action_number, module_to_deploy, remaining_load)
            else:
                # Need to consume. These are sources and sources_and_sinks, so need to only use sources_and_sinks.
                module_energy = self._consume_in_module(module_to_deploy, remaining_load)

            try:
                action[module_name][module_number].append(module_energy)
            except AttributeError:
                action[module_name][module_number] = module_energy

            remaining_load -= module_energy

            if total_module_actions > 1:
                # If we have, e.g. a genset (with two actions)
                action[module_name][module_number] = np.array(action[module_name][module_number])

        bad_keys = [k for k, v in action.items() if v is None]
        if len(bad_keys):
            raise RuntimeError(f'None values found in action, corresponding to keys\n\t{bad_keys}')

        return action

    def _consume_in_module(self, module_to_deploy, remaining_load):
        module_max_consumption = module_to_deploy.max_consumption

        assert remaining_load <= 0.0

        if module_to_deploy.is_sink:
            assert module_max_consumption >= 0

            if -1 * remaining_load > module_to_deploy.max_consumption:
                # Can't consume it all
                module_consumption = -1.0 * module_to_deploy.max_consumption
            else:
                # Can consume all
                module_consumption = remaining_load
        else:  # Not a sink
            module_consumption = 0.0

        assert module_consumption <= 0
        return module_consumption

    def _produce_from_module(self, module_action_number, module_to_deploy, remaining_load):
        try:
            max_production = module_to_deploy.next_max_production(module_action_number)
            min_production = module_to_deploy.next_min_production(module_action_number)
        except AttributeError:
            max_production, min_production = module_to_deploy.max_production, module_to_deploy.min_production
        if min_production <= remaining_load <= max_production:
            # Module can meet demand
            module_production = remaining_load
        elif remaining_load < min_production:
            # Module production too much
            module_production = min_production
        else:
            # Module production not enough
            module_production = max_production

        assert module_production >= 0
        return module_production

    def _get_load(self):
        loads = dict()
        total_load = 0.0
        for fixed_sink in self.fixed.sinks.iterlist():
            loads[fixed_sink.name] = fixed_sink.max_consumption
            total_load += fixed_sink.max_consumption

        return loads, total_load

    def _get_renewable(self):
        return np.sum([flex_source.max_production for flex_source in self.flex.sources.iterlist()])

    @property
    @abstractmethod
    def modules(self):
        pass

    @property
    @abstractmethod
    def fixed(self):
        pass

    @property
    @abstractmethod
    def flex(self):
        pass

    @abstractmethod
    def get_empty_action(self):
        pass
