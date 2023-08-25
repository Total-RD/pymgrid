from pymgrid.algos.priority_list import PriorityListElement as Element
from pymgrid.envs import DiscreteMicrogridEnv

import warnings


class IncrementalBatteryDiscreteEnv(DiscreteMicrogridEnv):
    def __init__(self,
                 modules,
                 add_unbalanced_module=True,
                 loss_load_cost=10,
                 overgeneration_cost=2,
                 reward_shaping_func=None,
                 trajectory_func=None,
                 flat_spaces=True,
                 observation_keys=None,
                 remove_redundant_gensets=True,
                 discharge_increments=(1.0, ),
                 discharge_module_names=('battery', )
                 ):

        assert all(0 < inc <= 1.0 for inc in discharge_increments), 'discharge_increments must be in (0, 1].'

        self._discharge_increments = discharge_increments
        self._discharge_module_names = discharge_module_names

        super().__init__(
            modules,
            add_unbalanced_module=add_unbalanced_module,
            loss_load_cost=loss_load_cost,
            overgeneration_cost=overgeneration_cost,
            reward_shaping_func=reward_shaping_func,
            trajectory_func=trajectory_func,
            flat_spaces=flat_spaces,
            observation_keys=observation_keys,
            remove_redundant_gensets=remove_redundant_gensets)

        # self.action_space, self.actions_list = self._get_action_space(
        #     remove_redundant_gensets=remove_redundant_gensets,
        #     n_discharges=n_discharges
        # )

    def _get_action_space(self, remove_redundant_gensets=False, n_discharges=1):
        space, priority_lists = super()._get_action_space(remove_redundant_gensets=remove_redundant_gensets)
        if n_discharges == 1:
            return space, priority_lists

        # deployments = set(sum([list(x) for x in priority_lists], []))
        # additional_deployments = Element() # todo define new elements
        # self.modules['battery']


        raise RuntimeError

    def _get_elements(self):
        controllable_sources = super()._get_elements()
        if len(self._discharge_increments) == 1:
            return controllable_sources

        modules_to_expand = sum([self.modules[name] for name in self._discharge_module_names], [])
        names = [module.name for module in modules_to_expand]

        new_sources = [
            Element(module.name, len(self._discharge_increments), n_actions, module.marginal_cost)
            for module in modules_to_expand
            for n_actions in range(len(self._discharge_increments))
        ]

        new_sources.extend(element for element in controllable_sources if element.module not in names)

        return new_sources

    def module_production_bounds(self, module, module_action_number):
        min_production, parent_max = super().module_production_bounds(module, module_action_number)

        if module.module_type[0] not in self._discharge_module_names:
            return min_production, parent_max
        elif module.module_type[0] != 'battery':
            warnings.warn('This behavior is only defined for a battery module.')
            return min_production, parent_max

        max_discharge = module.max_discharge * (self._discharge_increments[module_action_number])
        max_production = min(max_discharge, module.current_charge-module.min_capacity) * module.efficiency

        if self._discharge_increments[module_action_number] == 1.0:
            assert max_production == parent_max

        return min_production, max_production


if __name__ == '__main__':
    from tests.helpers.modular_microgrid import get_modular_microgrid

    env = IncrementalBatteryDiscreteEnv.from_scenario(get_modular_microgrid(), discharge_increments=[0.5, 1.0])

    action = env.sample_action(strict_bound=True)
    env.step(0)
    print(env.log)