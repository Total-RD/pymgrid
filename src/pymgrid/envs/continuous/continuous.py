from gym.spaces import Dict, Tuple, flatten_space
from warnings import warn

from pymgrid.microgrid.envs.base.base import BaseMicrogridEnv


class ContinuousMicrogridEnv(BaseMicrogridEnv):
    def __init__(self,
                 modules,
                 add_unbalanced_module=True,
                 loss_load_cost=10,
                 overgeneration_cost=2,
                 flat_spaces=True
                 ):

        self._nested_action_space = self._get_nested_action_space()
        super().__init__(modules,
                         add_unbalanced_module=add_unbalanced_module,
                         loss_load_cost=loss_load_cost,
                         overgeneration_cost=overgeneration_cost,
                         flat_spaces=flat_spaces)

    def _get_nested_action_space(self):
        return Dict({name: Tuple([module.action_spaces['normalized'] for module in modules_list])
                                 for name, modules_list in self.fixed.iterdict() if modules_list[0].is_source})

    def _get_action_space(self):
        return flatten_space(self._nested_action_space) if self._flat_spaces else self._nested_action_space

    def _get_action(self, action):
        # Action does not have fixed sinks (loads); add those values.
        assert action in self._nested_action_space, 'Action is not in action space.'
        action = action.copy()
        for name, module_list in self.fixed.sinks.iterdict():
            action[name] = [module.to_normalized(-1 * module.max_consumption, act=True) for module in module_list]
        return action

    def step(self, action):
        action = self._get_action(action)
        return super().run(action)

    def run(self, action, normalized=True):
        warn('run() should not be called directly in environments.')
        return super().run(action, normalized=normalized)
