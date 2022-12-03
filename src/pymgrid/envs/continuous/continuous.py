from gym.spaces import Dict, Tuple, flatten_space
from warnings import warn

from pymgrid.envs.base import BaseMicrogridEnv


class ContinuousMicrogridEnv(BaseMicrogridEnv):
    """
    Microgrid environment with a continuous action space.

    Implements the `OpenAI Gym API <https://www.gymlibrary.dev//>`_ for a microgrid;
    inherits from both :class:`.Microgrid` and :class:`gym.Env`.

    Parameters
    ----------
    modules : list, Microgrid, NonModularMicrogrid, or int.
        The constructor can be called in three ways:

        1. Passing a list of microgrid modules. This is identical to the :class:`.Microgrid` constructor.

        2. Passing a :class:`.Microgrid` or :class:`.NonModularMicrogrid` instance.
           This will effectively wrap the microgrid instance with the Gym API.

        3. Passing an integer in [0, 25).
           This will be result in loading the corresponding `pymgrid25` benchmark microgrids.

    add_unbalanced_module : bool, default True.
        Whether to add an unbalanced energy module to your microgrid. Such a module computes and attributes
        costs to any excess supply or demand.
        Set to True unless ``modules`` contains an :class:`.UnbalancedEnergyModule`.

    loss_load_cost : float, default 10.0
        Cost per unit of unmet demand. Ignored if ``add_unbalanced_module=False``.

    overgeneration_cost : float, default 2.0
        Cost per unit of excess generation.  Ignored if ``add_unbalanced_module=False``.

    flat_spaces : bool, default True
        Whether the environment's spaces should be flat.

        If True, all continuous spaces are :class:`gym:gym.spaces.Box`.

        Otherwise, they are nested :class:`gym:gym.spaces.Dict` of :class:`gym:gym.spaces.Tuple`
        of :class:`gym:gym.spaces.Box`, corresponding to the structure of the ``control`` arg of :meth:`.Microgrid.run`.

    """
    _nested_action_space = None

    def _get_nested_action_space(self):
        return Dict({name: Tuple([module.action_space['normalized'] for module in modules_list])
                                 for name, modules_list in self.fixed.iterdict() if modules_list[0].is_source})

    def _get_action_space(self):
        self._nested_action_space = self._get_nested_action_space()
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
