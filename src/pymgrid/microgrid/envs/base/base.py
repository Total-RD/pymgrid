from gym import Env
from gym.spaces import Box, Dict, Tuple
from src.pymgrid.microgrid.modular_microgrid.modular_microgrid import ModularMicrogrid
from abc import abstractmethod


class BaseMicrogridEnv(ModularMicrogrid, Env):
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
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        print('here')

    @abstractmethod
    def _get_action_space(self):
        pass

    def _get_observation_space(self):
        return Dict({name:
                         Tuple([module.observation_spaces['normalized'] for module in modules_list]) for
                     name, modules_list in self.modules.iterdict()})

    def step(self, action):
        return self.run(action)

    def reset(self):
        return super().reset()

    @classmethod
    def from_microgrid(cls, microgrid):
        return cls(microgrid.modules_list, add_unbalanced_module=False)

    @classmethod
    def from_nonmodular(cls, nonmodular):
        microgrid = super().from_nonmodular(nonmodular)
        return cls.from_microgrid(microgrid)
