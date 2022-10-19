from gym import Env
from gym.spaces import Box, Dict, Tuple
from abc import abstractmethod

from pymgrid import Microgrid, ModularMicrogrid
from pymgrid.microgrid.envs.base.skip_init import skip_init


class BaseMicrogridEnv(ModularMicrogrid, Env):
    def __new__(cls, modules, *args, **kwargs):
        if isinstance(modules, (Microgrid, ModularMicrogrid)):
            instance = cls.from_microgrid(modules)
            cls.__init__ = skip_init(cls, cls.__init__)
            return instance

        return super().__new__(cls)


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
        try:
            return cls(microgrid.module_tuples(), add_unbalanced_module=False)
        except AttributeError:
            return cls.from_nonmodular(microgrid)

    @classmethod
    def from_nonmodular(cls, nonmodular):
        microgrid = super().from_nonmodular(nonmodular)
        return cls.from_microgrid(microgrid)
