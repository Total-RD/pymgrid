from gym import Env
from gym.spaces import Dict, Tuple, flatten_space, flatten
from abc import abstractmethod

from pymgrid import NonModularMicrogrid, Microgrid
from pymgrid.microgrid.envs.base.skip_init import skip_init


class BaseMicrogridEnv(Microgrid, Env):
    def __new__(cls, modules, *args, **kwargs):
        if isinstance(modules, (NonModularMicrogrid, Microgrid)):
            instance = cls.from_microgrid(modules)
            cls.__init__ = skip_init(cls, cls.__init__)
            return instance
        elif "scenario" in kwargs or "microgrid_number" in kwargs:
            scenario = kwargs.get("scenario", "pymgrid25")
            microgrid_number = kwargs.get("microgrid_number", 0)
            instance = cls.from_scenario(scenario=scenario, microgrid_number=microgrid_number)
            cls.__init__ = skip_init(cls, cls.__init__)
            return instance

        return super().__new__(cls)

    def __init__(self,
                 modules,
                 add_unbalanced_module=True,
                 loss_load_cost=10,
                 overgeneration_cost=2,
                 flat_spaces=True
                 ):

        super().__init__(modules,
                         add_unbalanced_module=add_unbalanced_module,
                         loss_load_cost=loss_load_cost,
                         overgeneration_cost=overgeneration_cost)

        self._flat_spaces = flat_spaces
        self.action_space = self._get_action_space()
        self.observation_space, self._nested_observation_space = self._get_observation_space()

    @abstractmethod
    def _get_action_space(self):
        pass

    def _get_observation_space(self):
        obs_space = Dict({name:
                         Tuple([module.observation_spaces['normalized'] for module in modules_list]) for
                     name, modules_list in self.modules.iterdict()})

        return (flatten_space(obs_space) if self._flat_spaces else obs_space), obs_space

    def step(self, action, normalized=True):
        obs, reward, done, info = self.run(action, normalized=normalized)
        if self._flat_spaces:
            obs = flatten(self._nested_observation_space, obs)
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        return flatten(self._nested_observation_space, obs) if self._flat_spaces else obs

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
