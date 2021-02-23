from pymgrid.Environments.Environment import Environment
import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Space, Discrete, Box


class MicroGridEnv(Environment):
    """
    Markov Decision Process associated to the microgrid.

        Parameters
        ----------
            microgrid: microgrid, mandatory
                The controlled microgrid.
            random_seed: int, optional
                Seed to be used to generate the needed random numbers to size microgrids.

    """
    def __init__(self, env_config, seed=42):
        super().__init__(env_config, seed)
        self.Na = 4 + self.mg.architecture['grid'] * 3 + self.mg.architecture['genset'] * 1

        action_limits = [int(self.mg._pv_ts.max().values[0]),
                         int(self.mg.parameters['battery_power_charge'].values[0]),
                         int(self.mg.parameters['battery_power_discharge'].values[0]),
                         2,
                         ]
        if self.mg.architecture['genset'] ==1:
            action_limits.append(int(self.mg.parameters['genset_rated_power'].values[0]* self.mg.parameters['genset_pmax'].values[0]))

        if self.mg.architecture['grid'] == 1:
            action_limits.append(int(self.mg.parameters['grid_power_import'].values[0]))
            action_limits.append(int(self.mg.parameters['grid_power_export'].values[0]))
            action_limits.append(2)

        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(x) for x in action_limits])


    def get_action(self, action):
        return self.get_action_continuous(action)