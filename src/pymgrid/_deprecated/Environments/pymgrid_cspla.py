"""
Copyright 2020 Total S.A
Authors:Gonzague Henri <gonzague.henri@total.com>
Permission to use, modify, and distribute this software is given under the
terms of the pymgrid License.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2020/10/21 07:43 $
Gonzague Henri
"""

"""
<pymgrid is a Python library to simulate microgrids>
Copyright (C) <2020> <Total S.A.>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
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
        self.Na = 2 + self.mg.architecture['grid'] * 3 + self.mg.architecture['genset'] * 1 
        if self.mg.architecture['grid'] == 1 and self.mg.architecture['genset'] == 1:
            self.Na += 1
        self.action_space = Discrete(self.Na)



    def get_action(self, action):
        return self.get_action_priority_list(action)

    


