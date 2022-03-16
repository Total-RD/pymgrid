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

import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Space, Discrete, Box
from . import Preprocessing
from pymgrid.algos.Control import SampleAverageApproximation

DEFAULT_CONFIG={
    'microgrid': None, #need to be passed by user
    'training_reward_smoothing':'sqrt', #'peak_load'
    'resampling_on_reset':True,
    'forecast_args':None, #used to init the SAA for resampling on reset
    'baseline_sampling_args':None,
}

def generate_sampler(microgrid, forecast_args):
    """
    Generates an instance of SampleAverageApproximate to use in future sampling.
    :param microgrid:
    :param forecast_args:
    :return:
    """
    if forecast_args is None:
        forecast_args = dict()

    return SampleAverageApproximation(microgrid, **forecast_args)

class Environment(gym.Env):
    """
    Markov Decision Process associated to the microgrid.
        Parameters
        ----------
            microgrid: microgrid, mandatory
                The controlled microgrid.
            random_seed: int, optional
                Seed to be used to generate the needed random numbers to size microgrids.
    """

    def __init__(self, env_config, seed = 42):
        # Set seed
        np.random.seed(seed)

        self.states_normalization = Preprocessing.normalize_environment_states(env_config['microgrid'])

        self.TRAIN = True
        # Microgrid
        self.env_config = env_config
        self.mg = env_config['microgrid']
        # State space
        self.mg.train_test_split()
        #np.zeros(2+self.mg.architecture['grid']*3+self.mg.architecture['genset']*1)
        # Number of states
        self.Ns = len(self.mg._df_record_state.keys())+1
        # Number of actions

        #training_reward_smoothing
        try:
            self.training_reward_smoothing = env_config['training_reward_smoothing']
        except:
            self.training_reward_smoothing = 'sqrt'

        try:
            self.resampling_on_reset = env_config['resampling_on_reset']
        except:
            self.resampling_on_reset = False
        
        if self.resampling_on_reset == True:
            self.forecast_args = env_config['forecast_args']
            self.baseline_sampling_args = env_config['baseline_sampling_args']
            self.saa = generate_sampler(self.mg, self.forecast_args)
        
        self.observation_space = Box(low=-1, high=np.float('inf'), shape=(self.Ns,), dtype=np.float)
        #np.zeros(len(self.mg._df_record_state.keys()))
        # Action space
        self.metadata = {"render.modes": [ "human"]}
        
        self.state, self.reward, self.done, self.info, self.round = None, None, None, None, None
        self.round = None

        # Start the first round
        self.seed()
        self.reset()
        

        try:
            assert (self.observation_space.contains(self.state))
        except AssertionError:
            print("ERROR : INVALID STATE", self.state)

    def get_reward(self):
        if self.TRAIN == True:
            if self.training_reward_smoothing == 'sqrt':
                return -(self.mg.get_cost()**0.5)
            if self.training_reward_smoothing == 'peak_load':
                return -self.mg.get_cost()/self.mg.parameters['load'].values[0]
        return -self.mg.get_cost()

    def get_cost(self):
        return sum(self.mg._df_record_cost['cost'])



    def step(self, action):

        # CONTROL
        if self.done:
            print("WARNING : EPISODE DONE")  # should never reach this point
            return self.state, self.reward, self.done, self.info
        try:
            assert (self.observation_space.contains(self.state))
        except AssertionError:
            print("ERROR : INVALID STATE", self.state)

        try:
            assert (self.action_space.contains(action))
        except AssertionError:
            print("ERROR : INVALD ACTION", action)

        # UPDATE THE MICROGRID
        control_dict = self.get_action(action)
        self.mg.run(control_dict)

        # COMPUTE NEW STATE AND REWARD
        self.state = self.transition()
        self.reward = self.get_reward()
        self.done = self.mg.done
        self.info = {}
        self.round += 1

        return self.state, self.reward, self.done, self.info
        
#         control_dict = self.get_action(action)
#         self.mg.run(control_dict)
#         reward = self.reward()
#         s_ = self.transition()
#         self.state = s_
#         done = self.mg.done
#         self.round += 1
#         return s_, reward, done, {}

    def reset(self, testing=False):
        if "testing" in self.env_config:
            testing = self.env_config["testing"]
        self.round = 1
        # Reseting microgrid
        self.mg.reset(testing=testing)
        if testing == True:
            self.TRAIN = False
        elif self.resampling_on_reset == True:
            Preprocessing.sample_reset(self.mg.architecture['grid'] == 1, self.saa, self.mg, sampling_args=sampling_args)
        
        
        self.state, self.reward, self.done, self.info =  self.transition(), 0, False, {}
        
        return self.state


    def get_action(self, action):
        """
        :param action: current action
        :return: control_dict : dicco of controls
        """
        '''
        States are:
        binary variable whether charging or dischargin
        battery power, normalized to 1
        binary variable whether importing or exporting
        grid power, normalized to 1
        binary variable whether genset is on or off
        genset power, normalized to 1
        '''

        control_dict=[]

        return control_dict

    def states(self):  # soc, price, load, pv 'df status?'
        observation_space = []
        return observation_space

    # Transition function
    def transition(self):
        #         net_load = round(self.mg.load - self.mg.pv)
        #         soc = round(self.mg.battery.soc,1)
        #         s_ = (net_load, soc)  # next state
        updated_values = self.mg.get_updated_values()
        updated_values = {x:float(updated_values[x])/self.states_normalization[x] for x in self.states_normalization}  
        updated_values['hour_sin'] = np.sin(2*np.pi*updated_values['hour']) # the hour is already divided by 24 in the line above
        updated_values['hour_cos'] = np.cos(2*np.pi*updated_values['hour'])  
        updated_values.pop('hour', None)

        s_ = np.array(list(updated_values.values()))
        #np.array(self.mg.get_updated_values().values)#.astype(np.float)#self.mg.get_updated_values()
        #s_ = [ s_[key] for key in s_.keys()]
        return s_
    
    def seed (self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode="human"):
        txt = "state: " + str(self.state) + " reward: " + str(self.reward) + " info: " + str(self.info)
        print(txt)

    # Mapping between action and the control_dict
    def get_action_continuous(self, action):
        """
        :param action: current action
        :return: control_dict : dicco of controls
        """
        '''
        Actions are:
        binary variable whether charging or dischargin
        battery power, normalized to 1
        binary variable whether importing or exporting
        grid power, normalized to 1
        binary variable whether genset is on or off
        genset power, normalized to 1
        '''

        mg = self.mg
        pv = mg.pv
        load = mg.load
        net_load = load - pv
        capa_to_charge = mg.battery.capa_to_charge
        p_charge_max = mg.battery.p_charge_max
        p_charge = max(0, min(-net_load, capa_to_charge, p_charge_max))

        capa_to_discharge = mg.battery.capa_to_discharge
        p_discharge_max = mg.battery.p_discharge_max
        p_discharge = max(0, min(net_load, capa_to_discharge, p_discharge_max))

        control_dict = {}

        if mg.architecture['battery'] == 1:
            control_dict['battery_charge'] = max(0, action[0] * min(action[1] * mg.battery.capacity,
                                                                    mg.battery.capa_to_charge,
                                                                    mg.battery.p_charge_max))
            control_dict['battery_discharge'] = max(0, (1 - action[0]) * min(action[1] * mg.battery.capacity,
                                                                             mg.battery.capa_to_discharge,
                                                                             mg.battery.p_discharge_max))

        if mg.architecture['grid'] == 1:
            if mg.grid.status == 1:
                control_dict['grid_import'] = max(0, action[2] * min(action[3] * mg.grid.power_import,
                                                                     mg.grid.power_import))
                control_dict['grid_export'] = max(0, (1 - action[2]) * min(action[3] * mg.grid.power_export,
                                                                           mg.grid.power_export))
            else:
                # avoid warnings
                control_dict['grid_import'] = 0
                control_dict['grid_export'] = 0

        if mg.architecture['genset'] == 1:
            control_dict['genset'] = max(0, action[4] * min(action[5] * mg.genset.rated_power,
                                                            mg.genset.rated_power))
        return control_dict

    def get_action_discrete(self, action):
        """
        :param action: current action
        :return: control_dict : dicco of controls
        """
        '''
        Actions are:
        binary variable whether charging or dischargin
        battery power, normalized to 1
        binary variable whether importing or exporting
        grid power, normalized to 1
        binary variable whether genset is on or off
        genset power, normalized to 1
        '''
        control_dict={}

        control_dict['pv_consumed'] = action[0]
        if self.mg.architecture['battery'] == 1:
            control_dict['battery_charge'] = action[1] * action[3]
            control_dict['battery_discharge'] =  action[2] * (1- action[3])

        if self.mg.architecture['genset'] == 1:
            control_dict['genset'] = action[4]

            if self.mg.architecture['grid'] == 1:
                control_dict['grid_import'] = action[5] * action[7]
                control_dict['grid_export'] = action[6] * (1- action[7])

        elif self.mg.architecture['grid'] == 1:
            control_dict['grid_import'] = action[4] * action[6]
            control_dict['grid_export'] = action[5] * (1 - action[6])




        return control_dict

    # Mapping between action and the control_dict
    def get_action_priority_list(self, action):
        """
        :param action: current action
        :return: control_dict : dicco of controls
        """
        '''
        States are:
        binary variable whether charging or dischargin
        battery power, normalized to 1
        binary variable whether importing or exporting
        grid power, normalized to 1
        binary variable whether genset is on or off
        genset power, normalized to 1
        '''

        mg = self.mg
        pv = mg.pv
        load = mg.load
        net_load = load - pv
        capa_to_charge = mg.battery.capa_to_charge
        p_charge_max = mg.battery.p_charge_max
        p_charge = max(0, min(-net_load, capa_to_charge, p_charge_max))

        capa_to_discharge = mg.battery.capa_to_discharge
        p_discharge_max = mg.battery.p_discharge_max
        p_discharge = max(0, min(net_load, capa_to_discharge, p_discharge_max))

        control_dict = {}

        control_dict = self.actions_agent_discret(mg, action)

        return control_dict


    def actions_agent_discret(self, mg, action):
        if mg.architecture['genset'] == 1 and mg.architecture['grid'] == 1:
            control_dict = self.action_grid_genset(mg, action)

        elif mg.architecture['genset'] == 1 and mg.architecture['grid'] == 0:
            control_dict = self.action_genset(mg, action)

        else:
            control_dict = self.action_grid(mg, action)

        return control_dict

    def action_grid(self, mg, action):
        # slack is grid

        pv = mg.pv
        load = mg.load

        net_load = load - pv

        capa_to_charge = mg.battery.capa_to_charge
        p_charge_max = mg.battery.p_charge_max
        p_charge_pv = max(0, min(-net_load, capa_to_charge, p_charge_max))
        p_charge_grid = max(0, min( capa_to_charge, p_charge_max))

        capa_to_discharge = mg.battery.capa_to_discharge
        p_discharge_max = mg.battery.p_discharge_max
        p_discharge = max(0, min(net_load, capa_to_discharge, p_discharge_max))

        # Charge
        if action == 0:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': p_charge_pv,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': max(0, pv - min(pv, load) - p_charge_pv),
                            'genset': 0
                            }
        
        if action == 4:
            load = load + p_charge_grid
            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': p_charge_grid,
                            'battery_discharge': 0,
                            'grid_import': max(0, load - min(pv, load)),
                            'grid_export': max(0, pv - min(pv, load) - p_charge_grid) ,
                            'genset': 0
                            }


        # décharger full
        elif action == 1:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': p_discharge,
                            'grid_import': max(0, load - min(pv, load) - p_discharge),
                            'grid_export': 0,
                            'genset': 0
                            }

        # Import
        elif action == 2:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': 0,
                            'grid_import': max(0, net_load),
                            'grid_export': 0,
                            'genset': 0
                            }
        # Export
        elif action == 3:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': abs(min(net_load, 0)),
                            'genset': 0
                            }

        return control_dict

    def action_grid_genset(self, mg, action):
        # slack is grid

        pv = mg.pv
        load = mg.load

        net_load = load - pv
        status = mg.grid.status  # whether there is an outage or not
        capa_to_charge = mg.battery.capa_to_charge
        p_charge_max = mg.battery.p_charge_max
        p_charge_pv = max(0, min(-net_load, capa_to_charge, p_charge_max))
        p_charge_grid = max(0, min( capa_to_charge, p_charge_max))

        capa_to_discharge = mg.battery.capa_to_discharge
        p_discharge_max = mg.battery.p_discharge_max
        p_discharge = max(0, min(net_load, capa_to_discharge, p_discharge_max))

        capa_to_genset = mg.genset.rated_power * mg.genset.p_max
        p_genset = max(0, min(net_load, capa_to_genset))

        # Charge
        if action == 0:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': p_charge_pv,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': max(0, pv - min(pv, load) - p_charge_pv) * status,
                            'genset': 0
                            }
        if action == 5:
            load = load+p_charge_grid

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': p_charge_grid,
                            'battery_discharge': 0,
                            'grid_import': max(0, load - min(pv, load)) * status,
                            'grid_export': max(0, pv - min(pv, load) - p_charge_grid) * status,
                            'genset': 0
                            }


        # décharger full
        elif action == 1:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': p_discharge,
                            'grid_import': max(0, load - min(pv, load) - p_discharge) * status,
                            'grid_export': 0,
                            'genset': 0
                            }

        # Import
        elif action == 2:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': 0,
                            'grid_import': max(0, net_load) * status,
                            'grid_export': 0,
                            'genset': 0
                            }
        # Export
        elif action == 3:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': abs(min(net_load, 0)) * status,
                            'genset': 0
                            }
        # Genset
        elif action == 4:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': 0,
                            'genset': max(net_load, 0)
                            }

        elif action == 6:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': p_discharge,
                            'grid_import': 0,
                            'grid_export': 0,
                            'genset': max(0, load - min(pv, load) - p_discharge),
                            }

        return control_dict

    def action_genset(self, mg, action):
        # slack is genset

        pv = mg.pv
        load = mg.load

        net_load = load - pv

        capa_to_charge = mg.battery.capa_to_charge
        p_charge_max = mg.battery.p_charge_max
        p_charge = max(0, min(-net_load, capa_to_charge, p_charge_max))

        capa_to_discharge = mg.battery.capa_to_discharge
        p_discharge_max = mg.battery.p_discharge_max
        p_discharge = max(0, min(net_load, capa_to_discharge, p_discharge_max))

        capa_to_genset = mg.genset.rated_power * mg.genset.p_max
        p_genset = max(0, min(net_load, capa_to_genset))

        # Charge
        if action == 0:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': p_charge,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': 0,
                            'genset': 0
                            }


        # décharger full
        elif action == 1:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': p_discharge,
                            'grid_import': 0,
                            'grid_export': 0,
                            'genset': max(0, load - min(pv, load) - p_discharge)
                            }

        # Genset
        elif action == 2:

            control_dict = {'pv_consummed': min(pv, load),
                            'battery_charge': 0,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': 0,
                            'genset': max(0, load - min(pv, load))
                            }

        return control_dict
