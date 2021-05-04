"""
Copyright 2020 Total S.A., Tanguy Levent all rights reserved,
Authors:Gonzague Henri <gonzague.henri@total.com>, Tanguy Levent <>
Permission to use, modify, and distribute this software is given under the
terms of the pymgrid License.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2020/06/04 14:54 $
Gonzague Henri
"""

"""
<pymgrid is a Python library to simulate microgrids>
Copyright (C) <2020> <Total S.A.>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

"""

import pandas as pd
import numpy as np
from copy import copy
import cvxpy as cp
import operator
import sys
from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import cufflinks as cf
from IPython.display import display
from IPython import get_ipython
from pymgrid.algos.Control import Benchmarks

def in_ipynb():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except (NameError, AttributeError):
        return False

if in_ipynb():
    init_notebook_mode(connected=False)

np.random.seed(123)
#cf.set_config_file(offline=True, theme='pearl') #commented for now, issues with parallel processes

DEFAULT_HORIZON = 24 #in hours
DEFAULT_TIMESTEP = 1 #in hours
ZERO = 10**-5

'''
The following classes are used to contain the information related to the different components
of the microgrid. Their main use is for easy access in a notebook.
'''

class Battery:

    """
    The class battery is used to store the information related to the battery in a microgrid. One of the main use for
    this class is for an easy access to information in a notebook using the battery object contained in a microgrid.

    Parameters
    ----------
    param_battery : dataframe
        All the data to initialize the battery.
    capa_to_charge : float
        Represents the amount of energy that a battery can charge before being full.
    capa_to_discharge : float
        Represents the amount of energy available that a battery can discharge before being empty.

    Attributes
    ----------
    soc: float
        Value between 0 and 1 representing the state of charge of the battery (1 being full, 0 being empty)
    capacity: int
        Total energy capacity of the battery (kWh).
    soc_max: float
        Value representing the maximum SOC that a battery can reach
    soc_min: float
        Value representing the minimum SOC that a battery can reach
    p_charge_max: float
        Value representing the maximum charging rate of the battery (kW)
    p_discharge_max: float
        Value representing the maximum discharging rate of the battery (kW)
    efficiency: float
        Value between 0 and 1 representing a one-way efficiency of the battery (considering same efficiency for charging
        and discharging).
    cost_cycle: float
        Value representing the cost of using the battery in $/kWh.
    capa_to_charge : float
        Represents the amount of energy that a battery can charge before being full.
    capa_to_discharge : float
        Represents the amount of energy available that a battery can discharge before being empty.


    Notes
    -----
    Another way to use this information in a notebook is to use /tab/ after /microgrid.battery./ so you can see all the
    battery attributes.

    Examples
    --------
    >>> m_gen=mg.MicrogridGenerator(nb_microgrid=1,path='your_path')
    >>> m_gen.generate_microgrid()
    >>> m_gen.microgrids[0].battery
    You can then add a point and use tab to have suggestion of the different paramterers
    You can access state of charge for example with:
    >>> m_gen.microgrids[0].battery.soc
    """

    def __init__(self, param_battery, capa_to_charge, capa_to_discharge):

        self.soc = param_battery['battery_soc_0'].values[0]
        self.capacity = param_battery['battery_capacity'].values[0]
        self.soc_max = param_battery['battery_soc_max'].values[0]
        self.soc_min = param_battery['battery_soc_min'].values[0]
        self.p_charge_max = param_battery['battery_power_charge'].values[0]
        self.p_discharge_max = param_battery['battery_power_discharge'].values[0]
        self.efficiency = param_battery['battery_efficiency'].values[0]
        self.cost_cycle = param_battery['battery_cost_cycle'].values[0]
        self.capa_to_charge = capa_to_charge
        self.capa_to_discharge = capa_to_discharge



class Genset:
    """
    The class Genset is used to store the information related to the genset in a microgrid. One of the main use for
    this class is for an easy access to information in a notebook using the genset object contained in a microgrid.

    Parameters
    ----------
    param : dataframe
        All the data to initialize the genset.

    Attributes
    ----------
    rated_power: int
        Maximum rater power of the genset.
    p_min: float
        Value representing the minimum operating power of the genset (kW)
    p_max: float
        Value representing the maximum operating power of the genset (kW)
    fuel_cost: float
        Value representing the cost of using the genset in $/kWh.

    Notes
    -----
    Another way to use this information in a notebook is to use /tab/ after /microgrid.genset./ so you can see all the
    genset attributes.

    Examples
    --------
    >>> m_gen=mg.MicrogridGenerator(nb_microgrid=1,path='your_path')
    >>> m_gen.generate_microgrid()
    >>> m_gen.microgrids[0].genset
    You can then add a point and use tab to have suggestion of the different paramaterers
    You can access the maximum power max for example with:
    >>> m_gen.microgrids[0].genset.p_max
    """
    def __init__(self, param):
        self.rated_power = param['genset_rated_power'].values[0]
        self.p_min = param['genset_pmin'].values[0]
        self.p_max = param['genset_pmax'].values[0]
        self.fuel_cost = param['fuel_cost'].values[0]
        self.co2 = param['genset_co2'].values[0]


class Grid:
    """
    The class Grid is used to store the information related to the grid in a microgrid. One of the main use for
    this class is for an easy access to information in a notebook using the grid object contained in a microgrid.

    Parameters
    ----------
    param : dataframe
        All the data to initialize the grid.
    status: int
        Whether the grid is connected or not at the first time step.


    Attributes
    ----------
    power_export: float
        Value representing the maximum export power to the grid (kW)
    power_import: float
        Value representing the maximum import power from the grid (kW)
    price_export: float
        Value representing the cost of exporting to the grid in $/kWh.
    price_import: float
        Value representing the cost of importing to the grid in $/kWh.
    status: int, binary
        Binary value representing whether the grid is connected or not (for example 0 represent a black-out of the
        main grid).

    Notes
    -----
    Another way to use this information in a notebook is to use /tab/ after /microgrid.grid./ so you can see all the
    grid attributes.

    Examples
    --------
    >>> m_gen=mg.MicrogridGenerator(nb_microgrid=1,path='your_path')
    >>> m_gen.generate_microgrid()
    >>> m_gen.microgrids[0].grid
    You can then add a point and use tab to have suggestion of the different paramaterers
    You can access the status of the grid for example with:
    >>> m_gen.microgrids[0].grid.status
    """
    def __init__(self, param, status, price_import, price_export, co2):
        self.power_export = param['grid_power_export'].values[0]
        self.power_import = param['grid_power_import'].values[0]
        self.price_export = price_export #param['grid_price_export'].values[0]
        self.price_import = price_import # param['grid_price_import'].values[0]
        self.status = status
        self.co2 = co2


class Microgrid:

    """
    The class microgrid implement a microgrid. It is also used to run the simulation and different benchmarks.

    Parameters
    ----------
    parameters : dataframe
        In parameters we find:
            -'parameters': a dataframe containing all the fixed (not changing with time ) parameters of the microgrid
            -'architecture': a dictionnary containing a binary variable for each possible generator and indicating if
             this microgrid has one of them
            -'load': the load time series
            -'pv': the pv time series
            -'grid_ts': a time series of 1 and 0 indicating whether the grid is available
            -'df_actions': an empty dataframe representing the actions that the microgrid can take
            -'df_status': a dataframe representing the parameters of the microgrid that change with time, with the
             information for the first time step
            -'df_actual_generation': an empty dataframe that is used to store what actually happens in the microgrid
             after control actions are taken
            -'df_cost': dataframe to track the cost of operating the microgrid at each time step
            -'control_dict': an example of the control dictionnary that needs to be passed in run to operate the
             microgrid
    horizon : int, optional
        The horizon considered to control the microgrid, mainly used in the MPC function and to return the forecasting
         values (in hour).
    timestep : int, optional
        Time step the microgrid is operating at (in hour).

    Attributes
    ----------
        parameters: dataframe
            A dataframe containing all the fixed (not changing with time ) parameters of the microgrid
        architecture : dictionary
            A dictionary containing a binary variable for each possible generator and indicating if
            this microgrid has one of them
        _load_ts: dataframe
            The time series of load
        _pv_ts: dataframe
            Time series of PV generation
        pv: float
            The PV production at _run_timestep
        load: float
            The load consumption at _run_timestep
        _next_pv: float
            The PV production at _run_timestep +1
        _next_load: float
            The load consumption at _run_timestep + 1
        _grid_status_ts: dataframe
            A timeseries of binary values indicating whether the grid is connected or not.
        _df_record_control_dict: dataframe
            This dataframe is used to record the control actions taked at each time step.
        _df_record_state : dataframe
            This dataframe is used to record the variable parameters of the microgrid at each time step.
        _df_record_actual_production : dataframe
            This dataframe is used to record the actual generation of the microgrid at each time step.
        _df_record_cost : dataframe
            This dataframe is used to record the cost of operating the microgrid at each time step.
        _df_cost_per_epochs  : dataframe
            In the case we run the simulation through multiple epochs, this dataframe is used to record the cost of
             operating the microgrid at each time step of each epoch.
        horizon : int, optional
            The horizon considered to control the microgrid, mainly used in the MPC function and to return the forecasting
             values (in hour).
        _run_timestep : int
            Time step the microgrid is operating at (in hour).
        _data_length: int
            Represents the number of time steps in PV/Load files (minimum between the 2) that will be used to run the
            simulation.
        done: True or False
            Indicates whether a simulation is done or not
        _has_run_rule_based_baseline: True or False
            Indicates whether the rule based benchmark has already been run or not.
        _has_run_mpc_baseline: True or False
            Indicates whether the MPC benchmark has already been run or not.
        _epoch: int
            Tracks what epoch the microgrid is at
        _zero: float
            Approximate value to 0, used in some comparisons
        control_dict: dictionnary
            Represents the list of control actions to pass in the run function
        battery: object
            Represents all the parameter of the battery, including the value changing with time (in this case it is the
            value at _run_timestep).
        genset: object
            Represents all the parameter of the genset, including the value changing with time (in this case it is the
            value at _run_timestep).
        grid: object
            Represents all the parameter of the grid, including the value changing with time (in this case it is the
            value at _run_timestep).
        benchmarks: algos.Control.Benchmarks
            Benchmark object with the ability to run benchmark algorithms and store/print the results.

    Notes
    -----
    We are trying to keep hidden a lot of what is happening under the hood to simplify using this class for control or
    RL research at the maximum. A few notes, in this class parameters refer to the fixed parameters of the microgrid,
    meaning they don't vary with time. The varying parameters can be found in either the other classes or
    _df_record_state.

    Examples
    --------
    To create microgrids through MicrogridGenerator:
    >>> m_gen=mg.MicrogridGenerator(nb_microgrid=1,path='your_path')
    >>> m_gen.generate_microgrid()
    >>>microgrid = m_gen.microgrid[0]

    To plot informations about the microgrid:
    >>> microgrid.print_info()
    >>> microgrid.print_control_info()

    To compute the benchmarks:
    >>> microgrid.compute_benchmark() # to compute them all
    >>> microgrid.compute_benchmark('mpc_linprog') #to compute only the MPC

    For example, a simple control loop:
    >>> while m_gen.microgrids[0].done == False:
    >>>     load = mg_data['load']
    >>>     pv = mg_data['pv']
    >>>     control_dict = {'battery_charge': 0, 'battery_discharge': 0,'grid_import': max(0, load-pv),'grid_export':0,'pv': min(pv, load),}
    >>>     mg_data = m_gen.microgrids[0].run(control_dict)
    """

    def __init__(self, parameters, horizon=DEFAULT_HORIZON, timestep=DEFAULT_TIMESTEP):

        #list of parameters
        #this is a static dataframe: parameters of the microgrid that do not change with time

        #self._param_check(parameters)

        self.parameters = parameters['parameters']
        self.architecture =  parameters['architecture']

        #different timeseries
        self._load_ts=parameters['load']
        self._pv_ts=parameters['pv']

        self.pv = self._pv_ts.iloc[0,0]
        self.load = self._load_ts.iloc[0, 0]
        self._next_load = self._load_ts.iloc[1,0]
        self._next_pv = self._pv_ts.iloc[1,0]
        if parameters['architecture']['grid']==1:
            self._grid_status_ts=parameters['grid_ts'] #time series of outages
            #self.grid_status = self._grid_status_ts.iloc[0, 0]
            self._grid_price_import=parameters['grid_price_import']
            self._grid_price_export=parameters['grid_price_export']
            self._grid_co2 = parameters['grid_co2']

            self._next_grid_status = self._grid_status_ts.iloc[0, 0]
            self._next_grid_price_export = self._grid_price_export.iloc[0, 0]
            self._next_grid_price_import = self._grid_price_import.iloc[0, 0]
            self._next_grid_co2 = self._grid_co2.iloc[0, 0]

        # those dataframe record what is happening at each time step
        self._df_record_control_dict=parameters['df_actions']
        self._df_record_state = parameters['df_status']
        self._df_record_actual_production = parameters['df_actual_generation']
        self._df_record_cost = parameters['df_cost']
        self._df_record_co2 = parameters['df_co2']
        self._df_cost_per_epochs = []
        self.horizon = horizon
        self._tracking_timestep = 0
        self._data_length = min(self._load_ts.shape[0], self._pv_ts.shape[0])
        self.done = False
        self._has_run_rule_based_baseline = False
        self._has_run_mpc_baseline = False
        self._has_train_test_split = False
        self._epoch=0
        self._zero = ZERO
        self.control_dict = parameters['control_dict']
        self._data_set_to_use_default = 'all'
        self._data_set_to_use = 'all'

        self.benchmarks = Benchmarks(self)

        if self.architecture['battery'] == 1:
            self.battery = Battery(self.parameters,
                                   self._df_record_state['capa_to_charge'][0],
                                   self._df_record_state['capa_to_discharge'][0])
        if self.architecture['genset'] == 1:
            self.genset = Genset(self.parameters)
        if self.architecture['grid'] == 1:
            self.grid = Grid(self.parameters, self._grid_status_ts.iloc[0,0],
                             self._grid_price_import.iloc[0, 0],
                             self._grid_price_export.iloc[0, 0],
                             self._grid_co2.iloc[0, 0])

    def _param_check(self,parameters):
        """Simple parameter checks"""

        # Check parameters
        if not isinstance(parameters, dict):
            raise TypeError('parameters must be of type dict, is ({})'.format(type(parameters)))


        # Check architecture
        try:
            architecture = parameters['architecture']
        except KeyError:
            print('Dict of parameters does not appear to contain architecture key')
            raise
        if not isinstance(architecture, dict):
            raise TypeError('parameters[\'architecture\'] must be of type dict, is ({})'.format(type(architecture)))

        for key, val in architecture.items():
            if isinstance(val,bool):
                continue
            elif isinstance(val,int) and (val == 0 or val == 1):
                continue
            else:
                raise TypeError('Value ({}) of key ({}) in architecture is of unrecognizable type, '
                                'must be bool or in {{0,1}}, is ({})'.format(val, key, type(val)))

        # Ensure various DataFrames exist and are in fact DataFrames

        keys = ('parameters', 'load', 'pv', 'df_actions', 'df_status', 'df_actual_generation', 'df_cost')

        for key in keys:
            try:
                df = parameters[key]
            except KeyError:
                print('Dict of parameters does not appear to contain {} key'.format(key))
                raise
            if not isinstance(df, pd.DataFrame):
                raise TypeError('parameters[\'{}\'] must be of type pd.DataFrame, is ({})'.format(key, type(df)))



    def set_horizon(self, horizon):
        """Function used to change the horizon of the simulation."""
        self.horizon = horizon

    def set_cost_co2(self, co2_cost):
        """Function used to change the horizon of the simulation."""
        self.parameters['cost_co2'] = co2_cost

    def get_data(self):
        """Function to return the time series used in the microgrid"""
        return self._load_ts, self._pv_ts

    def get_training_testing_data(self):

        if self._has_train_test_split == True:

            return self._limit_index, self._load_train, self._pv_train, self._load_test, self._pv_test

        else:
            print('You have not split the dataset into training and testing sets')

    def get_control_dict(self):
        """ Function that returns the control_dict. """
        return self.control_dict


    def get_parameters(self):
        """ Function that returns the parameters of the microgrid. """
        return self.parameters


    def get_cost(self):
        """ Function that returns the cost associated the operation of the last time step. """
        return self._df_record_cost['cost'][-1]

    def get_co2(self):
        """ Function that returns the co2 emissions associated to the operation of the last time step. """
        return self._df_record_co2['co2'][-1]

    def get_updated_values(self):
        """
        Function that returns microgrid parameters that change with time. Depending on the architecture we have:
            - PV production
            - Load
            - Battery state of charge
            - Battery capacity to charge
            - Battery capacity to discharge
            - Whether the grid is connected or not
            - CO2 intensity of the grid
        """
        mg_data = {}

        for i in self._df_record_state:
            mg_data[i] = self._df_record_state[i][-1]

        return mg_data


    def forecast_all(self):
        """ Function that returns the PV, load and grid_status forecasted values for the next horizon. """
        forecast = {
            'pv': self.forecast_pv(),
            'load': self.forecast_load(),
        }
        if self.architecture['grid'] == 1:
            forecast['grid_status'] = self.forecast_grid_status()
            forecast['grid_import'], forecast['grid_export'] = self.forecast_grid_prices()
            forecast['grid_co2'] = self.forecast_grid_co2()

        return forecast


    def forecast_pv(self):
        """ Function that returns the PV forecasted values for the next horizon. """
        forecast = np.nan
        if self._data_set_to_use == 'training':
            forecast=self._pv_train.iloc[self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        if self._data_set_to_use == 'testing':
            forecast = self._pv_test.iloc[
                       self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        if self._data_set_to_use == 'all':
            forecast = self._pv_ts.iloc[self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        return forecast


    def forecast_load(self):
        """ Function that returns the load forecasted values for the next horizon. """
        forecast = np.nan
        if self._data_set_to_use == 'training':
            forecast = self._load_train.iloc[self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        if self._data_set_to_use == 'testing':
            forecast = self._load_test.iloc[self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        if self._data_set_to_use == 'all':
            forecast = self._load_ts.iloc[self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        return forecast

    def forecast_grid_status(self):
        """ Function that returns the grid_status forecasted values for the next horizon. """
        forecast = np.nan
        if self._data_set_to_use == 'training':
            forecast = self._grid_status_train.iloc[
               self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        if self._data_set_to_use == 'testing':
            forecast = self._grid_status_test.iloc[
               self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        if self._data_set_to_use == 'all':
            forecast = self._grid_status_ts.iloc[
               self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        return forecast

    def forecast_grid_co2(self):
        """ Function that returns the grid_status forecasted values for the next horizon. """
        forecast = np.nan
        if self._data_set_to_use == 'training':
            forecast = self._grid_co2_train.iloc[
                       self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        if self._data_set_to_use == 'testing':
            forecast = self._grid_co2_test.iloc[
                       self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        if self._data_set_to_use == 'all':
            forecast = self._grid_co2.iloc[
                       self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        return forecast

    def forecast_grid_prices(self):
        """ Function that returns the forecasted import and export prices for the next horizon. """
        forecast_import = np.nan
        forecast_export = np.nan
        if self._data_set_to_use == 'training':
            forecast_import = self._grid_price_import_train.iloc[
                       self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()
            forecast_export = self._grid_price_export_train.iloc[
                              self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        if self._data_set_to_use == 'testing':
            forecast_import = self._grid_price_import_test.iloc[
                       self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()
            forecast_export = self._grid_price_export_test.iloc[
                       self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        if self._data_set_to_use == 'all':
            forecast_import = self._grid_price_import.iloc[
                       self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()
            forecast_export = self._grid_price_export.iloc[
                       self._tracking_timestep:self._tracking_timestep + self.horizon].values.flatten()

        return forecast_import, forecast_export



    #if return whole pv and load ts, the time can be counted in notebook
    def run(self, control_dict):
        """
        Function to 'run' the microgrid and iterate over the dataset.

        Parameters
        ----------
        control_dict : dictionnary
            Dictionnary containing the different control actions we want to apply to the microgrid. Its fields depend
            on the architecture of the microgrid

        Return
        ----------
        self.get_updated_values(): dictionnary
            Return all the parameters that change with time in the microgrid. CF this function for more details.

        Notes
        ----------
        This loop is the main connexion with a user in a notebook. That is where the simulation is ran and where the
        control actions are recorder and applied.

        """

        control_dict['load'] = self.load
        control_dict['pv'] = self.pv

        self._df_record_control_dict = self._record_action(control_dict, self._df_record_control_dict)



        self._df_record_actual_production = self._record_production(control_dict,
                                                                         self._df_record_actual_production,
                                                                    self._df_record_state)

        if self.architecture['grid'] == 1:
            self._df_record_co2 = self._record_co2({ i:self._df_record_actual_production[i][-1] for i in self._df_record_actual_production},
                                                   self._df_record_co2, self.grid.co2)

            self._df_record_cost = self._record_cost({ i:self._df_record_actual_production[i][-1] for i in self._df_record_actual_production},
                                                               self._df_record_cost, self._df_record_co2, self.grid.price_import, self.grid.price_export)
            self._df_record_state = self._update_status(control_dict,
                                                        self._df_record_state, self._next_load, self._next_pv,
                                                        self._next_grid_status, self._next_grid_price_import,
                                                        self._next_grid_price_export, self._next_grid_co2)


        else:
            self._df_record_co2 = self._record_co2({ i:self._df_record_actual_production[i][-1] for i in self._df_record_actual_production},
                                                   self._df_record_co2)

            self._df_record_cost = self._record_cost({ i:self._df_record_actual_production[i][-1] for i in self._df_record_actual_production},
                                                     self._df_record_cost, self._df_record_co2)
            self._df_record_state = self._update_status(control_dict,
                                                        self._df_record_state, self._next_load, self._next_pv)

        if self._tracking_timestep == self._data_length - self.horizon or self._tracking_timestep == self._data_length - 1:
            self.done = True
            return self.get_updated_values()

        self._tracking_timestep += 1
        self.update_variables()

        return self.get_updated_values()


    def train_test_split(self, train_size=0.67, shuffle = False, cancel=False):
        """
        Function to split our data between a training and testing set.

        Parameters
        ----------
        train_size : float, optional
            Value between 0 and 1 reflecting the percentage of the dataset that should be in the training set.
        shuffle: boolean
            Variable to know if the training and testing sets should be shuffled or in the 'temporal' order
            Not implemented yet for shuffle = True
        cancel: boolean
            Variable indicating if the split needs to be reverted, and the data brought back into one dataset

        Attributes
        ----------
        _limit_index : int
            Index that delimit the training and testing sets in the time series
        load_train : dataframe
            Timeseries of load in training set
        pv_train: dataframe
            Timeseries of PV in training set
        load_test : dataframe
            Timeseries of load in testing set
        pv_test: dataframe
            Timeseries of PV in testing set
        grid_status_train: dataframe
            Timeseries of grid_status in training set
        grid_status_test: dataframe
            Timeseries of grid_status in testing set
        grid_price_import_train: dataframe
            Timeseries of price_import in training set
        grid_price_import_test: dataframe
            Timeseries of price_import in testing set
        grid_price_export_train: dataframe
            Timeseries of price_export in training set
        grid_price_export_test: dataframe
            Timeseries of price_export in testing set

        """

        if self._has_train_test_split ==  False:
            self._limit_index = int(np.ceil(self._data_length*train_size))
            self._load_train = self._load_ts.iloc[:self._limit_index]
            self._pv_train = self._pv_ts.iloc[:self._limit_index]

            self._load_test = self._load_ts.iloc[self._limit_index:]
            self._pv_test = self._pv_ts.iloc[self._limit_index:]

            if self.architecture['grid'] == 1:
                self._grid_status_train = self._grid_status_ts.iloc[:self._limit_index]
                self._grid_status_test = self._grid_status_ts.iloc[self._limit_index:]

                self._grid_price_import_train = self._grid_price_import.iloc[:self._limit_index]
                self._grid_price_import_test = self._grid_price_import.iloc[self._limit_index:]

                self._grid_price_export_train = self._grid_price_export.iloc[:self._limit_index]
                self._grid_price_export_test = self._grid_price_export.iloc[self._limit_index:]

                self._grid_co2_train = self._grid_co2.iloc[:self._limit_index]
                self._grid_co2_test = self._grid_co2.iloc[self._limit_index:]

            self._has_train_test_split = True
            self._data_set_to_use_default = 'training'
            self._data_set_to_use = 'training'

        elif self._has_train_test_split ==  True and cancel == True:
            self._has_train_test_split = False
            self._data_set_to_use_default = 'all'
            self._data_set_to_use = 'all'

        self.reset()

    def update_variables(self):
        """ Function that updates the variablers containing the parameters of the microgrid changing with time. """

        if self._data_set_to_use == 'training':
            self.pv = self._pv_train.iloc[self._tracking_timestep, 0]
            self.load = self._load_train.iloc[self._tracking_timestep, 0]

            self._next_pv = self._pv_train.iloc[self._tracking_timestep +1, 0]
            self._next_load = self._load_train.iloc[self._tracking_timestep+1, 0]


        if self._data_set_to_use == 'testing':
            self.pv = self._pv_test.iloc[self._tracking_timestep, 0]
            self.load = self._load_test.iloc[self._tracking_timestep, 0]

            self._next_pv = self._pv_test.iloc[self._tracking_timestep+1, 0]
            self._next_load = self._load_test.iloc[self._tracking_timestep+1, 0]

        if self._data_set_to_use == 'all':
            self.pv = self._pv_ts.iloc[self._tracking_timestep, 0]
            self.load = self._load_ts.iloc[self._tracking_timestep, 0]


            if self._tracking_timestep < self._data_length - 1:
                self._next_pv = self._pv_ts.iloc[self._tracking_timestep+1, 0]
                self._next_load = self._load_ts.iloc[self._tracking_timestep+1, 0]
            else:
                self._next_pv, self._next_load = None, None


        if self.architecture['grid']==1:
            if self._data_set_to_use == 'training':
                self.grid.status = self._grid_status_train.iloc[self._tracking_timestep, 0]
                self.grid.price_import = self._grid_price_import_train.iloc[self._tracking_timestep,0]
                self.grid.price_export = self._grid_price_export_train.iloc[self._tracking_timestep,0]
                self.grid.co2 = self._grid_co2_train.iloc[self._tracking_timestep, 0]

                self._next_grid_status = self._grid_status_train.iloc[self._tracking_timestep +1, 0]
                self._next_grid_price_import = self._grid_price_import_train.iloc[self._tracking_timestep +1, 0]
                self._next_grid_price_export = self._grid_price_export_train.iloc[self._tracking_timestep +1, 0]
                self._next_grid_co2 = self._grid_co2_train.iloc[self._tracking_timestep + 1, 0]

            if self._data_set_to_use == 'testing':
                self.grid.status = self._grid_status_test.iloc[self._tracking_timestep, 0]
                self.grid.price_import = self._grid_price_import_test.iloc[self._tracking_timestep, 0]
                self.grid.price_export = self._grid_price_export_test.iloc[self._tracking_timestep, 0]
                self.grid.co2 = self._grid_co2_test.iloc[self._tracking_timestep, 0]

                self._next_grid_status = self._grid_status_test.iloc[self._tracking_timestep + 1, 0]
                self._next_grid_price_import = self._grid_price_import_test.iloc[self._tracking_timestep + 1, 0]
                self._next_grid_price_export = self._grid_price_export_test.iloc[self._tracking_timestep + 1, 0]
                self._next_grid_co2 = self._grid_co2_test.iloc[self._tracking_timestep + 1, 0]


            if self._data_set_to_use == 'all':
                self.grid.status = self._grid_status_ts.iloc[self._tracking_timestep, 0]
                self.grid.price_import = self._grid_price_import.iloc[self._tracking_timestep, 0]
                self.grid.price_export = self._grid_price_export.iloc[self._tracking_timestep, 0]
                self.grid.co2 = self._grid_co2.iloc[self._tracking_timestep, 0]

                if self._tracking_timestep < self._data_length - 1:
                    self._next_grid_status = self._grid_status_ts.iloc[self._tracking_timestep + 1, 0]
                    self._next_grid_price_import = self._grid_price_import.iloc[self._tracking_timestep + 1, 0]
                    self._next_grid_price_export = self._grid_price_export.iloc[self._tracking_timestep + 1, 0]
                    self._next_grid_co2 = self._grid_co2.iloc[self._tracking_timestep + 1, 0]
                else:
                    self._next_grid_status, self._next_grid_price_import, self._next_grid_price_export, \
                    self._next_grid_co2 = None, None, None, None

        if self.architecture['battery'] == 1:
            self.battery.soc = self._df_record_state['battery_soc'][-1]
            self.battery.capa_to_discharge = self._df_record_state['capa_to_discharge'][-1]
            self.battery.capa_to_charge = self._df_record_state['capa_to_charge'][-1]

    def reset(self, testing=False):
        """This function is used to reset the dataframes that track what is happening in simulation. Mainly used in RL."""
        if self._data_set_to_use == 'training':
            temp_cost = copy(self._df_record_cost)
            temp_cost['epoch'] = self._epoch
            self._df_cost_per_epochs.append(temp_cost)

        self._df_record_control_dict = {i:[] for i in self._df_record_control_dict}
        self._df_record_state = {i:[self._df_record_state[i][0]] for i in self._df_record_state}
        self._df_record_actual_production = {i:[] for i in self._df_record_actual_production}
        self._df_record_cost = {i:[] for i in self._df_record_cost}
        self._df_record_co2 = {i:[] for i in self._df_record_co2}

        self._tracking_timestep = 0

        if testing == True and self._data_set_to_use_default == 'training':
            self._data_set_to_use = 'testing'
            self._data_length = min(self._load_test.shape[0], self._pv_test.shape[0])
        else:
            self._data_set_to_use = self._data_set_to_use_default
            if self._data_set_to_use == 'training':
                self._data_length = min(self._load_train.shape[0], self._pv_train.shape[0])
            else:
                self._data_length = min(self._load_ts.shape[0], self._pv_ts.shape[0])

        self.update_variables()
        self.done = False



        self._epoch+=1


    ########################################################
    # FUNCTIONS TO UPDATE THE INTERNAL DICTIONARIES
    ########################################################


    def _record_action(self, control_dict, df):
        """ This function is used to record the actions taken, before being checked for feasability. """
        if not isinstance(df, dict):
            raise TypeError('We know this should be named differently but df needs to be dict, is {}'.format(type(df)))
        for j in df:
            if j in control_dict.keys():
                df[j].append(control_dict[j])
            else:
                df[j].append({j:0})
        #df = df.append(control_dict,ignore_index=True)

        return df


    def _update_status(self, control_dict, df, next_load, next_pv, next_grid = 0, next_price_import =0, next_price_export = 0, next_co2 = 0):
        """ This function update the parameters of the microgrid that change with time. """
        #self.df_status = self.df_status.append(self.new_row, ignore_index=True)

        if not isinstance(df, dict):
            raise TypeError('We know this should be named differently but df needs to be dict, is {}'.format(type(df)))

        new_dict = {
            'load': next_load,
                    'pv': next_pv,
            'hour':self._tracking_timestep%24,
        }
        new_soc =np.nan
        if self.architecture['battery'] == 1:
            new_soc = df['battery_soc'][-1] + (control_dict['battery_charge']*self.parameters['battery_efficiency'].values[0]
                                                        - control_dict['battery_discharge']/self.parameters['battery_efficiency'].values[0])/self.parameters['battery_capacity'].values[0]
            #if col == 'net_load':
            capa_to_charge = max(
                (self.parameters['battery_soc_max'].values[0] * self.parameters['battery_capacity'].values[0] -
                 new_soc *
                 self.parameters['battery_capacity'].values[0]
                 ) / self.parameters['battery_efficiency'].values[0], 0)

            capa_to_discharge = max((new_soc *
                                     self.parameters['battery_capacity'].values[0]
                                     - self.parameters['battery_soc_min'].values[0] *
                                     self.parameters['battery_capacity'].values[0]
                                     ) * self.parameters['battery_efficiency'].values[0], 0)

            new_dict['battery_soc']=new_soc
            new_dict['capa_to_discharge'] = capa_to_discharge
            new_dict['capa_to_charge'] = capa_to_charge

        if self.architecture['grid'] == 1 :
            new_dict['grid_status'] = next_grid
            new_dict['grid_price_import'] = next_price_import
            new_dict['grid_price_export'] = next_price_export
            new_dict['grid_co2'] = next_co2

        for j in df:
            df[j].append(new_dict[j])

        #df = df.append(dict,ignore_index=True)



        return df


    #now we consider all the generators on all the time (mainly concern genset)

    def _check_constraints_genset(self, p_genset):
        """ This function checks that the constraints of the genset are respected."""
        if p_genset < 0:
            p_genset =0
            print('error, genset power cannot be lower than 0')

        if p_genset < self.parameters['genset_rated_power'].values[0] * self.parameters['genset_pmin'].values[0] and p_genset >1:
            p_genset = self.parameters['genset_rated_power'].values[0] * self.parameters['genset_pmin'].values[0]

        if p_genset > self.parameters['genset_rated_power'].values[0] * self.parameters['genset_pmax'].values[0]:
            p_genset = self.parameters['genset_rated_power'].values[0] * self.parameters['genset_pmax'].values[0]

        return p_genset

    def _check_constraints_grid(self, p_import, p_export):
        """ This function checks that the constraints of the grid are respected."""
        if p_import < 0:
            p_import = 0

        if p_export <0:
            p_export = 0

        if p_import > self._zero and p_export > self._zero:
        	pass
            #print ('cannot import and export at the same time')
            #todo how to deal with that?

        if p_import > self.parameters['grid_power_import'].values[0]:
            p_import = self.parameters['grid_power_import'].values[0]

        if p_export > self.parameters['grid_power_export'].values[0]:
            p_export = self.parameters['grid_power_export'].values[0]

        return p_import, p_export

    def _check_constraints_battery(self, p_charge, p_discharge, status):
        """ This function checks that the constraints of the battery are respected."""

        if p_charge < 0:
            p_charge = 0

        if p_discharge < 0:
            p_discharge = 0

        if p_charge > self._zero and p_discharge > self._zero:
            pass
            #print ('cannot import and export at the same time')
            #todo how to deal with that?

        capa_to_charge = max(
                        (self.parameters['battery_soc_max'].values[0] * self.parameters['battery_capacity'].values[0] -
                         status['battery_soc'][-1] *
                         self.parameters['battery_capacity'].values[0]
                         ) / self.parameters['battery_efficiency'].values[0], 0)

        capa_to_discharge = max((status['battery_soc'][-1] *
                                 self.parameters['battery_capacity'].values[0]
                                 - self.parameters['battery_soc_min'].values[0] *
                                 self.parameters['battery_capacity'].values[0]
                                 ) * self.parameters['battery_efficiency'].values[0], 0)

        if p_charge > capa_to_charge or p_charge > self.parameters['battery_power_charge'].values[0]:
            p_charge = min (capa_to_charge, self.parameters['battery_power_charge'].values[0])


        if p_discharge > capa_to_discharge or p_discharge > self.parameters['battery_power_discharge'].values[0]:
            p_discharge = min (capa_to_discharge, self.parameters['battery_power_discharge'].values[0])

        return p_charge, p_discharge

    def _record_production(self, control_dict, df, status):
        """
        This function records the actual production occuring in the microgrid. Based on the control actions and the
        parameters of the microgrid. This function will check that the control actions respect the constraints of
        the microgrid and then record what generators have produced energy.

        Parameters
        ----------
        control_dict : dictionnary
            Dictionnary representing the control actions taken by an algorithm (either benchmark or in the run function).
        df: dataframe
            Previous version of the record_production dataframe (coming from the run loop, or benchmarks).
        status: dataframe
            One line dataframe representing the changing parameters of the microgrid.

        Notes
        -----
        The mechanism to incure a penalty in case of over-generation is not yet in its final version.
        """

        #todo enforce constraints
        #todo make sure the control actions repect their respective constriants

        #todo pv

        if not isinstance(df, dict):
            raise TypeError('We know this should be named differently but df needs to be dict, is {}'.format(type(df)))

        total_load = 0
        total_production = 0
        threshold = 0.001
        total_load = control_dict['load']
        temp_pv = 0

        #check the generator constraints



        try:
            total_production += control_dict['loss_load']
        except:
            control_dict['loss_load'] =0
        try:
            total_production += control_dict['overgeneration']
        except:
            control_dict['overgeneration'] = 0

        if self.architecture['PV'] == 1:
            try:
                total_production += control_dict['pv']
                control_dict['pv_consummed'] = max(0,min(control_dict['pv_consummed'], control_dict['pv']))
                temp_pv += max(control_dict['pv_consummed'], control_dict['pv'])

            except:
                control_dict['pv_consummed'] = 0

            try:
                if control_dict['pv_curtailed'] <0:
                    control_dict['pv_curtailed'] = 0
                total_production -= control_dict['pv_curtailed']
            except:
                control_dict['pv_curtailed'] = 0
                control_dict['pv_curtailed'] = control_dict['pv'] - control_dict['pv_consummed']
                total_production -= control_dict['pv_curtailed']

        if self.architecture['genset'] == 1:
            try:
                p_genset = control_dict['genset']
            except:
                p_genset = 0
                print("this microgrid has a genset, you should add a 'genset' field to your control dictionnary")

            control_dict['genset'] = self._check_constraints_genset(p_genset)
            total_production += control_dict['genset']

        if self.architecture['grid'] == 1:
            try:
                p_import = control_dict['grid_import']
                p_export = control_dict['grid_export']
            except:
                p_import = 0
                p_export = 0
                print("this microgrid is grid connected, you should add a 'grid_import' and a 'grid_export' field to your control dictionnary")

            p_import, p_export = self._check_constraints_grid(p_import, p_export)

            control_dict['grid_import'] = p_import * status['grid_status'][-1]
            control_dict['grid_export'] = p_export * status['grid_status'][-1]

            total_production += control_dict['grid_import']
            total_production -= control_dict['grid_export']

        if self.architecture['battery'] == 1:

            try:
                p_charge=control_dict['battery_charge']
                p_discharge=control_dict['battery_discharge']

            except:
                p_charge = 0
                p_discharge = 0
                print(
                    "this microgrid is grid connected, you should add a 'battery_charge' and a 'battery_discharge' field to your control dictionnary")


            p_charge, p_discharge = self._check_constraints_battery(p_charge,
                                                                   p_discharge,
                                                                   status)
            control_dict['battery_discharge'] = p_discharge
            control_dict['battery_charge'] = p_charge

            total_production += control_dict['battery_discharge']
            total_production -= control_dict['battery_charge']

        if abs(total_production - total_load) < threshold:
            control_dict['overgeneration'] =0
            control_dict['loss_load'] = 0
            for j in df:
                df[j].append(control_dict[j])

        elif total_production > total_load :
            # here we consider we produced more than needed ? we pay the price of the full cost proposed?
            # penalties ?
            control_dict['overgeneration'] = total_production-total_load
            control_dict['loss_load'] = 0
            for j in df:
                df[j].append(control_dict[j])
            #df = df.append(control_dict, ignore_index=True)
            #print('total_production > total_load')
            #print(control_dict)

        elif total_production < total_load :
            control_dict['loss_load']+= total_load-total_production
            control_dict['overgeneration'] = 0
            for j in df:
                df[j].append(control_dict[j])
            #df = df.append(control_dict, ignore_index=True)
            #print('total_production < total_load')
            #print(control_dict)

        return df

    def _record_co2(self, control_dict, df, grid_co2=0):
        """ This function record the cost of operating the microgrid at each time step."""
        co2 = 0

        if self.architecture['genset'] == 1:
            co2 += control_dict['genset'] * self.parameters['genset_co2'].values[0]

        if self.architecture['grid'] == 1:
            co2 += grid_co2 * control_dict['grid_import']

        cost_dict = {'co2': co2}

        df['co2'].append( co2)

        return df


    def _record_cost(self, control_dict, df, df_co2, cost_import=0, cost_export=0):
        """ This function record the cost of operating the microgrid at each time step."""

        if not isinstance(df, dict):
            raise TypeError('We know this should be named differently but df needs to be dict, is {}'.format(type(df)))

        cost = 0
        cost += control_dict['loss_load'] * self.parameters['cost_loss_load'].values[0]
        cost += control_dict['overgeneration'] * self.parameters['cost_overgeneration'].values[0]

        if self.architecture['genset'] == 1:
            cost += control_dict['genset'] * self.parameters['fuel_cost'].values[0]

        if self.architecture['grid'] ==1:


            cost +=( cost_import * control_dict['grid_import']
                     - cost_export * control_dict['grid_export'])


        if self.architecture['battery'] ==1 :
            cost += (control_dict['battery_charge']+control_dict['battery_discharge'])*self.parameters['battery_cost_cycle'].values[0]

        cost += self.parameters['cost_co2'].values[0] * df_co2['co2'][-1]
        cost_dict= {'cost': cost}

        df['cost'].append(cost)

        return df

    ########################################################
    # PRINT FUNCTIONS
    ########################################################


    def print_load_pv(self):

        print('Load')
        fig1 = self._load_ts.iplot(asFigure=True)
        iplot(fig1)

        print('PV')
        fig2 =self._pv_ts.iplot(asFigure=True)
        iplot(fig2)

    def print_actual_production(self):
        if self._df_record_actual_production != type(pd.DataFrame()):
            df = pd.DataFrame(self._df_record_actual_production)
            fig1 = df.iplot(asFigure=True)
            iplot(fig1)
        else:
            fig1 = self._df_record_actual_production.iplot(asFigure=True)
            iplot(fig1)

    def print_control(self):
        if self._df_record_control_dict != type(pd.DataFrame()):
            df = pd.DataFrame(self._df_record_control_dict)
            fig1 = df.iplot(asFigure=True)
            iplot(fig1)
        else:
            fig1 = self._df_record_control_dict.iplot(asFigure=True)
            iplot(fig1)

    def print_co2(self):
        if self._df_record_co2 != type(pd.DataFrame()):
            df = pd.DataFrame(self._df_record_co2)
            fig1 = df.iplot(asFigure=True)
            iplot(fig1)
        else:
            fig1 = self._df_record_co2.iplot(asFigure=True)
            iplot(fig1)

    def print_cumsum_cost(self):
        if self._df_record_cost != type(pd.DataFrame()):
            df = pd.DataFrame(self._df_record_cost)
            plt.plot(df.cumsum())
            plt.show()
        else:
            plt.plot(self._df_record_cost.cumsum())
            plt.show()



    def print_benchmark_cost(self):
        """
        This function prints the cumulative cost of the different benchmark ran and different part of the dataset
        depending on if split it in train/test or not.
        """

        if len(self.benchmarks.outputs_dict) == 0:
            print('No benchmark algorithms have been run, running all.')
            #self.benchmarks.run_benchmarks()

        if self._has_train_test_split:
            self.benchmarks.describe_benchmarks(test_split=self._has_train_test_split, test_index=self._limit_index)

        else:
            self.benchmarks.describe_benchmarks(test_split=False)

    def print_info(self):
        """ This function prints the main information regarding the microgrid."""

        print('Microgrid parameters')
        display(self.parameters)
        print('Architecture:')
        print(self.architecture)
        print('Actions: ')
        print(self._df_record_control_dict.keys())
        print('Control dictionnary:')
        print(self.control_dict)
        print('Status: ')
        print(self._df_record_state.keys())
        print('Has run mpc baseline:')
        print(self._has_run_mpc_baseline)
        print('Has run rule based baseline:')
        print(self._has_run_rule_based_baseline)

    def print_control_info(self):
        """ This function prints the control_dict that needs to be used to control the microgrid"""

        print('you should fill this dictionnary at each time step')
        print('it is included in the mg_data object')
        print('you can copy it by: ctrl = mg_data.control_dict')
        print('or you can use self.get_conrol_dict()')
        print('Control dictionnary:')
        print(self.control_dict)

    def print_updated_parameters(self):
        """ This function prints the last values for the parameters of the microgrid changing with time."""
        state={}
        for i in self._df_record_state:
            state[i] = self._df_record_state[i][-1]

        print(state)

    ########################################################
    # RL UTILITY FUNCTIONS
    ########################################################
    #todo add a forecasting function that add noise to the time series
    #todo forecasting function can be used for both mpc benchmart and rl loop


    #todo verbose
    def penalty(self, coef = 1):
        """Penalty that represents discrepancies between control dict and what really happens. """
        penalty = 0
        for i in self._df_record_control_dict:
            penalty += abs(self._df_record_control_dict[i][-1] - self._df_record_actual_production[i][-1])

        return penalty*coef

