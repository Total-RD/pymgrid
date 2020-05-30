import pandas as pd
import numpy as np
from copy import copy
import cvxpy as cp
import operator
import math
import time
import sys
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import matplotlib.pyplot as plt
import cufflinks as cf

init_notebook_mode(connected=False)
np.random.seed(123)
cf.set_config_file(theme='pearl')

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
    def __init__(self, param, status, price_import, price_export):
        self.power_export = param['grid_power_export'].values[0]
        self.power_import = param['grid_power_import'].values[0]
        self.price_export = price_export #param['grid_price_export'].values[0]
        self.price_import = price_import # param['grid_price_import'].values[0]
        self.status = status


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
        architecture : dictionnary
            A dictionnary containing a binary variable for each possible generator and indicating if
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
        self.parameters = parameters['parameters']
        self.architecture =  parameters['architecture']

        #different timeseries
        self._load_ts=parameters['load']
        self._pv_ts=parameters['pv']

        self.pv=self._pv_ts.iloc[0,0]
        self.load = self._load_ts.iloc[0, 0]
        self._next_load = self._load_ts.iloc[1,0]
        self._next_pv = self._pv_ts.iloc[1,0]
        if parameters['architecture']['grid']==1:
            self._grid_status_ts=parameters['grid_ts'] #time series of outages
            #self.grid_status = self._grid_status_ts.iloc[0, 0]
            #todo if we move to time series of price
            self._grid_price_import=parameters['grid_price_import']
            self._grid_price_export=parameters['grid_price_export']

            self._next_grid_status = self._grid_status_ts.iloc[0, 0]
            self._next_grid_price_export = self._grid_price_export.iloc[0, 0]
            self._next_grid_price_import = self._grid_price_import.iloc[0, 0]

        # those dataframe record what is hapepning at each time step
        self._df_record_control_dict=parameters['df_actions']
        self._df_record_state = parameters['df_status']
        self._df_record_actual_production = parameters['df_actual_generation']
        self._df_record_cost = parameters['df_cost']
        self._df_cost_per_epochs = parameters['df_cost']
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

        if self.architecture['battery'] == 1:
            self.battery = Battery(self.parameters,
                                   self._df_record_state.capa_to_charge.iloc[0],
                                   self._df_record_state.capa_to_discharge.iloc[0])
        if self.architecture['genset'] == 1:
            self.genset = Genset(self.parameters)
        if self.architecture['grid'] == 1:
            self.grid = Grid(self.parameters, self._grid_status_ts.iloc[0, 0],
                             self._grid_price_import.iloc[0, 0],
                             self._grid_price_export.iloc[0, 0])


    def set_horizon(self, horizon):
        """Function used to change the horizon of the simulation."""
        self.horizon = horizon

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
        return self._df_record_cost.iloc[-1].values[0]


    def get_updated_values(self):
        """
        Function that returns microgrid parameters that change with time. Depending on the architecture we have:
            - PV production
            - Load
            - Battery state of charge
            - Battery capacity to charge
            - Battery capacity to discharge
            - Whether the grid is connected or not
        """
        mg_data = {}

        for i in self._df_record_state.columns:
            mg_data[i] = self._df_record_state[i].iloc[-1]

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

            self._df_record_cost = self._record_cost(self._df_record_actual_production.iloc[-1,:].to_dict(),
                                                               self._df_record_cost, self.grid.price_import, self.grid.price_export)
            self._df_record_state = self._update_status(control_dict,
                                                        self._df_record_state, self._next_load, self._next_pv,
                                                        self._next_grid_status, self._next_grid_price_import,
                                                        self._next_grid_price_export)


        else:
            self._df_record_cost = self._record_cost(self._df_record_actual_production.iloc[-1, :].to_dict(),
                                                     self._df_record_cost)
            self._df_record_state = self._update_status(control_dict,
                                                        self._df_record_state, self._next_load, self._next_pv)




        self._tracking_timestep += 1
        self.update_variables()

        if self._tracking_timestep == self._data_length - self.horizon:
            self.done = True


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

            self._next_pv = self._pv_ts.iloc[self._tracking_timestep+1, 0]
            self._next_load = self._load_ts.iloc[self._tracking_timestep+1, 0]


        if self.architecture['grid']==1:
            if self._data_set_to_use == 'training':
                self.grid.status = self._grid_status_train.iloc[self._tracking_timestep, 0]
                self.grid.price_import = self._grid_price_import_train.iloc[self._tracking_timestep,0]
                self.grid.price_export = self._grid_price_export_train.iloc[self._tracking_timestep,0]

                self._next_grid_status = self._grid_status_train.iloc[self._tracking_timestep +1, 0]
                self._next_grid_price_import = self._grid_price_import_train.iloc[self._tracking_timestep +1, 0]
                self._next_grid_price_export = self._grid_price_export_train.iloc[self._tracking_timestep +1, 0]

            if self._data_set_to_use == 'testing':
                self.grid.status = self._grid_status_test.iloc[self._tracking_timestep, 0]
                self.grid.price_import = self._grid_price_import_test.iloc[self._tracking_timestep, 0]
                self.grid.price_export = self._grid_price_export_test.iloc[self._tracking_timestep, 0]

                self._next_grid_status = self._grid_status_test.iloc[self._tracking_timestep + 1, 0]
                self._next_grid_price_import = self._grid_price_import_test.iloc[self._tracking_timestep + 1, 0]
                self._next_grid_price_export = self._grid_price_export_test.iloc[self._tracking_timestep + 1, 0]


            if self._data_set_to_use == 'all':
                self.grid.status = self._grid_status_ts.iloc[self._tracking_timestep, 0]
                self.grid.price_import = self._grid_price_import.iloc[self._tracking_timestep, 0]
                self.grid.price_export = self._grid_price_export.iloc[self._tracking_timestep, 0]

                self._next_grid_status = self._grid_status_ts.iloc[self._tracking_timestep + 1, 0]
                self._next_grid_price_import = self._grid_price_import.iloc[self._tracking_timestep + 1, 0]
                self._next_grid_price_export = self._grid_price_export.iloc[self._tracking_timestep + 1, 0]

        if self.architecture['battery'] == 1:
            self.battery.soc = self._df_record_state.battery_soc.iloc[-1]
            self.battery.capa_to_discharge = self._df_record_state.capa_to_discharge.iloc[-1]
            self.battery.capa_to_charge = self._df_record_state.capa_to_charge.iloc[-1]

    def reset(self, testing=False):
        """This function is used to reset the dataframes that track what is happening in simulation. Mainly used in RL."""
        if self._data_set_to_use == 'training':
            temp_cost = copy(self._df_record_cost)
            temp_cost['epoch'] = self._epoch
            self._df_cost_per_epochs = self._df_cost_per_epochs.append(temp_cost, ignore_index=True)

        self._df_record_control_dict = self._df_record_control_dict[0:0]
        self._df_record_state = self._df_record_state.iloc[:1]
        self._df_record_actual_production = self._df_record_actual_production[0:0]
        self._df_record_cost = self._df_record_cost[0:0]

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
    # FUNCTIONS TO UPDATE THE INTERNAL DATAFRAMES
    ########################################################


    def _record_action(self, control_dict, df):
        """ This function is used to record the actions taken, before being checked for feasability. """
        df = df.append(control_dict,ignore_index=True)

        return df


    def _update_status(self, control_dict, df, next_load, next_pv, next_grid = 0, next_price_import =0, next_price_export = 0):
        """ This function update the parameters of the microgrid that change with time. """
        #self.df_status = self.df_status.append(self.new_row, ignore_index=True)

        dict = {
            'load': next_load,
                    'pv': next_pv,
            'hour':self._tracking_timestep%24,
        }
        new_soc =np.nan
        if self.architecture['battery'] == 1:
            new_soc = df['battery_soc'].iloc[-1] + (control_dict['battery_charge']*self.parameters['battery_efficiency'].values[0]
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

            dict['battery_soc']=new_soc
            dict['capa_to_discharge'] = capa_to_discharge
            dict['capa_to_charge'] = capa_to_charge

        if self.architecture['grid'] == 1 :
            dict['grid_status'] = next_grid
            dict['grid_price_import'] = next_price_import
            dict['grid_price_export'] = next_price_export



        df = df.append(dict,ignore_index=True)



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
                         status['battery_soc'].iloc[-1] *
                         self.parameters['battery_capacity'].values[0]
                         ) / self.parameters['battery_efficiency'].values[0], 0)

        capa_to_discharge = max((status['battery_soc'].iloc[-1] *
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
                temp_pv += control_dict['pv_consummed']
            except:
                control_dict['pv_consummed'] = 0

            try:
                total_production -= control_dict['pv_curtailed']
            except:
                control_dict['pv_curtailed'] = 0
            control_dict['pv_curtailed'] = control_dict['pv'] - control_dict['pv_consummed']

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
            control_dict['grid_import'] = p_import
            control_dict['grid_export'] = p_export

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
            df = df.append(control_dict, ignore_index=True)

        elif total_production > total_load :
            # here we consider we produced more than needed ? we pay the price of the full cost proposed?
            # penalties ?
            control_dict['overgeneration'] = total_production-total_load
            df = df.append(control_dict, ignore_index=True)
            #print('total_production > total_load')
            #print(control_dict)

        elif total_production < total_load :
            control_dict['loss_load']+= total_load-total_production
            df = df.append(control_dict, ignore_index=True)
            #print('total_production < total_load')
            #print(control_dict)

        return df

    def _record_cost(self, control_dict, df, cost_import=0, cost_export=0):
        """ This function record the cost of operating the microgrid at each time step."""
        cost = 0
        cost += control_dict['loss_load'] * self.parameters['cost_loss_load'].values[0]
        cost += control_dict['overgeneration'] * self.parameters['cost_overgeneration'].values[0]

        if self.architecture['genset'] == 1:
            cost += control_dict['genset'] * self.parameters['fuel_cost'].values[0]

        if self.architecture['grid'] ==1:


            cost +=( cost_import * control_dict['grid_import']
                     - cost_export * control_dict['grid_export'])


        if self.architecture['battery'] ==1 :
            cost+= (control_dict['battery_charge']+control_dict['battery_discharge'])*self.parameters['battery_cost_cycle'].values[0]

        cost_dict= {'cost': cost}


        df = df.append({'cost': cost}, ignore_index=True)

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
        fig1 = self._df_record_actual_production.iplot(asFigure=True)
        iplot(fig1)

    def print_control(self):
        fig1 = self._df_record_control_dict.iplot(asFigure=True)
        iplot(fig1)

    def print_cumsum_cost(self):
        plt.plot(self._df_record_cost.cumsum())
        plt.show()



    def print_benchmark_cost(self):
        """
        This function prints the cumulative cost of the different benchmark ran and different part of the dataset
        depending on if split it in train/test or not.
        """
        if self._has_train_test_split == False:
            if self._has_run_rule_based_baseline == True:
                print('Rule based cost: ', self._baseline_priority_list_cost.sum())

            if self._has_run_mpc_baseline == True:
                print('MPC cost: ', self._baseline_linprog_cost.sum())

        else:
            if self._has_run_rule_based_baseline == True:
                print('Training rule based cost: ', self._baseline_priority_list_cost.iloc[:self._limit_index].sum())
                print('Testing rule based cost: ', self._baseline_priority_list_cost.iloc[self._limit_index:].sum())

            if self._has_run_mpc_baseline == True:
                print('Training MPC cost: ', self._baseline_linprog_cost.iloc[:self._limit_index].sum())
                print('Testing MPC cost: ', self._baseline_linprog_cost.iloc[self._limit_index:].sum())


    def print_info(self):
        """ This function prints the main information regarding the microgrid."""

        print('Microgrid parameters')
        display(self.parameters)
        print('Architecture:')
        print(self.architecture)
        print('Actions: ')
        print(self._df_record_control_dict.columns)
        print('Control dictionnary:')
        print(self.control_dict)
        print('Status: ')
        print(self._df_record_state.columns)
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
        for i in self._df_record_state.columns:
            state[i] = self._df_record_state[i].iloc[-1]

        print(state)


    ########################################################
    # BENCHMARK RELATED FUNCTIONS
    ########################################################

    def _generate_priority_list(self, architecture, parameters , grid_status=0, price_import = 0, price_export=0):
        """
        Depending on the architecture of the microgrid and grid related import/export costs, this function generates a
        priority list to be run in the rule based benchmark.
        """
        # compute marginal cost of each ressource
        # construct priority list
        # should receive fuel cost and cost curve, price of electricity
        if  architecture['grid'] == 1:


            if price_export / (parameters['battery_efficiency'].values[0]**2) < price_import:

                # should return something like ['gen', starting at in MW]?
                priority_dict = {'PV': 1 * architecture['PV'],
                                 'battery': 2 * architecture['battery'],
                                 'grid': int(3 * architecture['grid'] * grid_status),
                                 'genset': 4 * architecture['genset']}

            else:
                # should return something like ['gen', starting at in MW]?
                priority_dict = {'PV': 1 * architecture['PV'],
                                 'battery': 3 * architecture['battery'],
                                 'grid': int(2 * architecture['grid'] * grid_status),
                                 'genset': 4 * architecture['genset']}

        else:
            priority_dict = {'PV': 1 * architecture['PV'],
                             'battery': 2 * architecture['battery'],
                             'grid': 0,
                             'genset': 4 * architecture['genset']}

        return priority_dict

    def _run_priority_based(self, load, pv, parameters, status, priority_dict):
        """
        This function runs one loop of rule based control, based on a priority list, load and pv, dispatch the
        generators

        Parameters
        ----------
        load: float
            Demand value
        PV: float
            PV generation
        parameters: dataframe
            The fixed parameters of the mircrogrid
        status: dataframe
            The parameters of the microgrid changing with time.
        priority_dict: dictionnary
            Dictionnary representing the priority with which run each generator.

        """

        temp_load = load
        # todo add reserves to pymgrid
        excess_gen = 0

        pCharge = 0
        pDischarge = 0
        pImport = 0
        pExport = 0
        pGenset = 0
        load_not_matched = 0
        pv_not_curtailed = 0
        self_consumed_pv = 0


        sorted_priority = priority_dict
        min_load = 0
        if self.architecture['genset'] == 1:
            #load - pv - min(capa_to_discharge, p_discharge) > 0: then genset on and min load, else genset off
            grid_first = 0
            capa_to_discharge = max(min((status['battery_soc'].iloc[-1] *
                                     parameters['battery_capacity'].values[0]
                                     - parameters['battery_soc_min'].values[0] *
                                     parameters['battery_capacity'].values[0]
                                     ) * parameters['battery_efficiency'].values[0], self.battery.p_discharge_max), 0)

            if self.architecture['grid'] == 1 and sorted_priority['grid'] < sorted_priority['genset'] and sorted_priority['grid']>0:
                grid_first=1

            if temp_load > pv + capa_to_discharge and grid_first ==0:

                min_load = self.parameters['genset_rated_power'].values[0] * self.parameters['genset_pmin'].values[0]
                if min_load <= temp_load:
                    temp_load = temp_load - min_load
                else:
                    temp_load = min_load
                    priority_dict = {'PV': 0,
                                     'battery': 0,
                                     'grid': 0,
                                     'genset': 1}

        sorted_priority = sorted(priority_dict.items(), key=operator.itemgetter(1))
        # for gen with prio i in 1:max(priority_dict)
        # we sort the priority list
        # probably we should force the PV to be number one, the min_power should be absorbed by genset, grid?
        # print (sorted_priority)
        for gen, priority in sorted_priority:  # .iteritems():

            if priority > 0:

                if gen == 'PV':
                    temp_load_for_excess = copy(temp_load)
                    # print (temp_load * self.maximum_instantaneous_pv_penetration - run_dict['next_pv'])
                    self_consumed_pv = min(temp_load, pv)  # self.maximum_instantaneous_pv_penetration,
                    temp_load = max(0, temp_load - self_consumed_pv)
                    excess_gen = pv - self_consumed_pv
                    # temp_load = max(0, temp_load * self.maximum_instantaneous_pv_penetration - run_dict['next_pv'])
                    # excess_gen = abs(min(0, temp_load_for_excess * self.maximum_instantaneous_pv_penetration - run_dict['next_pv']))

                    # print (temp_load)
                    pv_not_curtailed = pv_not_curtailed + pv - excess_gen

                if gen == 'battery':

                    capa_to_charge = max(
                        (parameters['battery_soc_max'].values[0] * parameters['battery_capacity'].values[0] -
                         status['battery_soc'].iloc[-1] *
                         parameters['battery_capacity'].values[0]
                         ) / self.parameters['battery_efficiency'].values[0], 0)
                    capa_to_discharge = max((status['battery_soc'].iloc[-1] *
                                             parameters['battery_capacity'].values[0]
                                             - parameters['battery_soc_min'].values[0] *
                                             parameters['battery_capacity'].values[0]
                                             ) * parameters['battery_efficiency'].values[0], 0)
                    if temp_load > 0:
                        pDischarge = max(0, min(capa_to_discharge, parameters['battery_power_discharge'].values[0],
                                                temp_load))
                        temp_load = temp_load - pDischarge

                    elif excess_gen > 0:
                        pCharge = max(0, min(capa_to_charge, parameters['battery_power_charge'].values[0],
                                             excess_gen))
                        excess_gen = excess_gen - pCharge

                        pv_not_curtailed = pv_not_curtailed + pCharge

                if gen == 'grid':
                    if temp_load > 0:
                        pImport = temp_load
                        temp_load = 0



                    elif excess_gen > 0:
                        pExport = excess_gen
                        excess_gen = 0

                        pv_not_curtailed = pv_not_curtailed + pExport

                if gen == 'genset':
                    if temp_load > 0:
                        pGenset = temp_load + min_load
                        temp_load = 0
                        # pGenset = pGenset + min_load
                        min_load = 0

        if temp_load > 0:
            load_not_matched = 1

        control_dict = {'battery_charge': pCharge,
                        'battery_discharge': pDischarge,
                        'genset': pGenset,
                        'grid_import': pImport,
                        'grid_export': pExport,
                        'loss_load': load_not_matched,
                        'pv_consummed': pv_not_curtailed,
                        'pv_curtailed': pv - pv_not_curtailed,
                        'load': load,
                        'pv': pv}
        # 'nb_gen_min': nb_gen_min}

        return control_dict


    def _mpc_lin_prog_cvxpy(self, parameters, load, pv, grid, status, price_import, price_export, horizon=24):
        """ This function implements one loop of the MPC, optimizing the microgrid over the next horizon."""

        # todo switch to a matrix structure
        load = np.reshape(load, (horizon,))

        # variables
        # if self.architecture['genset'] ==1:
        p_genset = cp.Variable((horizon,), pos=True)
        u_genset = cp.Variable((horizon,), boolean=True)

        # if self.architecture['grid']==1:
        p_grid_import = cp.Variable((horizon,), pos=True)
        p_grid_export = cp.Variable((horizon,), pos=True)


        # if self.architecture['battery'] == 1:
        p_charge = cp.Variable((horizon,), pos=True)
        p_discharge = cp.Variable((horizon,), pos=True)


        # if self.architecture['pv']==1:
        p_curtail_pv = cp.Variable((horizon,), pos=True)

        p_loss_load = cp.Variable((horizon,), pos=True)

        # parameters
        cost_loss_load = parameters['cost_loss_load'].values[0] * np.ones(horizon)

        # Constraints
        constraints = []
        total_cost = 0.0
        constraints += [p_loss_load <= load]
        if self.architecture['genset'] == 1:
            p_genset_min = parameters['genset_pmin'].values[0] * parameters['genset_rated_power'].values[0]
            p_genset_max = parameters['genset_pmax'].values[0] * parameters['genset_rated_power'].values[0]
            fuel_cost = parameters['fuel_cost'].values[0] * np.ones(horizon)

            for t in range(horizon):
                constraints += [p_genset[t] >= u_genset[t]*p_genset_min,
                                p_genset[t] <= u_genset[t]*p_genset_max]

                total_cost += (p_genset[t] * fuel_cost[t])

        else:
            for t in range(horizon):
                constraints += [p_genset[t] == 0]

        if self.architecture['grid'] == 1:

            grid = np.reshape(grid, (horizon,))
            p_grid_import_max = parameters['grid_power_import'].values[0]
            p_grid_export_max = parameters['grid_power_export'].values[0]
            p_price_import = np.reshape(price_import, (horizon,))
            p_price_export = np.reshape(price_export, (horizon,))

            for t in range(horizon):
                constraints += [p_grid_import[t] <= p_grid_import_max * grid[t],
                                p_grid_export[t] <= p_grid_export_max * grid[t],
                                ]

                total_cost += (p_grid_import[t] * p_price_import[t]
                               - p_grid_export[t] * p_price_export[t])




        else:
            for t in range(horizon):
                constraints += [p_grid_import[t] == 0,
                                p_grid_export[t] == 0]

        if self.architecture['battery'] == 1:
            battery_soc = cp.Variable((horizon,), pos=True)

            cost_battery_cycle = parameters['battery_cost_cycle'].values[0] * np.ones(horizon)

            p_charge_max = parameters['battery_power_charge'].values[0]
            p_discharge_max = parameters['battery_power_discharge'].values[0]

            for t in range(horizon):
                constraints += [p_charge[t] <= p_charge_max,
                                p_discharge[t] <= p_discharge_max,]

                constraints += [battery_soc[t] >= parameters['battery_soc_min'].values[0],
                                battery_soc[t] <= parameters['battery_soc_max'].values[0]]

                total_cost += (p_charge[t] * cost_battery_cycle[t] + p_discharge[t] * cost_battery_cycle[t])

            soc_0 = status.iloc[-1]['battery_soc']
            constraints += [battery_soc[0] == soc_0 + (p_charge[0] * parameters['battery_efficiency'].values[0]
                                                       - p_discharge[0] / parameters['battery_efficiency'].values[
                                                           0]) /
                            parameters['battery_capacity'].values[0]]
            for t in range(1, horizon):
                constraints += [
                    battery_soc[t] == battery_soc[t - 1] + (p_charge[t] * parameters['battery_efficiency'].values[0]
                                                            - p_discharge[t] /
                                                            parameters['battery_efficiency'].values[
                                                                0]) / parameters['battery_capacity'].values[0]]


        else:
            for t in range(horizon):
                constraints += [p_charge[t] == 0,
                                p_discharge[t] == 0]

        if self.architecture['PV'] == 1:
            pv = np.reshape(pv, (horizon,))
            for t in range(horizon):
                constraints += [p_curtail_pv[t] <= pv[t]]
        else:
            for t in range(horizon):
                constraints += [p_curtail_pv[t] == 0]

        # constraint balance of power

        for t in range(horizon):
            total_cost += p_loss_load[t] * cost_loss_load[t]
            constraints += [p_genset[t]
                            + p_discharge[t]
                            - p_charge[t]
                            - p_curtail_pv[t]
                            + p_loss_load[t]
                            + p_grid_import[t]
                            - p_grid_export[t]
                            == load[t] - pv[t]]

        # Objective function
        obj = cp.Minimize(total_cost)

        prob = cp.Problem(obj, constraints)
        prob.solve()  # verbose=True)#, solver=cp.ECOS,)

        control_dict = {'battery_charge': p_charge.value[0],
                        'battery_discharge': p_discharge.value[0],
                        'genset': p_genset.value[0],
                        'grid_import': p_grid_import.value[0],
                        'grid_export': p_grid_export.value[0],
                        'loss_load': p_loss_load.value[0],
                        'pv_consummed': pv[0] - p_curtail_pv.value[0],
                        'pv_curtailed': p_curtail_pv.value[0],
                        'load': load[0],
                        'pv': pv[0]}

        return control_dict

    def _baseline_rule_based(self, priority_list=0, length=8760):
        """ This function runs the rule based benchmark over the datasets (load and pv profiles) in the microgrid."""

        self._baseline_priority_list_action = copy(self._df_record_control_dict)
        self._baseline_priority_list_update_status = copy(self._df_record_state)
        self._baseline_priority_list_record_production = copy(self._df_record_actual_production)
        self._baseline_priority_list_cost = copy(self._df_record_cost)

        n = length - self.horizon
        print_ratio = n/100

        for i in range(length - self.horizon):

            e = i

            if e == (n-1):

               e = n

            e = e/print_ratio

            sys.stdout.write("\rIn Progress %d%% " % e)
            sys.stdout.flush()

            if e == 100:

                sys.stdout.write("\nRules Based Calculation Finished")
                sys.stdout.flush()
                sys.stdout.write("\n")


            if self.architecture['grid'] == 1:
                priority_dict = self._generate_priority_list(self.architecture, self.parameters,
                                                             self._grid_status_ts.iloc[i].values[0],
                                                             self._grid_price_import.iloc[i].values[0],
                                                             self._grid_price_export.iloc[i].values[0])
            else:
                priority_dict = self._generate_priority_list(self.architecture, self.parameters)

            control_dict = self._run_priority_based(self._load_ts.iloc[i].values[0], self._pv_ts.iloc[i].values[0],
                                                    self.parameters,
                                                    self._baseline_priority_list_update_status, priority_dict)

            self._baseline_priority_list_action = self._record_action(control_dict,
                                                                      self._baseline_priority_list_action)

            self._baseline_priority_list_record_production = self._record_production(control_dict,
                                                                                     self._baseline_priority_list_record_production,
                                                                                     self._baseline_priority_list_update_status)


            if self.architecture['grid']==1:

                self._baseline_priority_list_update_status = self._update_status(
                    self._baseline_priority_list_record_production.iloc[-1, :].to_dict(),
                    self._baseline_priority_list_update_status, self._load_ts.iloc[i + 1].values[0],
                    self._pv_ts.iloc[i + 1].values[0],
                    self._grid_status_ts.iloc[i + 1].values[0],
                    self._grid_price_import.iloc[i + 1].values[0],
                    self._grid_price_export.iloc[i + 1].values[0])


                self._baseline_priority_list_cost = self._record_cost(
                    self._baseline_priority_list_record_production.iloc[-1, :].to_dict(),
                    self._baseline_priority_list_cost, self._grid_price_import.iloc[i,0], self._grid_price_export.iloc[i,0])
            else:

                self._baseline_priority_list_update_status = self._update_status(
                    self._baseline_priority_list_record_production.iloc[-1, :].to_dict(),
                    self._baseline_priority_list_update_status, self._load_ts.iloc[i + 1].values[0],
                    self._pv_ts.iloc[i + 1].values[0])

                self._baseline_priority_list_cost = self._record_cost(
                    self._baseline_priority_list_record_production.iloc[-1, :].to_dict(),
                    self._baseline_priority_list_cost)

        self._has_run_rule_based_baseline = True

    def _baseline_linprog(self, forecast_error=0, length=8760):
        """ This function runs the MPC benchmark over the datasets (load and pv profiles) in the microgrid."""

        self._baseline_linprog_action = copy(self._df_record_control_dict)
        self._baseline_linprog_update_status = copy(self._df_record_state)
        self._baseline_linprog_record_production = copy(self._df_record_actual_production)
        self._baseline_linprog_cost = copy(self._df_record_cost)

        n = length - self.horizon
        print_ratio = n/100

        for i in range(n):

            e = i

            if e == (n-1):

               e = n

            e = e/print_ratio

            sys.stdout.write("\rIn Progress %d%% " % e)
            sys.stdout.flush()

            if e == 100:

                sys.stdout.write("\nMPC Calculation Finished")
                sys.stdout.flush()  
                sys.stdout.write("\n")
        	
            if self.architecture['grid'] == 0:
                temp_grid = np.zeros(self.horizon)
                price_import =np.zeros(self.horizon)
                price_export = np.zeros(self.horizon)
            else:
                temp_grid = self._grid_status_ts.iloc[i:i + self.horizon].values
                price_import= self._grid_price_import.iloc[i:i + self.horizon].values
                price_export= self._grid_price_export.iloc[i:i + self.horizon].values

            control_dict = self._mpc_lin_prog_cvxpy(self.parameters, self._load_ts.iloc[i:i + self.horizon].values,
                                                    self._pv_ts.iloc[i:i + self.horizon].values, temp_grid,
                                                    self._baseline_linprog_update_status, price_import, price_export,
                                                    self.horizon)

            self._baseline_linprog_action = self._record_action(control_dict, self._baseline_linprog_action)

            self._baseline_linprog_record_production = self._record_production(control_dict,
                                                                               self._baseline_linprog_record_production,
                                                                               self._baseline_linprog_update_status)


            if self.architecture['grid'] == 1:

                self._baseline_linprog_update_status = self._update_status(
                    self._baseline_linprog_record_production.iloc[-1, :].to_dict(),
                    self._baseline_linprog_update_status,
                    self._load_ts.iloc[i + 1].values[0],
                    self._pv_ts.iloc[i + 1].values[0],
                    self._grid_status_ts.iloc[i + 1].values[0],
                    self._grid_price_import.iloc[i + 1].values[0],
                    self._grid_price_export.iloc[i + 1].values[0])

                self._baseline_linprog_cost = self._record_cost(
                    self._baseline_linprog_record_production.iloc[-1, :].to_dict(),
                    self._baseline_linprog_cost, self._grid_price_import.iloc[i,0], self._grid_price_export.iloc[i,0])



            else:

                self._baseline_linprog_update_status = self._update_status(
                    self._baseline_linprog_record_production.iloc[-1, :].to_dict(),
                    self._baseline_linprog_update_status,
                    self._load_ts.iloc[i + 1].values[0],
                    self._pv_ts.iloc[i + 1].values[0])

                self._baseline_linprog_cost = self._record_cost(
                    self._baseline_linprog_record_production.iloc[-1, :].to_dict(),
                    self._baseline_linprog_cost)


            self._has_run_mpc_baseline = True



    def compute_benchmark(self, benchmark_to_compute='all', length=8760):
        """
        This function can be used to run all the benchmarks, or one at a time depending on the argument being
        passed.
        """

        if benchmark_to_compute == 'all':
            if self._has_run_rule_based_baseline == False:
                self._baseline_rule_based(length=length)
            if self._has_run_mpc_baseline == False:
                self._baseline_linprog(length=length)

        if benchmark_to_compute == 'rule_based' and self._has_run_rule_based_baseline == False:
            self._baseline_rule_based(length=length)

        if benchmark_to_compute == 'mpc_linprog' and self._has_run_mpc_baseline == False:
            self._baseline_linprog(length=length)

    ########################################################
    # RL UTILITY FUNCTIONS
    ########################################################
    #todo add a forecasting function that add noise to the time series
    #todo forecasting function can be used for both mpc benchmart and rl loop


    #todo verbose
    def penalty(self, coef = 1):
        """Penalty that represents discrepancies between control dict and what really happens. """
        penalty = 0
        for i in self._df_record_control_dict.columns:
            penalty += abs(self._df_record_control_dict[i].iloc[-1] - self._df_record_actual_production[i].iloc[-1])

        return penalty*coef