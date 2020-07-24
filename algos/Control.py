from utils import DataGenerator as dg
from pymgrid import Microgrid
import pandas as pd
import numpy as np
from copy import copy
import time, sys
from matplotlib import pyplot as plt
import cvxpy as cp
from scipy.sparse import csr_matrix
from pymgrid import MicrogridGenerator
import operator
from IPython.display import display



def return_underlying_data(microgrid):
    """
    Returns the pv, load, and grid data from the  microgrid in the same format as samples.
    :param microgrid, pymgrid.Microgrid.Microgrid
        microgrid to reformat underlying data for
    :return:
        data: pd.DataFrame, shape (8760,3)
            DataFrame with columns 'pv', 'load', 'grid', values of these respectively at each timestep.
    """
    pv_data = microgrid._pv_ts
    load_data = microgrid._load_ts

    pv_data = pv_data[pv_data.columns[0]]
    load_data = load_data[load_data.columns[0]]
    pv_data.name = 'pv'
    load_data.name = 'load'

    if microgrid.architecture['grid'] != 0:
        grid_data = microgrid._grid_status_ts
        if isinstance(grid_data, pd.DataFrame):
            grid_data = grid_data[grid_data.columns[0]]
            grid_data.name = 'grid'
        elif isinstance(grid_data, pd.Series):
            grid_data.name = 'grid'
        else:
            raise RuntimeError('Unable to handle microgrid._grid_status_ts of type {}.'.format(type(grid_data)))
    else:
        grid_data = pd.Series(data=[0] * len(microgrid._load_ts), name='grid')

    return pd.concat([pv_data, load_data, grid_data], axis=1)

class SampleAverageApproximation:
    """
    A class to run a Sample Average Approximation version of Stochastic MPC.

    Parameters:

        microgrid: pymgrid.Microgrid.Microgrid
            the underlying microgrid
        control_duration: int
            number of iterations to learn over

    Attributes:

        microgrid: pymgrid.Microgrid.Microgrid
            the underlying microgrid
        control_duration: int
            number of iterations to learn over
        mpc: algos.Control.ModelPredictiveControl
            An instance of MPC class to run MPC over for each sample
        NPV: utils.DataGenerator.NoisyPVData
            An instance of NoisyPVData to produce pv forecast and samples
        NL: utils.DataGenerator.NoisyLoadData
            An instance of NoisyLoadData to produce initial load forecast
        NG: utils.DataGenerator.NoisyGridData or None
            An instance of NoisyGridData to produce initial grid forecast. None if there is no grid
        forecasts: pd.DataFrame, shape (8760,3)
            load, pv, grid forecasts. See create_forecasts for details.
        samples: list of pd.DataFrame of shape (8760,3), or None
            list of samples created from sampling from distributions defined in forecasts.
                See sample_from_forecasts for details. None if sample_from_forecasts hasn't been called
    """
    def __init__(self, microgrid, control_duration=8760):
        if control_duration > 8760:
            raise ValueError('control_duration must be less than 8760')

        if not isinstance(microgrid, Microgrid.Microgrid):
            raise TypeError('microgrid must be of type \'pymgrid.Microgrid.Microgrid\', is {}'.format(type(microgrid)))

        self.microgrid = microgrid
        self.control_duration = control_duration
        self.mpc = ModelPredictiveControl(self.microgrid)

        self.NPV = dg.NoisyPVData(pv_data=self.microgrid._pv_ts)
        self.NL = dg.NoisyLoadData(load_data=self.microgrid._load_ts)
        if self.microgrid.architecture['grid'] != 0:
            self.NG = dg.NoisyGridData(grid_data=self.microgrid._grid_status_ts)
        else:
            self.NG = None

        self.forecasts = self.create_forecasts()
        self.samples = None

        # TODO: Then aggregate controls: 2) learn a function
        # TODO: Use these in sample average approximation to learn a policy

    def create_forecasts(self):
        """
        Creates pv, load, and grid forecasts that are then used to create samples.

        :return:
            df, pd.DataFrame,  shape (8760,3)
                DataFrame with columns of 'pv', 'load', and 'grid', containing values for each at all 8760 timesteps
        """
        # TODO: modify this to allow for different amounts of noise in forecasts

        pv_forecast = self.NPV.sample()
        load_forecast = self.NL.sample()

        if self.microgrid.architecture['grid'] != 0:
            grid_forecast = self.NG.sample()
        else:
            grid_forecast = pd.Series(data=[0] * len(self.microgrid._load_ts), name='grid')

        return pd.concat([pv_forecast, load_forecast, grid_forecast], axis=1)

    def _return_underlying_data(self):
        """
        Returns the pv, load, and grid data from the underlying microgrid in the same format as samples.
        :return:
            data: pd.DataFrame, shape (8760,3)
                DataFrame with columns 'pv', 'load', 'grid', values of these respectively at each timestep.
        """

        return return_underlying_data(self.microgrid)

    def sample_from_forecasts(self, n_samples=100, **sampling_args):
        """
            Generates samples of load, grid, pv data by sampling from the distributions defined by using self.forecasts
                as a baseline in NoisyLoadData, NoisyPVData, NoisyGridData.

        :param n_samples: int, default 100
            Number of samples to generate
        :param sampling_args: dict
            Sampling arguments to be passed to NPV.sample() and NL.sample()
        :return:
            samples: list of pd.DataFrame of shape (8760,3)
            list of samples created from sampling from distributions defined in forecasts.
        """
        # TODO: modify this to allow for noise variations, and maybe sample from better distribution
        NPV = self.NPV
        NL = dg.NoisyLoadData(load_data=self.forecasts['load'])
        NG = dg.NoisyGridData(grid_data=self.forecasts['grid'])

        samples = []
        samples = []

        for j in range(n_samples):
            print('Creating sample {}'.format(j))
            pv_forecast = NPV.sample(noise_types=(None, 'gaussian'), **sampling_args)
            load_forecast = NL.sample(**sampling_args)

            grid_forecast = NG.sample()

            sample = pd.concat([pv_forecast, load_forecast, grid_forecast], axis=1)

            truncated_index = min(len(NPV.unmunged_data), len(NL.unmunged_data), len(NG.unmunged_data))
            sample = sample.iloc[:truncated_index]
            samples.append(sample)

        # self.samples = samples
        return samples

    def plot(self, var='load', days_to_plot=(0, 10), original=True, forecast=True, samples=True):
        """
        Function to plot the load, pv, or grid data versus the forecast or original data
        :param var: str, default 'load'
            one of 'load', 'pv', 'grid', which variable to plot
        :param days_to_plot: tuple, len 2. default (0,10)
            defines the days to plot. Plots all hours from days_to_plot[0] to days_to_plot[1]
        :param original: bool, default True
            whether to plot the underlying microgrid data
        :param forecast: bool, default True
            whether to plot the forecast stored in self.forecast
        :param samples: bool, default True
            whether to plot the samples stored in self.samples
        :return:
            None
        """

        if var not in self.forecasts.columns:
            raise ValueError('Cound not find var {} in self.forecasts, should be one of {}'.format(var, self.forecasts.columns))

        indices = slice(24 * days_to_plot[0], 24 * days_to_plot[1])

        if original:
            underlying_data = self._return_underlying_data()
            plt.plot(underlying_data.loc[indices, var].index, underlying_data.loc[indices, var].values,
                     label='original', color='b')
        if forecast:
            plt.plot(self.forecasts.loc[indices, var].index, self.forecasts.loc[indices, var].values, label='forecast',
                     color='r')
        if samples:
            for sample in self.samples:
                plt.plot(sample.loc[indices, var].index, sample.loc[indices, var].values, color='k')

        plt.legend()
        plt.show()

    def run(self, n_samples=100, forecast_steps=None, use_previous_samples=True, verbose=False):
        """
        Runs MPC over a number of samples for to average out for SAA
        :param n_samples: int, default 100
            number of samples to run
        :param forecast_steps: int or None, default None
            number of steps to use in forecast. If None, uses 8760-self.horizon
        :param use_previous_samples: bool, default True
            whether to use previous previous stored in self.samples if they are available
        :param verbose: bool, default False
            verbosity
        :return:
            outputs, list of ControlOutput
                list of ControlOutputs for each sample. See ControlOutput or run_mpc_on_sample for details.
        """
        if self.samples is None or not use_previous_samples:
            self.samples = self.sample_from_forecasts(n_samples=n_samples)

        outputs = []

        t0 = time.time()

        for i, sample in enumerate(self.samples):
            if verbose:
                ratio = 100 * (i / len(self.samples))
                sys.stdout.write("\r Overall progress: %d%%\n" % ratio)
                sys.stdout.write("Cumulative running time: %d minutes" % ((time.time() - t0) / 60))
                sys.stdout.flush()

            output = self.mpc.run_mpc_on_sample(sample, forecast_steps=forecast_steps, verbose=verbose)
            outputs.append(output)

        return outputs

    def determine_optimal_actions(self, outputs, percentile=0.5):
        """
        Given a list of samples from run(), determines which one has cost at the percentile in percentile.

        :param outputs: list of ControlOutput
            list of ControlOutputs from run()
        :param percentile: float, default 0.5
            which percentile to return as optimal.
        :return:
            optimal_output, ControlOutput
                output at optimal percentile
        """

        if percentile < 0. or percentile > 1.:
            raise ValueError('percentile must be in [0,1]')

        partition_val = int(np.floor(len(outputs)*percentile))

        partition = np.partition(outputs, partition_val)

        return partition[partition_val]


class ControlOutput(dict):
    """
    Helper class that allows comparisons between controls by comparing the sum of their resultant costs
    Parameters:
        names: tuple, len 4
            names of each of the dataframes output in MPC
        dfs: tuple, len 4
            DataFrames of the outputs of MPC
        alg_name: str
            Name of the algorithm that produced the output
    Usage: dict-like, e.g.:

     >>>  names = ('action', 'status', 'production', 'cost')
     >>>  dfs = (baseline_linprog_action, baseline_linprog_update_status,
     >>>          baseline_linprog_record_production, baseline_linprog_cost) # From MPC
     >>> M = ControlOutput(names, dfs,'mpc')
     >>> actions = M['action'] # returns the dataframe baseline_linprog_action

    """
    def __init__(self, names, dfs, alg_name):

        names_needed = ('action', 'status', 'production', 'cost')
        if any([needed not in names for needed in names_needed]):
            raise ValueError('Unable to parse names, values are missing')

        super(ControlOutput, self).__init__(zip(names, dfs))
        self.alg_name = alg_name

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return (self['cost'].sum() == other['cost'].sum()).item()

    def __lt__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return (self['cost'].sum() < other['cost'].sum()).item()

    def __gt__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return (self['cost'].sum() > other['cost'].sum()).item()


class ModelPredictiveControl:

    # TODO add a function that runs this on any of the microgrids in the generator to compare last 2/3 baselines
    """
    A class to run Model Predictive Control using the model outlined in the pymgrid paper

    Parameters:
        microgrid: Microgrid.Microgrid
            The underlying microgrid on which MPC will be run

    Attributes:
    --------------
    microgrid: Microgrid.Microgrid
        The underlying microgrid

    horizon: int
        The forecast horizon being used in MPC

    has_genset: bool
        Whether the microgrid has a genset or not

    p_vars: cvxpy.Variable, shape ((7+self.has_genset)*horizon,)
        Vector of all of the controls, at all timesteps. See P in pymgrid paper for details.

    u_genset: None or cvxpy.Variable, shape (self.horizon,)
        Boolean vector variable denoting the status of the genset (on or off) at each timestep if
        the genset exists. If not genset, u_genset = None.

    costs: cvxpy.Parameter, shape ((7+self.has_genset)*self.horizon,)
        Parameter vector of all of the respective costs, at all timesteps. See C in pymgrid paper for details.

    equality_rhs: cvxpy.Parameter, shape (2 * self.horizon,)
        Parameter vector contraining the RHS of the equality constraint equation. See b in pymgrid paper for details.

    inequality_rhs: cvxpy.Parameter, shape (8 * self.horizon,)
        Parameter vector contraining the RHS of the inequality constraint equation. See d in pymgrid paper for details.

    problem: cvxpy.problems.problem.Problem
        The constraint optimization problem to solve


    """
    def __init__(self, microgrid):
        self.microgrid = microgrid
        self.horizon = microgrid.horizon
        if self.microgrid.architecture['genset']==1:
            self.has_genset = True
        else:
            self.has_genset = False

        if self.has_genset:
            self.p_vars = cp.Variable((8*self.horizon,), pos=True)
            self.u_genset = cp.Variable((self.horizon,), boolean=True)
            self.costs = cp.Parameter(8 * self.horizon)
            self.inequality_rhs = cp.Parameter(9 * self.horizon)


        else:
            self.p_vars = cp.Variable((7*self.horizon,), pos=True)
            self.u_genset = None
            self.costs = cp.Parameter(7 * self.horizon, nonneg=True)
            self.inequality_rhs = cp.Parameter(8 * self.horizon)

        self.equality_rhs = cp.Parameter(2 * self.horizon)  # rhs

        parameters = self._parse_microgrid()

        self.problem = self._create_problem(*parameters)

    def _parse_microgrid(self):
        """
        Protected helper function.
        Parses the microgrid in self.microgrid to extract the parameters necessary to run MPC.
        :return:
            eta: float
                battery efficiency
            battery_capacity: float
                battery capacity for normalization
            fuel_cost: float
                fuel cost for the genset
            cost_battery_cycle: float
                cost of cycling the battery
            cost_loss_load: float
                cost of loss load
            p_genset_min: float
                minimum production of the genset
            p_genset_max: float
                maximum production of the genset

        """

        parameters = self.microgrid.parameters

        eta = parameters['battery_efficiency'].values[0]
        battery_capacity = parameters['battery_capacity'].values[0]

        if self.microgrid.architecture['genset'] == 1:
            fuel_cost = parameters['fuel_cost'].values[0]
        else:
            fuel_cost = 0

        cost_battery_cycle = parameters['battery_cost_cycle'].values[0]
        cost_loss_load = parameters['cost_loss_load'].values[0]

        if self.has_genset:
            p_genset_min = parameters['genset_pmin'].values[0] * parameters['genset_rated_power'].values[0]
            p_genset_max = parameters['genset_pmax'].values[0] * parameters['genset_rated_power'].values[0]

        else:
            p_genset_min = 0
            p_genset_max = 0

        return eta, battery_capacity, fuel_cost, cost_battery_cycle, cost_loss_load, p_genset_min, p_genset_max

    def _create_problem(self, eta, battery_capacity, fuel_cost, cost_battery_cycle, cost_loss_load,
                        p_genset_min, p_genset_max):

        """
        Protected, automatically called on initialization.

        Defines the constrainted optimization problem to be stored in self.problem.
        The parameters defined here do not change between timesteps.

        :param eta: float
            battery efficiency
        :param battery_capacity: float
            battery capacity for normalization
        :param fuel_cost: float
            fuel cost for the genset
        :param cost_battery_cycle: float
            cost of cycling the battery
        :param cost_loss_load: float
            cost of loss load
        :param p_genset_min: float
            minimum production of the genset
        :param p_genset_max: float
            maximum production of the genset
        :return :
            problem: cvxpy.problems.problem.Problem
                The constrainted optimization problem to be solved at each step of the MPC.
        """

        delta_t = 1

        # Define matrix Y
        if self.has_genset:
            Y = np.zeros((self.horizon, self.horizon * 8))

            Y[0, 3] = -1.0 * eta * delta_t/battery_capacity
            Y[0, 4] = delta_t / (eta * battery_capacity)
            Y[0, 7] = 1

            gamma = np.zeros(16)
            gamma[7] = -1
            gamma[11] = -1.0 * eta * delta_t/battery_capacity
            gamma[12] = delta_t / (eta * battery_capacity)
            gamma[15] = 1

            for j in range(1, self.horizon):
                start = (j - 1) * 8

                Y[j, start:start + 16] = gamma
        else:
            Y = np.zeros((self.horizon, self.horizon * 7))
            Y[0, 2] = -1.0 * eta * delta_t / battery_capacity
            Y[0, 3] = delta_t / (eta * battery_capacity)
            Y[0, 6] = 1

            gamma = np.zeros(14)
            gamma[6] = -1
            gamma[9] = -1.0 * eta * delta_t/battery_capacity
            gamma[10] = delta_t / (eta * battery_capacity)
            gamma[13] = 1

            for j in range(1, self.horizon):
                start = (j - 1) * 7

                Y[j, start:start + 14] = gamma

        # done with Y
        if self.has_genset:
            X = np.zeros((self.horizon, self.horizon * 8))

            alpha = np.ones(8)
            alpha[2] = -1
            alpha[3] = -1
            alpha[5] = -1
            alpha[7] = 0

            for j in range(self.horizon):
                start = j * 8
                X[j, start:start + 8] = alpha

        else:

            X = np.zeros((self.horizon, self.horizon * 7))

            alpha = np.ones(7)
            alpha[1] = -1
            alpha[2] = -1
            alpha[4] = -1
            alpha[6] = 0

            for j in range(self.horizon):
                start = j * 7
                X[j, start:start + 7] = alpha

        A = np.concatenate((X, Y))  # lhs
        A = csr_matrix(A)

        # Define inequality constraints

        # Inequality lhs
        # This is for one timestep

        C_block = np.zeros((9, 8))
        C_block[0, 0] = 1
        C_block[1, 7] = 1
        C_block[2, 7] = -1
        C_block[3, 3] = 1
        C_block[4, 4] = 1
        C_block[5, 1] = 1
        C_block[6, 2] = 1
        C_block[7, 5] = 1
        C_block[8, 6] = 1

        if not self.has_genset:             # drop the first column if no genset
            C_block = C_block[1:, 1:]

        # For all timesteps
        block_lists = [[C_block if i == j else np.zeros(C_block.shape) for i in range(self.horizon)] for j in
                       range(self.horizon)]
        C = np.block(block_lists)
        C = csr_matrix(C)

        # Inequality rhs

        constraints = [A @ self.p_vars == self.equality_rhs, C @ self.p_vars <= self.inequality_rhs]

        if self.has_genset:
            constraints.extend((p_genset_min * self.u_genset <= self.p_vars[:: 8],
                                self.p_vars[:: 8] <= p_genset_max * self.u_genset))

        # Define  objective
        if self.has_genset:
            cost_vector = np.array([fuel_cost, 0, 0,
                                cost_battery_cycle, cost_battery_cycle, 0, cost_loss_load, 0])
        else:
            cost_vector = np.array([0, 0,
                                    cost_battery_cycle, cost_battery_cycle, 0, cost_loss_load, 0])

        costs_vector = np.concatenate([cost_vector] * self.horizon)

        self.costs.value = costs_vector

        objective = cp.Minimize(self.costs @ self.p_vars)

        return cp.Problem(objective, constraints)

    def _set_parameters(self, load_vector, pv_vector, grid_vector, import_price, export_price,
                        e_max, e_min, p_max_charge, p_max_discharge,
                        p_max_import, p_max_export, soc_0, p_genset_max):

        """
        Protected, called by set_and_solve.
        Sets the time-varying (and some static) parameters in the optimization problem at any given timestep.

        :param load_vector: np.ndarray, shape (self.horizon,)
            load values over the horizon
        :param pv_vector: np.ndarray, shape (self.horizon,)
            pv values over the horizon
        :param grid_vector: np.ndarray, shape (self.horizon,)
            grid values (boolean) over the horizon
        :param import_price: np.ndarray, shape (self.horizon,)
            import prices over the horizon
        :param export_price: np.ndarray, shape (self.horizon,)
            export prices over the horizon
        :param e_max: float
            maximum state of charge of the battery
        :param e_min: float
            minimum state of charge of the battery
        :param p_max_charge: float
            maximum amount of power the battery can charge in one timestep
        :param p_max_discharge: float
            maximum amount of power the battery can discharge in one timestep
        :param p_max_import: float
            maximum amount of power that can be imported in one timestep
        :param p_max_export: float
            maximum amount of power that can be exported in one timestep
        :param soc_0: float
            state of charge of the battery at the timestep just preceding the current horizon
        :return:
            None
        """


        if not isinstance(load_vector,np.ndarray):
            raise TypeError('load_vector must be np.ndarray')
        if not isinstance(pv_vector,np.ndarray):
            raise TypeError('pv_vector must be np.ndarray')
        if not isinstance(grid_vector,np.ndarray):
            raise TypeError('grid_vector must be np.ndarray')
        if not isinstance(import_price,np.ndarray):
            raise TypeError('import_price must be np.ndarray')
        if not isinstance(export_price,np.ndarray):
            raise TypeError('export_price must be np.ndarray')

        if len(load_vector.shape) != 1 and load_vector.shape[0]!=self.horizon:
            raise ValueError('Invalid load_vector, must be of shape ({},)'.format(self.horizon))
        if len(pv_vector.shape) != 1 and pv_vector.shape[0]!=self.horizon:
            raise ValueError('Invalid pv_vector, must be of shape ({},)'.format(self.horizon))
        if len(grid_vector.shape) != 1 and grid_vector.shape[0]!=self.horizon:
            raise ValueError('Invalid grid_vector, must be of shape ({},)'.format(self.horizon))
        if len(import_price.shape) != 1 and import_price.shape[0]!=self.horizon:
            raise ValueError('Invalid import_price, must be of shape ({},)'.format(self.horizon))
        if len(export_price.shape) != 1 and export_price.shape[0]!=self.horizon:
            raise ValueError('Invalid export_price, must be of shape ({},)'.format(self.horizon))

        # Set equality rhs
        equality_rhs_vals = np.zeros(self.equality_rhs.shape)
        equality_rhs_vals[:self.horizon] = load_vector-pv_vector
        equality_rhs_vals[self.horizon] = soc_0
        self.equality_rhs.value = equality_rhs_vals

        # Set inequality rhs
        if self.has_genset:
            inequality_rhs_block = np.array([p_genset_max, e_max, -e_min, p_max_charge, p_max_discharge,
                                             np.nan, np.nan, np.nan, np.nan])
        else:
            inequality_rhs_block = np.array([e_max, -e_min, p_max_charge, p_max_discharge,
                                         np.nan, np.nan, np.nan, np.nan])

        inequality_rhs_vals = np.concatenate([inequality_rhs_block]*self.horizon)

        # set d7-d10
        if self.has_genset:
            inequality_rhs_vals[5::9] = p_max_import * grid_vector
            inequality_rhs_vals[6::9] = p_max_export * grid_vector
            inequality_rhs_vals[7::9] = pv_vector
            inequality_rhs_vals[8::9] = load_vector
            
        else:
            inequality_rhs_vals[4::8] = p_max_import * grid_vector
            inequality_rhs_vals[5::8] = p_max_export * grid_vector
            inequality_rhs_vals[6::8] = pv_vector
            inequality_rhs_vals[7::8] = load_vector

        if np.isnan(inequality_rhs_vals).any():
            raise RuntimeError('There are still nan values in inequality_rhs_vals, something is wrong')

        self.inequality_rhs.value = inequality_rhs_vals

        # Set costs
        if self.has_genset:
            self.costs.value[1::8] = import_price.reshape(-1)
            self.costs.value[2::8] = export_price.reshape(-1)
        else:
            self.costs.value[0::7] = import_price.reshape(-1)
            self.costs.value[1::7] = export_price.reshape(-1)

        if np.isnan(self.costs.value).any():
            raise RuntimeError('There are still nan values in self.costs.value, something is wrong')

    def set_and_solve(self, load_vector, pv_vector, grid_vector, import_price, export_price, e_max, e_min, p_max_charge,
                      p_max_discharge, p_max_import, p_max_export, soc_0, p_genset_max, iteration=None, total_iterations=None):
        """
        Sets the parameters in the problem and then solves the problem.
            Specifically, sets the right-hand sides b and d from the paper of the
            equality and inequality equations, respectively, and the costs vector by calling _set_parameters, then
            solves the problem and returns a control dictionary


        :param load_vector: np.ndarray, shape (self.horizon,)
            load values over the horizon
        :param pv_vector: np.ndarray, shape (self.horizon,)
            pv values over the horizon
        :param grid_vector: np.ndarray, shape (self.horizon,)
            grid values (boolean) over the horizon
        :param import_price: np.ndarray, shape (self.horizon,)
            import prices over the horizon
        :param export_price: np.ndarray, shape (self.horizon,)
            export prices over the horizon
        :param e_max: float
            maximum state of charge of the battery
        :param e_min: float
            minimum state of charge of the battery
        :param p_max_charge: float
            maximum amount of power the battery can charge in one timestep
        :param p_max_discharge: float
            maximum amount of power the battery can discharge in one timestep
        :param p_max_import: float
            maximum amount of power that can be imported in one timestep
        :param p_max_export: float
            maximum amount of power that can be exported in one timestep
        :param soc_0: float
            state of charge of the battery at the timestep just preceding the current horizon
        :param p_genset_max: float
            maximum amount of production of the genset
        :param iteration: int
            Current iteration, used for verbosity
        :param total_iterations:
            Total iterations, used for verbosity
        :return:
            control_dict, dict
            dictionary of the controls of the first timestep, as MPC does.
        """

        self._set_parameters(load_vector, pv_vector, grid_vector, import_price, export_price,
                        e_max, e_min, p_max_charge, p_max_discharge,
                        p_max_import, p_max_export, soc_0, p_genset_max)

        self.problem.solve(warm_start = True)

        if self.problem.status == 'infeasible':
            print(self.problem.status)
            print('Infeasible problem on step {} of {}, retrying with GLPK_MI solver'.format(iteration,total_iterations))
            self.problem.solve(solver = cp.GLPK_MI)
            if self.problem.status == 'infeasible':
                print('Failed again')
            else:
                print('Optimizer found with GLPK_MI solver')

        try:
            if self.has_genset:
                control_dict = {'battery_charge': self.p_vars.value[3],
                                'battery_discharge': self.p_vars.value[4],
                                'genset': self.p_vars.value[0],
                                'grid_import': self.p_vars.value[1],
                                'grid_export': self.p_vars.value[2],
                                'loss_load': self.p_vars.value[6],
                                'pv_consummed': pv_vector[0] - self.p_vars.value[5],
                                'pv_curtailed': self.p_vars.value[5],
                                'load': load_vector[0],
                                'pv': pv_vector[0]}
            else:
                control_dict = {'battery_charge': self.p_vars.value[2],
                                'battery_discharge': self.p_vars.value[3],
                                'grid_import': self.p_vars.value[0],
                                'grid_export': self.p_vars.value[1],
                                'loss_load': self.p_vars.value[5],
                                'pv_consummed': pv_vector[0] - self.p_vars.value[4],
                                'pv_curtailed': self.p_vars.value[4],
                                'load': load_vector[0],
                                'pv': pv_vector[0]}

        except Exception:
            control_dict = None

        return control_dict

    def run_mpc_on_sample(self, sample, forecast_steps=None, verbose=False):
        """
        Runs MPC on a sample over a number of iterations

        :param sample: pd.DataFrame, shape (8760,3)
            sample to run the MPC on. Must contain columns 'load', 'pv', and 'grid'.
        :param forecast_steps: int, default None
            Number of steps to run MPC on. If None, runs over 8760-self.horizon steps
        :param verbose: bool
            Whether to discuss progress
        :return:
            output, ControlOutput
                dict-like containing the DataFrames ('action', 'status', 'production', 'cost'),
                but with an ordering defined via comparing the costs.
        """
        if not isinstance(sample, pd.DataFrame):
            raise TypeError('sample must be of type pd.DataFrame, is {}'.format(type(sample)))
        if sample.shape != (8760, 3):
            sample = sample.iloc[:8760]

        # dataframes, copied API from _baseline_linprog
        baseline_linprog_action = copy(self.microgrid._df_record_control_dict)
        baseline_linprog_update_status = copy(self.microgrid._df_record_state)
        baseline_linprog_record_production = copy(self.microgrid._df_record_actual_production)
        baseline_linprog_cost = copy(self.microgrid._df_record_cost)

        T = len(sample)
        horizon = self.microgrid.horizon

        if forecast_steps is None:
            num_iter = T - horizon
        else:
            assert forecast_steps <= T - horizon, 'forecast steps can\'t look past horizon'
            num_iter = forecast_steps

        t0 = time.time()
        old_control_dict = None

        for i in range(num_iter):

            if verbose and i % 100 == 0:
                ratio = i / num_iter
                sys.stdout.write("\r Progress of current MPC: %d%%\n" % (100 * ratio))
                sys.stdout.flush()

            if self.microgrid.architecture['grid'] == 0:
                temp_grid = np.zeros(horizon)
                price_import = np.zeros(horizon)
                price_export = np.zeros(horizon)
            else:
                temp_grid = sample.loc[i:i + horizon - 1, 'grid'].values
                price_import = self.microgrid._grid_price_import.iloc[i:i + horizon].values
                price_export = self.microgrid._grid_price_export.iloc[i:i + horizon].values

                if temp_grid.shape != price_export.shape and price_export.shape != price_import.shape:
                    raise RuntimeError('I think this is a problem')


            e_min = self.microgrid.parameters['battery_soc_min'].values[0]
            e_max = self.microgrid.parameters['battery_soc_max'].values[0]
            p_max_charge = self.microgrid.parameters['battery_power_charge'].values[0]
            p_max_discharge = self.microgrid.parameters['battery_power_discharge'].values[0]
            p_max_import = self.microgrid.parameters['grid_power_import'].values[0]
            p_max_export = self.microgrid.parameters['grid_power_export'].values[0]
            soc_0 = baseline_linprog_update_status.iloc[-1]['battery_soc']

            if self.has_genset:
                p_genset_max = self.microgrid.parameters['genset_pmax'].values[0] *\
                           self.microgrid.parameters['genset_rated_power'].values[0]
            else:
                p_genset_max = None

            # Solve one step of MPC
            control_dict = self.set_and_solve(sample.loc[i:i + horizon - 1, 'load'].values,
                                              sample.loc[i:i + horizon - 1, 'pv'].values, temp_grid, price_import,
                                              price_export, e_max, e_min, p_max_charge, p_max_discharge, p_max_import,
                                              p_max_export, soc_0, p_genset_max, iteration = i, total_iterations = num_iter)

            if control_dict is not None:
                baseline_linprog_action = self.microgrid._record_action(control_dict, baseline_linprog_action)
                baseline_linprog_record_production = self.microgrid._record_production(control_dict,
                                                                                       baseline_linprog_record_production,
                                                                                       baseline_linprog_update_status)
                old_control_dict = control_dict.copy()

            elif old_control_dict is not None:
                print('Using previous controls')
                baseline_linprog_action = self.microgrid._record_action(old_control_dict, baseline_linprog_action)
                baseline_linprog_record_production = self.microgrid._record_production(old_control_dict,
                                                                                       baseline_linprog_record_production,
                                                                                       baseline_linprog_update_status)
            else:
                raise RuntimeError('Fell through, was unable to solve for control_dict and could not find previous control dict')

            if self.microgrid.architecture['grid'] == 1:
                baseline_linprog_update_status = self.microgrid._update_status(
                    baseline_linprog_record_production.iloc[-1, :].to_dict(),
                    baseline_linprog_update_status,
                    sample.at[i + 1, 'load'],
                    sample.at[i + 1, 'pv'],
                    sample.at[i + 1, 'grid'],
                    self.microgrid._grid_price_import.iloc[i + 1].values[0],
                    self.microgrid._grid_price_export.iloc[i + 1].values[0]
                )

                baseline_linprog_cost = self.microgrid._record_cost(
                    baseline_linprog_record_production.iloc[-1, :].to_dict(),
                    baseline_linprog_cost, self.microgrid._grid_price_import.iloc[i, 0],
                    self.microgrid._grid_price_export.iloc[i, 0])
            else:
                baseline_linprog_update_status = self.microgrid._update_status(
                    baseline_linprog_record_production.iloc[-1, :].to_dict(),
                    baseline_linprog_update_status,
                    sample.at[i + 1, 'load'],
                    sample.at[i + 1, 'pv']
                )
                baseline_linprog_cost = self.microgrid._record_cost(
                    baseline_linprog_record_production.iloc[-1, :].to_dict(),
                    baseline_linprog_cost
                )

        names = ('action', 'status', 'production', 'cost')

        dfs = (baseline_linprog_action, baseline_linprog_update_status,
               baseline_linprog_record_production, baseline_linprog_cost)

        if verbose:
            print('Total time: {} minutes'.format(round((time.time()-t0)/60, 2)))

        return ControlOutput(names, dfs, 'mpc')

    def run_mpc_on_microgrid(self, forecast_steps=None, verbose=False):
        """
        Function that allows MPC to be run on self.microgrid by first parsing its data

        :param forecast_steps: int, default None
            Number of steps to run MPC on. If None, runs over 8760-self.horizon steps
        :param verbose: bool
            Whether to discuss progress
        :return:
            output, ControlOutput
                dict-like containing the DataFrames ('action', 'status', 'production', 'cost'),
                but with an ordering defined via comparing the costs.
        """

        sample = return_underlying_data(self.microgrid)

        return self.run_mpc_on_sample(sample, forecast_steps=forecast_steps, verbose=verbose)

# TODO Set up continuous action space RL for MG

class RuleBasedControl:
    def __init__(self, microgrid):
        if not isinstance(microgrid, Microgrid.Microgrid):
            raise TypeError('microgrid must be of type Microgrid, is {}'.format(type(microgrid)))

        self.microgrid = microgrid

    def _generate_priority_list(self, architecture, parameters, grid_status=0, price_import=0, price_export=0):
        """
        Depending on the architecture of the microgrid and grid related import/export costs, this function generates a
        priority list to be run in the rule based benchmark.
        """
        # compute marginal cost of each resource
        # construct priority list
        # should receive fuel cost and cost curve, price of electricity
        if architecture['grid'] == 1:

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

        p_charge = 0
        p_discharge = 0
        p_import = 0
        p_export = 0
        p_genset = 0
        load_not_matched = 0
        pv_not_curtailed = 0
        self_consumed_pv = 0


        sorted_priority = priority_dict
        min_load = 0
        if self.microgrid.architecture['genset'] == 1:
            #load - pv - min(capa_to_discharge, p_discharge) > 0: then genset on and min load, else genset off
            grid_first = 0
            capa_to_discharge = max(min((status['battery_soc'].iloc[-1] *
                                     parameters['battery_capacity'].values[0]
                                     - parameters['battery_soc_min'].values[0] *
                                     parameters['battery_capacity'].values[0]
                                     ) * parameters['battery_efficiency'].values[0], self.microgrid.battery.p_discharge_max), 0)

            if self.microgrid.architecture['grid'] == 1 and sorted_priority['grid'] < sorted_priority['genset'] and sorted_priority['grid']>0:
                grid_first=1

            if temp_load > pv + capa_to_discharge and grid_first ==0:

                min_load = self.microgrid.parameters['genset_rated_power'].values[0] * self.microgrid.parameters['genset_pmin'].values[0]
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
                    self_consumed_pv = min(temp_load, pv)  # self.maximum_instantaneous_pv_penetration,
                    temp_load = max(0, temp_load - self_consumed_pv)
                    excess_gen = pv - self_consumed_pv
                    pv_not_curtailed = pv_not_curtailed + pv - excess_gen

                if gen == 'battery':

                    capa_to_charge = max(
                        (parameters['battery_soc_max'].values[0] * parameters['battery_capacity'].values[0] -
                         status['battery_soc'].iloc[-1] *
                         parameters['battery_capacity'].values[0]
                         ) / self.microgrid.parameters['battery_efficiency'].values[0], 0)
                    capa_to_discharge = max((status['battery_soc'].iloc[-1] *
                                             parameters['battery_capacity'].values[0]
                                             - parameters['battery_soc_min'].values[0] *
                                             parameters['battery_capacity'].values[0]
                                             ) * parameters['battery_efficiency'].values[0], 0)
                    if temp_load > 0:
                        p_discharge = max(0, min(capa_to_discharge, parameters['battery_power_discharge'].values[0],
                                                temp_load))
                        temp_load = temp_load - p_discharge

                    elif excess_gen > 0:
                        p_charge = max(0, min(capa_to_charge, parameters['battery_power_charge'].values[0],
                                             excess_gen))
                        excess_gen = excess_gen - p_charge

                        pv_not_curtailed = pv_not_curtailed + p_charge

                if gen == 'grid':
                    if temp_load > 0:
                        p_import = temp_load
                        temp_load = 0



                    elif excess_gen > 0:
                        p_export = excess_gen
                        excess_gen = 0

                        pv_not_curtailed = pv_not_curtailed + p_export

                if gen == 'genset':
                    if temp_load > 0:
                        p_genset = temp_load + min_load
                        temp_load = 0
                        min_load = 0

        if temp_load > 0:
            load_not_matched = 1

        control_dict = {'battery_charge': p_charge,
                        'battery_discharge': p_discharge,
                        'genset': p_genset,
                        'grid_import': p_import,
                        'grid_export': p_export,
                        'loss_load': load_not_matched,
                        'pv_consummed': pv_not_curtailed,
                        'pv_curtailed': pv - pv_not_curtailed,
                        'load': load,
                        'pv': pv}

        return control_dict

    def run_rule_based(self, priority_list=0, length=8760):
        # TODO: update this to be able to run on a sample?
        """ This function runs the rule based benchmark over the datasets (load and pv profiles) in the microgrid."""

        baseline_priority_list_action = copy(self.microgrid._df_record_control_dict)
        baseline_priority_list_update_status = copy(self.microgrid._df_record_state)
        baseline_priority_list_record_production = copy(self.microgrid._df_record_actual_production)
        baseline_priority_list_cost = copy(self.microgrid._df_record_cost)

        n = length - self.microgrid.horizon
        print_ratio = n/100

        for i in range(length - self.microgrid.horizon):

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


            if self.microgrid.architecture['grid'] == 1:
                priority_dict = self._generate_priority_list(self.microgrid.architecture, self.microgrid.parameters,
                                                             self.microgrid._grid_status_ts.iloc[i].values[0],
                                                             self.microgrid._grid_price_import.iloc[i].values[0],
                                                             self.microgrid._grid_price_export.iloc[i].values[0])
            else:
                priority_dict = self._generate_priority_list(self.microgrid.architecture, self.microgrid.parameters)

            control_dict = self._run_priority_based(self.microgrid._load_ts.iloc[i].values[0], self.microgrid._pv_ts.iloc[i].values[0],
                                                    self.microgrid.parameters,
                                                    baseline_priority_list_update_status, priority_dict)

            baseline_priority_list_action = self.microgrid._record_action(control_dict,
                                                                      baseline_priority_list_action)

            baseline_priority_list_record_production = self.microgrid._record_production(control_dict,
                                                                                     baseline_priority_list_record_production,
                                                                                     baseline_priority_list_update_status)


            if self.microgrid.architecture['grid']==1:

                baseline_priority_list_update_status = self.microgrid._update_status(
                    baseline_priority_list_record_production.iloc[-1, :].to_dict(),
                    baseline_priority_list_update_status, self.microgrid._load_ts.iloc[i + 1].values[0],
                    self.microgrid._pv_ts.iloc[i + 1].values[0],
                    self.microgrid._grid_status_ts.iloc[i + 1].values[0],
                    self.microgrid._grid_price_import.iloc[i + 1].values[0],
                    self.microgrid._grid_price_export.iloc[i + 1].values[0])


                baseline_priority_list_cost = self.microgrid._record_cost(
                    baseline_priority_list_record_production.iloc[-1, :].to_dict(),
                    baseline_priority_list_cost, self.microgrid._grid_price_import.iloc[i,0], self.microgrid._grid_price_export.iloc[i,0])
            else:

                baseline_priority_list_update_status = self.microgrid._update_status(
                    baseline_priority_list_record_production.iloc[-1, :].to_dict(),
                    baseline_priority_list_update_status, self.microgrid._load_ts.iloc[i + 1].values[0],
                    self.microgrid._pv_ts.iloc[i + 1].values[0])

                baseline_priority_list_cost = self.microgrid._record_cost(
                    baseline_priority_list_record_production.iloc[-1, :].to_dict(),
                    baseline_priority_list_cost)

        names = ('action', 'status', 'production', 'cost')

        dfs = (baseline_priority_list_action, baseline_priority_list_update_status,
               baseline_priority_list_record_production, baseline_priority_list_cost)

        return ControlOutput(names, dfs, 'rbc')


class Benchmarks:
    """
    Class to run various control algorithms. Currently supports MPC and rule-based control.

    Parameters
    -----------
    microgrid: Microgrid.Microgrid
        microgrid on which to run the benchmarks

    Attributes
    -----------
    microgrid, Microgrid.Microgrid
        microgrid on which to run the benchmarks
    mpc_output: ControlOutput or None, default None
        output of MPC if it has been run, otherwise None
    outputs_dict: dict
        Dictionary of the outputs of all run algorithm. Keys are names of algorithms, any or all of 'mpc' or 'rbc' as of now.
    has_mpc_benchmark: bool, default False
        whether the MPC benchmark has been run or not
    rule_based_output: ControlOutput or None, default None
        output of rule basded control if it has been run, otherwise None
    has_rule_based_benchmark: bool, default False
        whether the rule based benchmark has been run or not

    """
    def __init__(self, microgrid):
        if not isinstance(microgrid, Microgrid.Microgrid):
            raise TypeError('microgrid must be of type Microgrid, is {}'.format(type(microgrid)))

        self.microgrid = microgrid
        self.outputs_dict = dict()

        self.mpc_output = None
        self.has_mpc_benchmark = False
        self.rule_based_output = None
        self.has_rule_based_benchmark = False

    def run_mpc_benchmark(self, verbose=False):
        """
        Run the MPC benchmark and store the output in self.mpc_output
        :return:
            None
        """
        MPC = ModelPredictiveControl(self.microgrid)
        self.mpc_output = MPC.run_mpc_on_microgrid(verbose=verbose)
        self.has_mpc_benchmark = True
        self.outputs_dict[self.mpc_output.alg_name] = self.mpc_output

    def run_rule_based_benchmark(self):
        """
        Run the rule based benchmark and store the output in self.rule_based_output
        :return:
            None
        """
        RBC = RuleBasedControl(self.microgrid)
        self.rule_based_output = RBC.run_rule_based()
        self.has_rule_based_benchmark = True
        self.outputs_dict[self.rule_based_output.alg_name] = self.rule_based_output

    def run_benchmarks(self, verbose=False):
        """
        Runs both run_mpc_benchmark() and self.run_mpc_benchmark() and stores the results.
        :param verbose: bool, default False
            Whether to describe benchmarks after running.
        :return:
            None
        """
        self.run_mpc_benchmark(verbose=verbose)
        self.run_rule_based_benchmark()

        if verbose:
            self.describe_benchmarks()

    def describe_benchmarks(self, test_split=False, test_ratio=None, test_index=None):
        """
        Prints the cost of any and all benchmarks that have been run.
        If test_split==True, must have either a test_ratio or a test_index but not both.

        :param test_split: bool, default False
            Whether to report the cost of the partial tail (e.g. the last third steps) or all steps.
        :param test_ratio: float, default None
            If test_split, the percentage of the data set to report on.
        :param test_index: int, default None
            If test_split, the index to split the data into train/test sets
        :return:
            None
        """

        T = len(self.mpc_output['cost'])

        if test_split:
            if test_ratio is None and test_index is None:
                raise ValueError('If test_split, must have either a test_ratio or test_index')
            elif test_ratio is not None and test_index is not None:
                raise ValueError('Cannot have both test_ratio and test_split')
            elif test_ratio is not None and not (0 <= test_ratio <= 1):
                raise ValueError('test_ratio must be in [0,1], is {}'.format(test_ratio))
            elif test_index is not None and test_index > T:
                raise ValueError('test_index cannot be larger than length of output')

        if T != 8736:
            print('length of MPCOutput cost is {}, not 8736, may be invalid'.format(T))

        if not test_split or test_ratio is not None:
            if not test_split:
                test_ratio = 1

            steps = T - int(np.ceil(T * (1 - test_ratio)))
            percent = round(test_ratio * 100, 1)

            if self.has_mpc_benchmark:
                cost = round(self.mpc_output['cost'].iloc[int(np.ceil(T*(1-test_ratio))):].sum().squeeze(),2)
                print('Cost of the last {} steps ({} percent of all steps) using MPC: {}'.format(steps, percent, cost))

            if self.has_rule_based_benchmark:
                cost = round(self.rule_based_output['cost'].iloc[int(np.ceil(T*(1-test_ratio))):].sum().squeeze(),2)
                print('Cost of the last {} steps ({} percent of all steps) using rule-based control: {}'.format(steps, percent, cost))

        else:

            if self.has_mpc_benchmark:
                cost_train = round(self.mpc_output['cost'].iloc[:test_index].sum().squeeze(), 2)
                cost_test = round(self.mpc_output['cost'].iloc[test_index:].sum().squeeze(), 2)

                print('Test set cost using MPC: {}'.format(cost_test))
                print('Train set cost using MPC: {}'.format(cost_train))

            if self.has_rule_based_benchmark:
                cost_train = round(self.rule_based_output['cost'].iloc[:test_index].sum().squeeze(),
                             2)
                cost_test = round(self.rule_based_output['cost'].iloc[test_index:].sum().squeeze(),
                             2)

                print('Test set cost using MPC: {}'.format(cost_test))
                print('Train set cost using MPC: {}'.format(cost_train))


if __name__=='__main__':

    m_gen = MicrogridGenerator.MicrogridGenerator(nb_microgrid=100,
                                  path='/Users/ahalev/Dropbox/Avishai/gradSchool/internships/totalInternship/pymgrid_git')
    m_gen = m_gen.load('pymgrid25')
    microgrid = m_gen.microgrids[0]

    RBC = RuleBasedControl(microgrid)
    rbc_output = RBC.run_rule_based()
