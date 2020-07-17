from utils import DataGenerator as dg
from pymgrid import Microgrid
import pandas as pd
import numpy as np
from copy import copy
import time, sys
from matplotlib import pyplot as plt
import cvxpy as cp
from scipy.sparse import csr_matrix
import logging
import fenics

logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SampleAverageApproximation:
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

        # TODO: Then aggregate controls: 1) use median, 2) learn a function
        # TODO: Use these in sample average approximation to learn a policy
        # You can use scipy optimize to do this
        # No you should use cvxpy to do this

    def create_forecasts(self):
        # TODO: modify this to allow for different amounts of noise in forecasts

        pv_forecast = self.NPV.sample()
        load_forecast = self.NL.sample()

        if self.microgrid.architecture['grid'] != 0:
            grid_forecast = self.NG.sample()
        else:
            grid_forecast = pd.Series(data=[0] * len(self.microgrid._load_ts), name='grid')

        return pd.concat([pv_forecast, load_forecast, grid_forecast], axis=1)

    def _return_underlying_data(self):
        pv_data = self.microgrid._pv_ts
        load_data = self.microgrid._load_ts

        pv_data = pv_data[pv_data.columns[0]]
        load_data = load_data[load_data.columns[0]]
        pv_data.name = 'pv'
        load_data.name = 'load'

        if self.microgrid.architecture['grid'] != 0:
            grid_data = self.microgrid._grid_status_ts
            if isinstance(grid_data, pd.DataFrame):
                grid_data = grid_data[grid_data.columns[0]]
                grid_data.name = 'grid'
            elif isinstance(grid_data, pd.Series):
                grid_data.name = 'grid'
            else:
                raise RuntimeError('Unable to handle microgrid._grid_status_ts of type {}.'.format(type(grid_data)))
        else:
            grid_data = pd.Series(data=[0] * len(self.microgrid._load_ts), name='grid')

        return pd.concat([pv_data, load_data, grid_data], axis=1)

    def sample_from_forecasts(self, n_samples=100, **sampling_args):
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

        self.samples = samples
        return samples

    def plot(self, var='load', days_to_plot=(0, 10), original=True, forecast=True, samples=True):
        """

        :param var:
        :param days_to_plot:
        :param original:
        :param forecast:
        :param samples:
        :return:
        """

        indices = slice(24 * days_to_plot[0], 24 * days_to_plot[1])

        if original:
            underlying_data = self._return_underlying_data()
            plt.plot(underlying_data.loc[indices, var].index, underlying_data.loc[indices, var].values,
                     label='original', color='b')
        if forecast:
            # print('index')
            # print(self.forecasts.index)
            # print('columns')
            # print(self.forecasts.columns)
            plt.plot(self.forecasts.loc[indices, var].index, self.forecasts.loc[indices, var].values, label='forecast',
                     color='r')
        if samples:
            for sample in self.samples:
                plt.plot(sample.loc[indices, var].index, sample.loc[indices, var].values, color='k')

        plt.legend()
        plt.show()

    def run_mpc_on_sample_old(self, sample, forecast_steps=None, verbose=False):
        """
        Deprecated.
        Runs model predictive control on the load, pv, grid sample in 'sample'.
        Uses a call of microgrid._mpc_lin_prog_cvxpy() to do this, which is inefficient.

        :param sample: pd.DataFrame, shape (8760, 3)
            DataFrame of load, pv, and grid data.
        :param forecast_steps: int
            number of steps to compute the controls for
        :param verbose: bool
            verbosity
        :return:
            baseline_linprog_action: pd.DataFrame
                actions

            baseline_linprog_update_status: pd.DataFrame
                update status

            baseline_linprog_record_production: pd.DataFrame
                recorded production

            baseline_linprog_cost: pd.DataFrame
                costs
        """
        if not isinstance(sample, pd.DataFrame):
            raise TypeError('sample must be of type pd.DataFrame, is {}'.format(type(sample)))
        if sample.shape != (8760, 3):
            raise ValueError('sample must have shape (8760,3), is of shape {}'.format(sample.shape))

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

        for i in range(num_iter):

            if verbose and i % 100 == 0:
                ratio = i / num_iter
                sys.stdout.write("\r Progress of current MPC: %d%%\n" % (100 * ratio))
                # sys.stdout.write("Cumulative running time: %d minutes" % ((time.time()-t0)/60))
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

            ## TODO: Figure out this function call
            control_dict = self.microgrid._mpc_lin_prog_cvxpy(self.microgrid.parameters,
                                                              sample.loc[i:i + horizon - 1, 'load'].values,
                                                              sample.loc[i:i + horizon - 1, 'pv'].values,
                                                              temp_grid, baseline_linprog_update_status,
                                                              price_import, price_export, horizon)

            baseline_linprog_action = self.microgrid._record_action(control_dict, baseline_linprog_action)
            baseline_linprog_record_production = self.microgrid._record_production(control_dict,
                                                                                   baseline_linprog_record_production,
                                                                                   baseline_linprog_update_status)

            # print('sample:')
            # print(sample.loc[i+1,'load'])
            # print(type(sample.loc[i+1,'load']))
            # print('with at', sample.at[i+1,'load'])
            # print('underlying:')
            # print(self.microgrid._load_ts.iloc[i+1].values[0])
            # print(type(self.microgrid._load_ts.iloc[i+1].values[0]))

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

        return dict(zip(names, dfs))

    def run(self, n_samples=100, forecast_steps=None, use_previous_samples=True, verbose=False):

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


        # TODO: use these outputs to determine an optimal action
        return outputs

    def determine_optimal_actions(self, outputs, percentile = 0.5):
        # naive way

        if percentile<0. or percentile>1.:
            raise ValueError('percentile must be in [0,1]')

        partition_val = int(np.floor(len(outputs)*percentile))

        partition = np.partition(outputs, partition_val)

        return partition[partition_val]


class MPCOutput(dict):
    """
    Helper class that allows comparisons between controls by comparing the sum of their resultant costs
    Parameters:
        names: tuple, len 4
            names of each of the dataframes output in MPC
        dfs: tuple, len 4
            DataFrames of the outputs of MPC
    """
    def __init__(self, names, dfs):
        names_needed = ('action', 'status', 'production', 'cost')

        if any([needed not in names for needed in names_needed]):
            raise ValueError('Unable to parse names, is missing values')

        super(MPCOutput, self).__init__(zip(names, dfs))

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

    p_vars: cvxpy.Variable, shape ((7+has_genset)*horizon,)
        Vector of all of the controls, at all timesteps. See P in pymgrid paper for details.

    u_genset: None or cvxpy.Variable, shape (self.horizon,)
        Boolean vector variable denoting the status of the genset (on or off) at each timestep if
        the genset exists. If not genset, u_genset = None.

    costs: cvxpy.Parameter, shape ((7+has_genset)*horizon,)
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
            battery_capacity: float
            fuel_cost: float
            cost_battery_cycle: float
            cost_loss_load: float
            p_genset_min: float
            p_genset_max: float

            TODO describe these parameters

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
        :param battery_capacity: float
        :param fuel_cost: float
        :param cost_battery_cycle: float
        :param cost_loss_load: float
        :param p_genset_min: float
        :param p_genset_max: float
        :return:
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

        print('Condition number of A', np.linalg.cond(A))


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

        print('Condition number of  C', np.linalg.cond(C))

        C = csr_matrix(C)

        # Inequality rhs
        # inequality_rhs = cp.Parameter(10 * self.horizon)

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

    def _set_parameters(self, load_vector: np.ndarray, pv_vector, grid_vector, import_price, export_price,
                        e_max, e_min, p_max_charge, p_max_discharge,
                        p_max_import, p_max_export, soc_0, p_genset_max):

        # TODO: finish describing the parameters

        """
        Protected, called by set_and_solve.
        Sets the time-varying (and some static) parameters in the optimization problem at any given timestep.

        :param load_vector: np.ndarray, shape (self.horizon,)

        :param pv_vector: np.ndarray, shape (self.horizon,)
        :param grid_vector: np.ndarray, shape (self.horizon,)
        :param import_price: np.ndarray, shape (self.horizon,)
        :param export_price: np.ndarray, shape (self.horizon,)
        :param e_max: float
        :param e_min: float
        :param p_max_charge: float
        :param p_max_discharge: float
        :param p_max_import: float
        :param p_max_export: float
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



        :param load_vector:
        :param pv_vector:
        :param grid_vector:
        :param import_price:
        :param export_price:
        :param e_max:
        :param e_min:
        :param p_max_charge:
        :param p_max_discharge:
        :param p_max_import:
        :param p_max_export:
        :param soc_0:
        :param p_genset_max:
        :param iteration:
        :param total_iterations:
        :return:
            control_dict, dict
            dictionary of the controls of the first timestep, as MPC does.
        """

        self._set_parameters(load_vector, pv_vector, grid_vector, import_price, export_price,
                        e_max, e_min, p_max_charge, p_max_discharge,
                        p_max_import, p_max_export, soc_0, p_genset_max)

        self.problem.solve(warm_start = True)
        # print('iters', self.problem.solver_stats.num_iters)

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
            output, MPCOutput
                dict-like containing the DataFrames ('action', 'status', 'production', 'cost'),
                but with an ordering defined via comparing the costs.
        """
        if not isinstance(sample, pd.DataFrame):
            raise TypeError('sample must be of type pd.DataFrame, is {}'.format(type(sample)))
        if sample.shape != (8760, 3):
            sample = sample.iloc[:8760]
            # raise ValueError('sample must have shape (8760,3), is of shape {}'.format(sample.shape))

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
                # sys.stdout.write("Cumulative running time: %d minutes" % ((time.time()-t0)/60))
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

            # control_dict = self.microgrid._mpc_lin_prog_cvxpy(self.microgrid.parameters,
            #                                                   sample.loc[i:i + horizon - 1, 'load'].values,
            #                                                   sample.loc[i:i + horizon - 1, 'pv'].values,
            #                                                   temp_grid, baseline_linprog_update_status,
            #                                                   price_import, price_export, horizon)

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

        return MPCOutput(names, dfs)


class ModelPredictiveControlOld:
    # TODO write docstrings
    def __init__(self, microgrid):
        self.microgrid = microgrid
        self.horizon = microgrid.horizon

        self.p_vars = cp.Variable((8 * self.horizon,), pos=True)
        self.u_genset = cp.Variable((self.horizon,), boolean=True)

        self.equality_rhs = cp.Parameter(2 * self.horizon)  # rhs
        self.inequality_rhs = cp.Parameter(10 * self.horizon)
        self.costs = cp.Parameter(8 * self.horizon)

        parameters = self._parse_microgrid()

        self.problem = self._create_problem(*parameters)

    def _parse_microgrid(self):

        parameters = self.microgrid.parameters

        eta = parameters['battery_efficiency'].values[0]
        battery_capacity = parameters['battery_capacity'].values[0]

        if self.microgrid.architecture['genset'] == 1:
            fuel_cost = parameters['fuel_cost'].values[0]
        else:
            fuel_cost = 0

        cost_battery_cycle = parameters['battery_cost_cycle'].values[0]
        cost_loss_load = parameters['cost_loss_load'].values[0]
        p_genset_min = parameters['genset_pmin'].values[0] * parameters['genset_rated_power'].values[0]
        p_genset_max = parameters['genset_pmax'].values[0] * parameters['genset_rated_power'].values[0]

        return eta, battery_capacity, fuel_cost, cost_battery_cycle, cost_loss_load, p_genset_min, p_genset_max

    def _create_problem(self, eta, battery_capacity, fuel_cost, cost_battery_cycle, cost_loss_load,
                        p_genset_min, p_genset_max):

        # TODO set up a conditional so there are no genset constraints and vars if genset is off

        delta_t = 1

        # Define matrix Y
        Y = np.zeros((self.horizon, self.horizon * 8))

        Y[0, 3] = -1.0 * eta * delta_t / battery_capacity
        Y[0, 4] = delta_t / (eta * battery_capacity)
        Y[0, 7] = 1

        gamma = np.zeros(16)
        gamma[7] = -1
        gamma[11] = -1.0 * eta * delta_t / battery_capacity
        gamma[12] = delta_t / (eta * battery_capacity)
        gamma[15] = 1

        for j in range(1, self.horizon):
            start = (j - 1) * 8

            Y[j, start:start + 16] = gamma

        X = np.zeros((self.horizon, self.horizon * 8))

        alpha = np.ones(8)
        alpha[2] = -1
        alpha[3] = -1
        alpha[5] = -1
        alpha[7] = 0

        for j in range(self.horizon):
            start = j * 8
            X[j, start:start + 8] = alpha

        A = np.concatenate((X, Y))  # lhs
        A = csr_matrix(A)
        # equality_rhs = cp.Parameter(2 * self.horizon) # rhs
        # Define inequality constraints

        # Inequality lhs
        # This is for one timestep
        C_block = np.zeros((10, 8))
        C_block[0, 7] = 1
        C_block[1, 7] = -1
        C_block[2, 3] = 1
        C_block[3, 3] = -1
        C_block[4, 4] = 1
        C_block[5, 4] = -1
        C_block[6, 1] = 1
        C_block[7, 2] = 1
        C_block[8, 5] = 1
        C_block[9, 6] = 1

        # For all timesteps
        block_lists = [[C_block if i == j else np.zeros(C_block.shape) for i in range(self.horizon)] for j in
                       range(self.horizon)]
        C = np.block(block_lists)
        C = csr_matrix(C)

        # Inequality rhs
        # inequality_rhs = cp.Parameter(10 * self.horizon)

        constraints = [A @ self.p_vars == self.equality_rhs, C @ self.p_vars <= self.inequality_rhs,
                       p_genset_min * self.u_genset <= self.p_vars[:: 8],
                       self.p_vars[:: 8] <= p_genset_max * self.u_genset]

        # Define  objective

        cost_vector = np.array([fuel_cost, 0, 0,
                                cost_battery_cycle, cost_battery_cycle, 0, cost_loss_load, 0])
        costs_vector = np.concatenate([cost_vector] * self.horizon)
        self.costs.value = costs_vector

        objective = cp.Minimize(self.costs @ self.p_vars)

        return cp.Problem(objective, constraints)

    def _set_parameters(self, load_vector, pv_vector, grid_vector, import_price, export_price,
                        e_max, e_min, p_max_charge, p_min_charge, p_max_discharge, p_min_discharge,
                        p_max_import, p_max_export, soc_0):

        if not isinstance(load_vector, np.ndarray):
            raise TypeError('load_vector must be np.ndarray')
        if not isinstance(pv_vector, np.ndarray):
            raise TypeError('pv_vector must be np.ndarray')
        if not isinstance(grid_vector, np.ndarray):
            raise TypeError('grid_vector must be np.ndarray')
        if not isinstance(import_price, np.ndarray):
            raise TypeError('import_price must be np.ndarray')
        if not isinstance(export_price, np.ndarray):
            raise TypeError('export_price must be np.ndarray')

        if len(load_vector.shape) != 1 and load_vector.shape[0] != self.horizon:
            raise ValueError('Invalid load_vector, must be of shape ({},)'.format(self.horizon))
        if len(pv_vector.shape) != 1 and pv_vector.shape[0] != self.horizon:
            raise ValueError('Invalid pv_vector, must be of shape ({},)'.format(self.horizon))
        if len(grid_vector.shape) != 1 and grid_vector.shape[0] != self.horizon:
            raise ValueError('Invalid grid_vector, must be of shape ({},)'.format(self.horizon))
        if len(import_price.shape) != 1 and import_price.shape[0] != self.horizon:
            raise ValueError('Invalid import_price, must be of shape ({},)'.format(self.horizon))
        if len(export_price.shape) != 1 and export_price.shape[0] != self.horizon:
            raise ValueError('Invalid export_price, must be of shape ({},)'.format(self.horizon))

        # Set equality rhs
        equality_rhs_vals = np.zeros(self.equality_rhs.shape)
        equality_rhs_vals[:self.horizon] = load_vector - pv_vector
        equality_rhs_vals[self.horizon] = soc_0
        self.equality_rhs.value = equality_rhs_vals

        # Set inequality rhs
        inequality_rhs_block = np.array(
            [e_max, -e_min, p_max_charge, -p_min_charge, p_max_discharge, -p_min_discharge,
             np.nan, np.nan, np.nan, np.nan])

        inequality_rhs_vals = np.concatenate([inequality_rhs_block] * self.horizon)

        # set c7-c10
        inequality_rhs_vals[6::10] = p_max_import * grid_vector
        inequality_rhs_vals[7::10] = p_max_export * grid_vector
        inequality_rhs_vals[8::10] = pv_vector
        inequality_rhs_vals[9::10] = load_vector

        if np.isnan(inequality_rhs_vals).any():
            raise RuntimeError('There are still nan values in inequality_rhs_vals, something is wrong')

        self.inequality_rhs.value = inequality_rhs_vals

        # Set costs

        self.costs.value[1::8] = import_price.reshape(-1)
        self.costs.value[2::8] = export_price.reshape(-1)

        if np.isnan(self.costs.value).any():
            raise RuntimeError('There are still nan values in self.costs.value, something is wrong')

    def set_and_solve(self, load_vector, pv_vector, grid_vector, import_price, export_price,
                      e_max, e_min, p_max_charge, p_min_charge, p_max_discharge, p_min_discharge,
                      p_max_import, p_max_export, soc_0):

        self._set_parameters(load_vector, pv_vector, grid_vector, import_price, export_price,
                             e_max, e_min, p_max_charge, p_min_charge, p_max_discharge, p_min_discharge,
                             p_max_import, p_max_export, soc_0)

        self.problem.solve()

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

        return control_dict

    def run_mpc_on_sample(self, sample, forecast_steps=None, verbose=False):
        if not isinstance(sample, pd.DataFrame):
            raise TypeError('sample must be of type pd.DataFrame, is {}'.format(type(sample)))
        if sample.shape != (8760, 3):
            sample = sample.iloc[:8760]
            # raise ValueError('sample must have shape (8760,3), is of shape {}'.format(sample.shape))

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

        for i in range(num_iter):

            if verbose and i % 100 == 0:
                ratio = i / num_iter
                sys.stdout.write("\r Progress of current MPC: %d%%\n" % (100 * ratio))
                # sys.stdout.write("Cumulative running time: %d minutes" % ((time.time()-t0)/60))
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

            # control_dict = self.microgrid._mpc_lin_prog_cvxpy(self.microgrid.parameters,
            #                                                   sample.loc[i:i + horizon - 1, 'load'].values,
            #                                                   sample.loc[i:i + horizon - 1, 'pv'].values,
            #                                                   temp_grid, baseline_linprog_update_status,
            #                                                   price_import, price_export, horizon)

            e_min = self.microgrid.parameters['battery_soc_min'].values[0]
            e_max = self.microgrid.parameters['battery_soc_max'].values[0]
            p_max_charge = self.microgrid.parameters['battery_power_charge'].values[0]
            p_max_discharge = self.microgrid.parameters['battery_power_discharge'].values[0]
            p_min_charge = 0  # TODO this is related to an unnecessary constraint, remove the constraint
            p_min_discharge = 0  # TODO this is related to an unnecessary constraint, remove the constraint
            p_max_import = self.microgrid.parameters['grid_power_import'].values[0]
            p_max_export = self.microgrid.parameters['grid_power_export'].values[0]
            soc_0 = baseline_linprog_update_status.iloc[-1]['battery_soc']

            control_dict = self.set_and_solve(sample.loc[i:i + horizon - 1, 'load'].values,
                                              sample.loc[i:i + horizon - 1, 'pv'].values,
                                              temp_grid, price_import, price_export, e_max, e_min, p_max_charge,
                                              p_min_charge, p_max_discharge, p_min_discharge, p_max_import,
                                              p_max_export,
                                              soc_0)

            baseline_linprog_action = self.microgrid._record_action(control_dict, baseline_linprog_action)
            baseline_linprog_record_production = self.microgrid._record_production(control_dict,
                                                                                   baseline_linprog_record_production,
                                                                                   baseline_linprog_update_status)

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
            print('Total time: {} minutes'.format(round((time.time() - t0) / 60, 2)))

        return dict(zip(names, dfs))


if __name__=='__main__':
    import pymgrid.MicrogridGenerator as mg

    m_gen = mg.MicrogridGenerator(nb_microgrid=100,
                                  path='/Users/ahalev/Dropbox/Avishai/gradSchool/internships/totalInternship/pymgrid_git')
    m_gen = m_gen.load('pymgrid25')

    sampling_args = dict(load_variance_scale=1.2, noise_params=(None, {'std_ratio': 0.3}))

    microgrid = m_gen.microgrids[1]

    if microgrid.architecture['genset'] != 1:
        raise Exception('no')

    SAA = SampleAverageApproximation(microgrid)
    SAA.sample_from_forecasts(n_samples=5, **sampling_args)

    outputs = SAA.run(verbose=True)


