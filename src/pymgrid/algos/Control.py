
"""
Copyright 2020 Total S.A.
Authors:Gonzague Henri <gonzague.henri@total.com>, Avishai Halev <>
Permission to use, modify, and distribute this software is given under the
terms of the pymgrid License.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2020/08/27 08:04 $
Gonzague Henri
"""
"""
<pymgrid is a Python library to simulate microgrids>
Copyright (C) <2020> <Total S.A.>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
from pymgrid.utils import DataGenerator as dg
# from pymgrid import Microgrid
import pandas as pd
import numpy as np
from copy import deepcopy
import time, sys
from matplotlib import pyplot as plt
import cvxpy as cp
from scipy.sparse import csr_matrix
import operator

np.random.seed(0)

# TODO commented type checks to test imports in Microgrid/Benchmarks

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

    Parameters

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
    def __init__(self, microgrid, control_duration=8760, **forecast_args):
        if control_duration > 8760:
            raise ValueError('control_duration must be less than 8760')

        # if not isinstance(microgrid, Microgrid.Microgrid):
        #     raise TypeError('microgrid must be of type \'pymgrid.Microgrid.Microgrid\', is {}'.format(type(microgrid)))

        self.microgrid = microgrid
        self.control_duration = control_duration
        self.mpc = ModelPredictiveControl(self.microgrid)

        self.NPV = dg.NoisyPVData(pv_data=self.microgrid._pv_ts)
        self.NL = dg.NoisyLoadData(load_data=self.microgrid._load_ts)
        if self.microgrid.architecture['grid'] != 0:
            self.NG = dg.NoisyGridData(grid_data=self.microgrid._grid_status_ts)
        else:
            self.NG = None

        self.underlying_data = return_underlying_data(self.microgrid)
        self.forecasts = self.create_forecasts(**forecast_args)
        self.samples = None

        # TODO: Then aggregate controls: 2) learn a function
        # TODO: Use these in sample average approximation to learn a policy

    def create_forecasts(self, pv_args=None, load_args=None, preset_to_use=None, print_mape=False, **forecast_args):
        """
        Creates pv, load, and grid forecasts that are then used to create samples.

        :return:
            df, pd.DataFrame,  shape (8760,3)
                DataFrame with columns of 'pv', 'load', and 'grid', containing values for each at all 8760 timesteps
        """
        if pv_args is None and load_args is None and preset_to_use is not None:
            print('Using preset forecast arguments')
            args = ForecastArgSet(preset_to_use=preset_to_use)
            pv_args = args['pv_args']
            load_args = args['load_args']
        else:
            if pv_args is None:
                pv_args = dict()
            if load_args is None:
                load_args = dict()

        pv_forecast = self.NPV.sample(**pv_args)
        load_forecast = self.NL.sample(**load_args)

        if self.microgrid.architecture['grid'] != 0:
            grid_forecast = self.NG.sample()
        else:
            grid_forecast = pd.Series(data=[0] * len(self.microgrid._load_ts), name='grid')

        forecast = pd.concat([pv_forecast, load_forecast, grid_forecast], axis=1)

        if print_mape:
            mape = self.validate_forecasts(forecasts=forecast, aggregate=True)
            print('MAPE: {}'.format(mape))

        if hasattr(self, 'forecasts'):
            self.forecasts = forecast
        else:
            return forecast

    def test_args(self, iters_per_set = 3):
        num_pv_noise_params_0 = 3
        num_pv_std_ratio = 3
        num_load_variance_scale = 3
        max_load_var_scale = 2.
        max_pv_std_ratio = 0.5
        num_push_peak_ratio = 3
        num_push_individual_ratio = 3

        forecast_args = ForecastArgs(num_pv_noise_params_0, num_pv_std_ratio, num_load_variance_scale,
                                     num_push_peak_ratio, num_push_individual_ratio,
                                     max_load_var_scale=max_load_var_scale, max_pv_std_ratio=max_pv_std_ratio)

        for j, arg_set in enumerate(forecast_args.param_sets):
            for k in range(iters_per_set):
                print('iter {}.{}'.format(j, k))
                forecasts = self.create_forecasts(**arg_set)
                mape = self.validate_forecasts(forecasts, aggregate=True)
                arg_set.update_with_mape(mape)

        return forecast_args

    def validate_forecasts(self, forecasts=None, aggregate=False):

        if forecasts is None:
            forecasts = self.forecasts

        mape_vals = dict()

        for col in ('pv', 'load'):
            mape_vals[col] = self.mape(self.underlying_data[col], forecasts[col])

        if aggregate:
            return np.sqrt(np.mean(np.array(list(mape_vals.values()))**2))
        return mape_vals

    def mape(self, actual_vals, forecast_vals):
        if isinstance(actual_vals.squeeze(), pd.Series):
            actual_vals = actual_vals.to_numpy()
        elif isinstance(actual_vals, np.ndarray):
            pass
        else:
            raise TypeError('actual_vals must be squeezable to single column, has shape {}'.format(actual_vals.shape))
        if isinstance(forecast_vals.squeeze(), pd.Series):
            forecast_vals = forecast_vals.to_numpy()
        elif isinstance(forecast_vals, np.ndarray):
            pass
        else:
            raise TypeError('forecast_vals must be squeezable to single column, has shape {}'.format(forecast_vals.shape))

        ratios = np.abs(((actual_vals-forecast_vals)/actual_vals))
        mape = np.mean(ratios[~np.isnan(ratios)])

        return mape

    def sample_from_forecasts(self, n_samples=10, **sampling_args):
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

        if 'noise_types' not in sampling_args.keys():
            sampling_args['noise_types'] = (None,'gaussian')

        for j in range(n_samples):
            print('Creating sample {}'.format(j))
            pv_forecast = NPV.sample(**sampling_args)
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
            plt.plot(self.underlying_data.loc[indices, var].index, self.underlying_data.loc[indices, var].values,
                     label='original', color='b')
        if forecast:
            plt.plot(self.forecasts.loc[indices, var].index, self.forecasts.loc[indices, var].values, label='forecast',
                     color='r')
        if samples:
            for sample in self.samples:
                plt.plot(sample.loc[indices, var].index, sample.loc[indices, var].values, color='k')

        plt.legend()
        plt.show()

    def run(self, n_samples=10, forecast_steps=None, optimal_percentile=0.5, use_previous_samples=True, verbose=False, **kwargs):
        """
        Runs MPC over a number of samples for to average out for SAA
        :param n_samples: int, default 25
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
            self.samples = self.sample_from_forecasts(n_samples=n_samples, **kwargs)

        outputs = []

        t0 = time.time()

        output = self.run_mpc_on_group(self.samples, forecast_steps=forecast_steps,
                                        optimal_percentile=optimal_percentile, verbose=verbose)

        if verbose:
            print('Running time: {}'.format(round(time.time()-t0)))

        return output

    def determine_optimal_actions(self, outputs=None, percentile=0.5, verbose=False):
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

        if verbose:
            sorted_outputs = np.sort(outputs)
            selected_output = partition[partition_val]
            print()
            for j, output in enumerate(sorted_outputs):
                print('Output {}, cost: {}, battery charge {}, discharge {}:'.format(
                    j, round(output.cost,2) , round(output.first_dict['battery_charge'],2), round(output.first_dict['battery_discharge'],2)))
                if output is selected_output:
                    print('Selected output {} with percentile {}'.format(j, percentile))

        return partition[partition_val]

    def run_mpc_on_group(self, samples, forecast_steps=None, optimal_percentile=0.5, verbose=False):
        columns_needed = ('pv', 'load', 'grid')

        output = ControlOutput(alg_name='saa', empty=True, microgrid=self.microgrid)

        T = min([len(sample) for sample in samples])
        if forecast_steps is None:
            forecast_steps = T-self.microgrid.horizon
        elif forecast_steps>T-self.microgrid.horizon:
            raise ValueError('forecast steps must be less than length of samples minus horizon')

        for j in range(forecast_steps):
            if verbose:
                print('iter {}'.format(j))

            horizon_outputs = []

            for sample in samples:
                if not isinstance(sample, pd.DataFrame):
                    raise TypeError('samples must be pd.DataFrame')
                if not all([needed in sample.columns.values for needed in columns_needed]):
                    raise KeyError('samples must contain columns {}, currently contains {}'.format(
                        columns_needed, sample.columns.values))

                sample.iloc[j] = self.underlying_data.iloc[j]  # overwrite with actual data

                # TODO return controls of all steps in horizon
                # TODO then, pick 0th step controls of sample with 'percentile' cost over horizon

                horizon_output = self.mpc.mpc_single_step(sample, output, j)

                horizon_outputs.append(horizon_output)

            # return horizon_outputs

            optimal_output = self.determine_optimal_actions(outputs=horizon_outputs, percentile=optimal_percentile)
            output.append(optimal_output, actual_load=self.underlying_data.loc[j,'load'],
                          actual_pv=self.underlying_data.loc[j,'pv'],
                          actual_grid=self.underlying_data.loc[j,'grid'])

        return output

    def run_deterministic_on_forecast(self, forecast_steps=None, verbose=False):

        sample = self.forecasts.copy()
        output = ControlOutput(alg_name='mpc', empty=True, microgrid=self.microgrid)

        T = len(sample)

        if forecast_steps is None:
            forecast_steps = T - self.microgrid.horizon
        elif forecast_steps > T - self.microgrid.horizon:
            raise ValueError('forecast steps must be less than length of samples minus horizon')

        for j in range(forecast_steps):
            if verbose:
                print('iter {}'.format(j))

                sample.iloc[j] = self.underlying_data.iloc[j]  # overwrite with actual data

                # TODO return controls of all steps in horizon
                # TODO then, pick 0th step controls of sample with 'percentile' cost over horizon

                horizon_output = self.mpc.mpc_single_step(sample, output, j)

                output.append(horizon_output)

            return output

class ForecastArgs:
    def __init__(self, num_pv_noise_params_0, num_pv_std_ratio, num_load_variance_scale, num_push_peak_ratio,
                 num_push_individual_ratio, max_load_var_scale=2., max_pv_std_ratio=0.5):

        pv_params = self.pv_parameters(num_pv_noise_params_0, num_pv_std_ratio, num_push_peak_ratio, num_push_individual_ratio,
                                       max_std_ratio=max_pv_std_ratio)
        load_params = self.load_parameters(num_load_variance_scale, max_load_var_scale=max_load_var_scale)

        self.param_sets = self.combine_sets(pv_params, load_params)

    def pv_parameters(self,num_noise_params_0, num_std_ratio, num_push_peak_ratio,  num_push_individual_ratio,
                      max_std_ratio=0.5):

        pv_params = []
        for individual_ratio in np.linspace(0,1,num_push_individual_ratio):
            for peak_ratio in np.linspace(0,1,num_push_peak_ratio):
                for std_ratio in np.linspace(0,max_std_ratio, num_std_ratio):
                    for lower in np.linspace(0,1,num_noise_params_0):
                        for upper in np.linspace(1,lower,num_noise_params_0):
                            if upper >= lower:
                                pv_params.append(dict(noise_params=(dict(lower=lower, upper=upper), dict(std_ratio=std_ratio)),
                                                      push_peak_val=True, push_peak_ratio=peak_ratio,
                                                      push_individual_vals=True, push_individual_ratio=individual_ratio))
                            else:
                                print('upper not geq lower')

        return pv_params

    def load_parameters(self,num_load_variance_scale, max_load_var_scale=2.):
        load_params = []
        for var_scale in np.linspace(0,max_load_var_scale,num_load_variance_scale):
            load_params.append(dict(load_variance_scale=var_scale))

        return load_params

    def combine_sets(self, pv_params, load_params):

        sets = []

        for pv_param in pv_params:
            for load_param in load_params:
                sets.append(ForecastArgSet(pv_param_set = pv_param, load_param_set = load_param))

        return sets


class ForecastArgSet(dict):
    def __init__(self, pv_param_set=None, load_param_set=None, preset_to_use=None):

        if pv_param_set is None and load_param_set is None and preset_to_use is not None:
            saved_dict = self.get_preset(preset_to_use)
            super(ForecastArgSet, self).__init__(saved_dict)

        elif pv_param_set is not None and load_param_set is not None and preset_to_use is None:
            super(ForecastArgSet, self).__init__(pv_args=pv_param_set, load_args=load_param_set)

        else:
            raise KeyError('Unable to parse inputs')

        self.mape_vals = []
        self.mape_mean = None
        self.mape_std = None

    def update_with_mape(self, mape):

        self.mape_vals.append(mape)
        self.mape_mean = np.mean(self.mape_vals)
        self.mape_std = np.std(self.mape_vals)

    def get_preset(self,forecast_accuracy=50):
        potential_forecast_accuracies = (50, 70, 85)
        if forecast_accuracy not in potential_forecast_accuracies:
            raise ValueError('do not have relevant sampling parameters for forecast accuracy {}, must be one of {}'.format(
                forecast_accuracy, potential_forecast_accuracies))

        if forecast_accuracy == 50:
             return {'pv_args': {'noise_params': ({'lower': 0.0, 'upper': 0.5},
                        {'std_ratio': 0.25}),
                         'push_peak_val': True,
                         'push_peak_ratio': 0.0,
                         'push_individual_vals': True,
                         'push_individual_ratio': 0.5},
             'load_args': {'load_variance_scale': 2.0}}

        if forecast_accuracy == 70:
            return {'pv_args': {'noise_params': ({'lower': 0.0, 'upper': 0.5},
                        {'std_ratio': 0.25}),
                         'push_peak_val': True,
                         'push_peak_ratio': 0.0,
                         'push_individual_vals': True,
                         'push_individual_ratio': 0.65},
             'load_args': {'load_variance_scale': 2.0}}

        elif forecast_accuracy == 85:
            return {'pv_args': {'noise_params': ({'lower': 0.0, 'upper': 0.5},
                       {'std_ratio': 0.25}),
                      'push_peak_val': True,
                      'push_peak_ratio': 0.0,
                      'push_individual_vals': True,
                      'push_individual_ratio': 1.0},
                     'load_args': {'load_variance_scale': 2.0}}


    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented

        return self.mape_mean == other.mape_mean

    def __lt__(self, other):
        if type(self) != type(other):
            return NotImplemented

        return self.mape_mean < other.mape_mean

    def __gt__(self, other):
        if type(self) != type(other):
            return NotImplemented

        return self.mape_mean > other.mape_mean

class HorizonOutput:

    def __init__(self,control_dicts, microgrid, current_step):
        self.df = pd.DataFrame(control_dicts)
        self.microgrid = microgrid
        self.current_step = current_step
        self.cost = self.compute_cost_over_horizon(current_step)
        self.first_dict = control_dicts[0]

    def compute_cost_over_horizon(self, current_step):

        horizon = self.microgrid.horizon
        cost = 0.0

        cost += self.df['loss_load'].sum()*self.microgrid.parameters['cost_loss_load'].values[0]  # loss load

        if self.microgrid.architecture['genset'] == 1:
            cost += self.df['genset'].sum() * self.microgrid.parameters['fuel_cost'].values[0]

        if self.microgrid.architecture['grid'] == 1:
            price_import = self.microgrid._grid_price_import.iloc[current_step:current_step + horizon].values
            price_export = self.microgrid._grid_price_export.iloc[current_step:current_step + horizon].values

            import_cost_vec = price_import.reshape(-1)*self.df['grid_import']
            export_cost_vec = price_export.reshape(-1)*self.df['grid_export']
            grid_cost = import_cost_vec.sum()-export_cost_vec.sum()

            cost += grid_cost

        return cost

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.cost == other.cost

    def __lt__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.cost < other.cost

    def __gt__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.cost > other.cost


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

     >>>  names = ('action', 'status', 'production', 'cost', 'co2')
     >>>  dfs = (baseline_linprog_action, baseline_linprog_update_status,
     >>>          baseline_linprog_record_production, baseline_linprog_cost) # From MPC
     >>> M = ControlOutput(names, dfs,'mpc')
     >>> actions = M['action'] # returns the dataframe baseline_linprog_action

    """
    def __init__(self, names=None, dfs=None, alg_name=None, empty=False, microgrid=None):

        if not empty:
            if names is None:
                raise TypeError('names cannot be None unless initializing empty and empty=True')
            if dfs is None:
                raise TypeError('dfs cannot be None unless initializing empty and empty=True')
            if alg_name is None:
                raise TypeError('alg_name cannot be None unless initializing empty and empty=True')
        # else:
            # if not isinstance(microgrid,Microgrid.Microgrid):
            #     raise TypeError('microgrid must be a Microgrid if empty is True')

        if not empty:
            names_needed = ('action', 'status', 'production', 'cost', 'co2')
            if any([needed not in names for needed in names_needed]):
                raise ValueError('Names must contain {}, currently contains {}'.format(names,names_needed))

            super(ControlOutput, self).__init__(zip(names, dfs))
            self.alg_name = alg_name
            self.microgrid = microgrid

        else:
            names = ('action', 'status', 'production', 'cost', 'co2')
            baseline_linprog_action = deepcopy(microgrid._df_record_control_dict)
            baseline_linprog_update_status = deepcopy(microgrid._df_record_state)
            baseline_linprog_record_production = deepcopy(microgrid._df_record_actual_production)
            baseline_linprog_cost = deepcopy(microgrid._df_record_cost)
            baseline_linprog_co2 = deepcopy(microgrid._df_record_co2)

            dfs = (baseline_linprog_action, baseline_linprog_update_status,
               baseline_linprog_record_production, baseline_linprog_cost, baseline_linprog_co2)

            super(ControlOutput, self).__init__(zip(names, dfs))
            self.alg_name = alg_name
            self.microgrid = microgrid

    def append(self, other_output, actual_load=None, actual_pv=None, actual_grid = None, slice_to_use=0):
        if isinstance(other_output, ControlOutput):
            for name in self.keys():
                if name not in other_output.keys():
                    raise KeyError('name {} not founds in other_output keys'.format(name))

                self[name].append(other_output[name].iloc[slice_to_use], ignore_index=True)

        elif isinstance(other_output, HorizonOutput):
            action = self['action']
            production = self['production']
            cost = self['cost']
            status = self['status']
            co2 = self['co2']

            action = self.microgrid._record_action(other_output.first_dict, action)
            production = self.microgrid._record_production(other_output.first_dict, production, status)

            last_prod = dict([(key, production[key][-1]) for key in production])

            i = other_output.current_step

            if self.microgrid.architecture['grid'] == 1:
                co2 = self.microgrid._record_co2(
                    last_prod,
                    co2,
                    self.microgrid._grid_co2.iloc[i].values[0]
                )

                status = self.microgrid._update_status(
                    last_prod,
                    status,
                    actual_load,
                    actual_pv,
                    actual_grid,
                    self.microgrid._grid_price_import.iloc[i + 1].values[0],
                    self.microgrid._grid_price_export.iloc[i + 1].values[0],
                    self.microgrid._grid_co2.iloc[i + 1].values[0]
                )

                cost = self.microgrid._record_cost(
                    last_prod,
                    cost,
                    co2,
                    self.microgrid._grid_price_import.iloc[i, 0],
                    self.microgrid._grid_price_export.iloc[i, 0])
            else:

                co2 = self.microgrid._record_co2(
                    last_prod,
                    co2,
                )

                status = self.microgrid._update_status(
                    last_prod,
                    status,
                    actual_load,
                    actual_pv
                )
                cost = self.microgrid._record_cost(
                    last_prod,
                    cost,
                    co2
                )

            self['action'] = action
            self['production'] = production
            self['cost'] = cost
            self['status'] = status
            self['co2'] = co2


    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return np.sum(self['cost']) == np.sum(other['cost'])

    def __lt__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return np.sum(self['cost']) < np.sum(other['cost'])

    def __gt__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return np.sum(self['cost']) > np.sum(other['cost'])


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
        cost_co2 = parameters['cost_co2'].values[0]

        if self.has_genset:
            p_genset_min = parameters['genset_pmin'].values[0] * parameters['genset_rated_power'].values[0]
            p_genset_max = parameters['genset_pmax'].values[0] * parameters['genset_rated_power'].values[0]
            genset_co2 = parameters['genset_co2'].values[0]

        else:
            p_genset_min = 0
            p_genset_max = 0
            genset_co2 = 0

        return eta, battery_capacity, fuel_cost, cost_battery_cycle, cost_loss_load, p_genset_min, p_genset_max, cost_co2, genset_co2

    def _create_problem(self, eta, battery_capacity, fuel_cost, cost_battery_cycle, cost_loss_load,
                        p_genset_min, p_genset_max, cost_co2, genset_co2):

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
            cost_vector = np.array([fuel_cost + cost_co2 * genset_co2, 0, 0,
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
                        p_max_import, p_max_export, soc_0, p_genset_max, cost_co2, grid_co2, genset_co2,):

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
            self.costs.value[1::8] = import_price.reshape(-1) + grid_co2.reshape(-1) * cost_co2
            self.costs.value[2::8] = export_price.reshape(-1)
        else:
            self.costs.value[0::7] = import_price.reshape(-1) + grid_co2.reshape(-1) * cost_co2
            self.costs.value[1::7] = export_price.reshape(-1)

        if np.isnan(self.costs.value).any():
            raise RuntimeError('There are still nan values in self.costs.value, something is wrong')

    def set_and_solve(self, load_vector, pv_vector, grid_vector, import_price, export_price, e_max, e_min, p_max_charge,
                      p_max_discharge, p_max_import, p_max_export, soc_0, p_genset_max, cost_co2, grid_co2, genset_co2, iteration=None, total_iterations=None, return_steps=0):
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
                        p_max_import, p_max_export, soc_0, p_genset_max, cost_co2, grid_co2, genset_co2,)

        self.problem.solve(warm_start = True)

        if self.problem.status == 'infeasible':
            print(self.problem.status)
            print('Infeasible problem on step {} of {}, retrying with GLPK_MI solver'.format(iteration,total_iterations))
            self.problem.solve(solver = cp.GLPK_MI)
            if self.problem.status == 'infeasible':
                print('Failed again')
            else:
                print('Optimizer found with GLPK_MI solver')

        if return_steps == 0:
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

            return control_dict

        else:
            if return_steps > self.microgrid.horizon:
                raise ValueError('return_steps cannot be greater than horizon')

            control_dicts = []

            if self.has_genset:
                for j in range(return_steps):
                    start_index = j*8

                    control_dict = {'battery_charge': self.p_vars.value[start_index+3],
                                    'battery_discharge': self.p_vars.value[start_index+4],
                                    'genset': self.p_vars.value[start_index],
                                    'grid_import': self.p_vars.value[start_index+1],
                                    'grid_export': self.p_vars.value[start_index+2],
                                    'loss_load': self.p_vars.value[start_index+6],
                                    'pv_consummed': pv_vector[j] - self.p_vars.value[start_index+5],
                                    'pv_curtailed': self.p_vars.value[start_index+5],
                                    'load': load_vector[j],
                                    'pv': pv_vector[j]}

                    control_dicts.append(control_dict)

            else:
                for j in range(return_steps):
                    start_index = j * 7

                    control_dict = {'battery_charge': self.p_vars.value[start_index + 2],
                                    'battery_discharge': self.p_vars.value[start_index + 3],
                                    'grid_import': self.p_vars.value[start_index],
                                    'grid_export': self.p_vars.value[start_index + 1],
                                    'loss_load': self.p_vars.value[start_index + 5],
                                    'pv_consummed': pv_vector[j] - self.p_vars.value[start_index + 4],
                                    'pv_curtailed': self.p_vars.value[start_index + 4],
                                    'load': load_vector[j],
                                    'pv': pv_vector[j]}

                    control_dicts.append(control_dict)

            return control_dicts


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
        baseline_linprog_action = deepcopy(self.microgrid._df_record_control_dict)
        baseline_linprog_update_status = deepcopy(self.microgrid._df_record_state)
        baseline_linprog_record_production = deepcopy(self.microgrid._df_record_actual_production)
        baseline_linprog_cost = deepcopy(self.microgrid._df_record_cost)
        baseline_linprog_co2 = deepcopy(self.microgrid._df_record_co2)

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
                p_max_import = 0
                p_max_export = 0
                grid_co2 = np.zeros(horizon)
            else:
                temp_grid = sample.loc[i:i + horizon - 1, 'grid'].values
                price_import = self.microgrid._grid_price_import.iloc[i:i + horizon].values
                price_export = self.microgrid._grid_price_export.iloc[i:i + horizon].values
                grid_co2 = self.microgrid._grid_co2.iloc[i:i + horizon].values
                p_max_import = self.microgrid.parameters['grid_power_import'].values[0]
                p_max_export = self.microgrid.parameters['grid_power_export'].values[0]

                if temp_grid.shape != price_export.shape and price_export.shape != price_import.shape:
                    raise RuntimeError('I think this is a problem')


            e_min = self.microgrid.parameters['battery_soc_min'].values[0]
            e_max = self.microgrid.parameters['battery_soc_max'].values[0]
            p_max_charge = self.microgrid.parameters['battery_power_charge'].values[0]
            p_max_discharge = self.microgrid.parameters['battery_power_discharge'].values[0]

            soc_0 = baseline_linprog_update_status['battery_soc'][-1]

            cost_co2 = self.microgrid.parameters['cost_co2'].values[0]

            if self.has_genset:
                p_genset_max = self.microgrid.parameters['genset_pmax'].values[0] *\
                           self.microgrid.parameters['genset_rated_power'].values[0]
                genset_co2 = self.microgrid.parameters['genset_co2'].values[0]
            else:
                p_genset_max = None
                genset_co2 = None

            # Solve one step of MPC
            control_dict = self.set_and_solve(sample.loc[i:i + horizon - 1, 'load'].values,
                                              sample.loc[i:i + horizon - 1, 'pv'].values, temp_grid, price_import,
                                              price_export, e_max, e_min, p_max_charge, p_max_discharge, p_max_import,
                                              p_max_export, soc_0, p_genset_max, cost_co2, grid_co2, genset_co2, iteration = i, total_iterations = num_iter)

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
                baseline_linprog_co2 = self.microgrid._record_co2(
                    {i: baseline_linprog_record_production[i][-1] for i in baseline_linprog_record_production},
                    baseline_linprog_co2,
                    self.microgrid._grid_co2.iloc[i].values[0],
                )

                baseline_linprog_update_status = self.microgrid._update_status(
                    {i: baseline_linprog_record_production[i][-1] for i in baseline_linprog_record_production},
                    baseline_linprog_update_status,
                    sample.at[i + 1, 'load'],
                    sample.at[i + 1, 'pv'],
                    sample.at[i + 1, 'grid'],
                    self.microgrid._grid_price_import.iloc[i + 1].values[0],
                    self.microgrid._grid_price_export.iloc[i + 1].values[0],
                    self.microgrid._grid_co2.iloc[i + 1].values[0],
                )

                baseline_linprog_cost = self.microgrid._record_cost(
                    {i: baseline_linprog_record_production[i][-1] for i in baseline_linprog_record_production},
                    baseline_linprog_cost,
                    baseline_linprog_co2,
                    self.microgrid._grid_price_import.iloc[i, 0],
                    self.microgrid._grid_price_export.iloc[i, 0])
            else:
                baseline_linprog_co2 = self.microgrid._record_co2(
                    {i: baseline_linprog_record_production[i][-1] for i in baseline_linprog_record_production},
                    baseline_linprog_co2,
                )

                baseline_linprog_update_status = self.microgrid._update_status(
                    {i: baseline_linprog_record_production[i][-1] for i in baseline_linprog_record_production},
                    baseline_linprog_update_status,
                    sample.at[i + 1, 'load'],
                    sample.at[i + 1, 'pv']
                )
                baseline_linprog_cost = self.microgrid._record_cost(
                    {i: baseline_linprog_record_production[i][-1] for i in baseline_linprog_record_production},
                    baseline_linprog_cost,
                    baseline_linprog_co2,
                )

        names = ('action', 'status', 'production', 'cost', 'co2')

        dfs = (baseline_linprog_action, baseline_linprog_update_status,
               baseline_linprog_record_production, baseline_linprog_cost, baseline_linprog_co2)

        if verbose:
            print('Total time: {} minutes'.format(round((time.time()-t0)/60, 2)))

        return ControlOutput(names, dfs, 'mpc')

    def run_mpc_on_microgrid(self, forecast_steps=None, verbose=False, **kwargs):
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

    def mpc_single_step(self, sample, previous_output, current_step):

        if not isinstance(previous_output, ControlOutput):
            raise TypeError('previous_output must be ControlOutput, unless first_step is True')

        # baseline_linprog_update_status = pd.DataFrame(previous_output['status'].iloc[-1].squeeze()).transpose()

        horizon = self.microgrid.horizon

        if self.microgrid.architecture['grid'] == 0:
            temp_grid = np.zeros(horizon)
            price_import = np.zeros(horizon)
            price_export = np.zeros(horizon)
            grid_co2 = np.zeros(horizon)
            p_max_import = 0
            p_max_export = 0
        else:
            temp_grid = sample.loc[current_step:current_step + horizon - 1, 'grid'].values
            price_import = self.microgrid._grid_price_import.iloc[current_step:current_step + horizon].values
            price_export = self.microgrid._grid_price_export.iloc[current_step:current_step + horizon].values
            grid_co2 = self.microgrid._grid_co2.iloc[current_step:current_step + horizon].values
            p_max_import = self.microgrid.parameters['grid_power_import'].values[0]
            p_max_export = self.microgrid.parameters['grid_power_export'].values[0]

            if temp_grid.shape != price_export.shape and price_export.shape != price_import.shape:
                raise RuntimeError('I think this is a problem')

        e_min = self.microgrid.parameters['battery_soc_min'].values[0]
        e_max = self.microgrid.parameters['battery_soc_max'].values[0]
        p_max_charge = self.microgrid.parameters['battery_power_charge'].values[0]
        p_max_discharge = self.microgrid.parameters['battery_power_discharge'].values[0]
        soc_0 = previous_output['status']['battery_soc'][-1]

        cost_co2 = self.microgrid.parameters['cost_co2'].values[0]

        if self.has_genset:
            p_genset_max = self.microgrid.parameters['genset_pmax'].values[0] * \
                           self.microgrid.parameters['genset_rated_power'].values[0]
            genset_co2 = self.microgrid.parameters['genset_co2'].values[0]
        else:
            p_genset_max = None
            genset_co2 = 0

        # Solve one step of MPC
        control_dicts = self.set_and_solve(sample.loc[current_step:current_step + horizon - 1, 'load'].values,
                                          sample.loc[current_step:current_step + horizon - 1, 'pv'].values, temp_grid, price_import,
                                          price_export, e_max, e_min, p_max_charge, p_max_discharge, p_max_import,
                                          p_max_export, soc_0, p_genset_max, cost_co2, grid_co2, genset_co2, iteration=current_step, return_steps=self.microgrid.horizon)

        if any([d is None for d in control_dicts]):
            for j, d in enumerate(control_dicts):
                if d is None:
                    raise TypeError('control_dict number {} is None'.format(j))

        return HorizonOutput(control_dicts, self.microgrid, current_step)



class RuleBasedControl:
    def __init__(self, microgrid):
        # if not isinstance(microgrid, Microgrid.Microgrid):
        #     raise TypeError('microgrid must be of type Microgrid, is {}'.format(type(microgrid)))

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
            capa_to_discharge = max(min((status['battery_soc'][-1] *
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
                         status['battery_soc'][-1] *
                         parameters['battery_capacity'].values[0]
                         ) / self.microgrid.parameters['battery_efficiency'].values[0], 0)
                    capa_to_discharge = max((status['battery_soc'][-1] *
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

        """ This function runs the rule based benchmark over the datasets (load and pv profiles) in the microgrid."""

        baseline_priority_list_action = deepcopy(self.microgrid._df_record_control_dict)
        baseline_priority_list_update_status = deepcopy(self.microgrid._df_record_state)
        baseline_priority_list_record_production = deepcopy(self.microgrid._df_record_actual_production)
        baseline_priority_list_cost = deepcopy(self.microgrid._df_record_cost)
        baseline_priority_list_co2 = deepcopy(self.microgrid._df_record_co2)


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

                baseline_priority_list_co2 = self.microgrid._record_co2(
                    {i: baseline_priority_list_record_production[i][-1] for i in baseline_priority_list_record_production},
                    baseline_priority_list_co2,
                    self.microgrid._grid_co2.iloc[i].values[0],
                )

                baseline_priority_list_update_status = self.microgrid._update_status(
                    {i: baseline_priority_list_record_production[i][-1] for i in baseline_priority_list_record_production},
                    baseline_priority_list_update_status, self.microgrid._load_ts.iloc[i + 1].values[0],
                    self.microgrid._pv_ts.iloc[i + 1].values[0],
                    self.microgrid._grid_status_ts.iloc[i + 1].values[0],
                    self.microgrid._grid_price_import.iloc[i + 1].values[0],
                    self.microgrid._grid_price_export.iloc[i + 1].values[0],
                    self.microgrid._grid_co2.iloc[i + 1].values[0],
                )


                baseline_priority_list_cost = self.microgrid._record_cost(
                    {i: baseline_priority_list_record_production[i][-1] for i in
                     baseline_priority_list_record_production},
                    baseline_priority_list_cost,
                    baseline_priority_list_co2,
                    self.microgrid._grid_price_import.iloc[i,0], self.microgrid._grid_price_export.iloc[i,0])
            else:

                baseline_priority_list_co2 = self.microgrid._record_co2(
                    {i: baseline_priority_list_record_production[i][-1] for i in
                     baseline_priority_list_record_production},
                     baseline_priority_list_co2,
                )

                baseline_priority_list_update_status = self.microgrid._update_status(
                    {i: baseline_priority_list_record_production[i][-1] for i in
                     baseline_priority_list_record_production},
                    baseline_priority_list_update_status, self.microgrid._load_ts.iloc[i + 1].values[0],
                    self.microgrid._pv_ts.iloc[i + 1].values[0])

                baseline_priority_list_cost = self.microgrid._record_cost(
                    {i: baseline_priority_list_record_production[i][-1] for i in
                     baseline_priority_list_record_production},
                    baseline_priority_list_cost,
                    baseline_priority_list_co2)

        names = ('action', 'status', 'production', 'cost', 'co2')

        dfs = (baseline_priority_list_action, baseline_priority_list_update_status,
               baseline_priority_list_record_production, baseline_priority_list_cost, baseline_priority_list_co2)

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
        # if not isinstance(microgrid, Microgrid.Microgrid):
        #     raise TypeError('microgrid must be of type Microgrid, is {}'.format(type(microgrid)))

        self.microgrid = microgrid
        self.outputs_dict = dict()

        self.mpc_output = None
        self.has_mpc_benchmark = False
        self.rule_based_output = None
        self.has_rule_based_benchmark = False
        self.saa_output = None
        self.has_saa_benchmark = False

    def run_mpc_benchmark(self, verbose=False, **kwargs):
        """
        Run the MPC benchmark and store the output in self.mpc_output
        :return:
            None
        """
        MPC = ModelPredictiveControl(self.microgrid)
        self.mpc_output = MPC.run_mpc_on_microgrid(verbose=verbose, **kwargs)
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

    def run_saa_benchmark(self, preset_to_use=85, **kwargs):
        SAA = SampleAverageApproximation(self.microgrid, preset_to_use=preset_to_use, **kwargs)
        self.saa_output = SAA.run(**kwargs)
        self.has_saa_benchmark = True
        self.outputs_dict[self.saa_output.alg_name] = self.saa_output

    def run_benchmarks(self, algo=None, verbose=False, preset_to_use=85, **kwargs):
        """
        Runs both run_mpc_benchmark() and self.run_mpc_benchmark() and stores the results.
        :param verbose: bool, default False
            Whether to describe benchmarks after running.
        :return:
            None
        """

        if algo == 'mpc':
            self.run_mpc_benchmark(verbose=verbose, **kwargs)
        elif algo == 'rbc':
            self.run_rule_based_benchmark()
        elif algo == 'saa':
            self.run_saa_benchmark(preset_to_use=preset_to_use, **kwargs)
        else:
            self.run_mpc_benchmark(verbose=verbose, **kwargs)
            self.run_rule_based_benchmark()
            self.run_saa_benchmark(preset_to_use=preset_to_use, **kwargs)

        if verbose:
            self.describe_benchmarks()

    def describe_benchmarks(self, test_split=False, test_ratio=None, test_index=None, algorithms=None):
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
        possible_benchmarks = ('saa', 'mpc', 'rbc')

        if algorithms is not None:
            if any([b_name not in possible_benchmarks for b_name in algorithms]):
                raise ValueError('Unable to recognize one or multiple of list_of_benchmarks: {}, can only contain {}'.format(
                    algorithms, possible_benchmarks))
        else:
            algorithms = possible_benchmarks

        t_vals = []
        for key in self.outputs_dict:
            t_vals.append(len(self.outputs_dict[key]['cost']['cost']))

        if not all([t_val == t_vals[0] for t_val in t_vals]):
            raise ValueError('Outputs are of different lengths')

        T = t_vals[0]

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

            if self.has_mpc_benchmark and 'mpc' in algorithms:
                cost = round(np.sum(self.mpc_output['cost']['cost'][int(np.ceil(T*(1-test_ratio))):]), 2)
                print('Cost of the last {} steps ({} percent of all steps) using MPC: {}'.format(steps, percent, cost))

            if self.has_rule_based_benchmark and 'rbc' in algorithms:
                cost = round(np.sum(self.rule_based_output['cost']['cost'][int(np.ceil(T*(1-test_ratio))):]), 2)
                print('Cost of the last {} steps ({} percent of all steps) using rule-based control: {}'.format(steps, percent, cost))

            if self.has_saa_benchmark and 'saa' in algorithms:
                cost = round(np.sum(self.saa_output['cost']['cost'][int(np.ceil(T*(1-test_ratio))):]), 2)
                print('Cost of the last {} steps ({} percent of all steps) using sample-average MPC control: {}'.format(steps, percent, cost))

        else:

            if self.has_mpc_benchmark and 'mpc' in algorithms:
                cost_train = round(np.sum(self.mpc_output['cost']['cost'][:test_index]), 2)
                cost_test = round(np.sum(self.mpc_output['cost']['cost'][test_index:]), 2)

                print('Test set cost using MPC: {}'.format(cost_test))
                print('Train set cost using MPC: {}'.format(cost_train))

            if self.has_rule_based_benchmark and 'rbc' in algorithms:
                cost_train = round(np.sum(self.rule_based_output['cost']['cost'][:test_index]), 2)
                cost_test = round(np.sum(self.rule_based_output['cost']['cost'][test_index:]), 2)

                print('Test set cost using RBC: {}'.format(cost_test))
                print('Train set cost using RBC: {}'.format(cost_train))

            if self.has_saa_benchmark and 'saa' in algorithms:
                cost_train = round(np.sum(self.saa_output['cost']['cost'][:test_index]), 2)
                cost_test = round(np.sum(self.saa_output['cost']['cost'][test_index:]), 2)

                print('Test set cost using SAA: {}'.format(cost_test))
                print('Train set cost using SAA: {}'.format(cost_train))


if __name__=='__main__':
    # TODO this has new code, you need to debug SAA
    from src.pymgrid import MicrogridGenerator

    m_gen = MicrogridGenerator.MicrogridGenerator(nb_microgrid=25)
    # m_gen = m_gen.load('pymgrid25')


    m_gen.generate_microgrid(verbose=False)

    for j, microgrid in enumerate(m_gen.microgrids):
        print(j, microgrid.architecture)
    #     if microgrid.architecture['grid']==1:
    #         print('Found a good one on iter', j)
    #         break
    #
    # if microgrid.architecture['grid']==0:
    #     raise ValueError('no')

    microgrid = m_gen.microgrids[1]
    log = open('grid_and_genset.log','a')
    sys.stdout = log
    MPC = ModelPredictiveControl(microgrid)
    MPC.run_mpc_on_microgrid(forecast_steps=2000)

    # sampling_args = dict(load_variance_scale=1.2, noise_params=(None, {'std_ratio': 0.3}), verbose=False)
    # SAA = SampleAverageApproximation(microgrid)
    # samples = SAA.sample_from_forecasts(n_samples=3, **sampling_args)
    # underlying_data_list = [SAA.underlying_data]*3
    # output = SAA.run_mpc_on_group(samples, verbose=True)



    # t0 = time.time()
    # benchmarks = Benchmarks(microgrid)
    # benchmarks.run_saa_benchmark(verbose=True, n_samples=2)
    # cProfile.run('benchmarks.run_saa_benchmark(verbose=True, n_samples=2)', sort='cumtime')

    # benchmarks.describe_benchmarks()


    # print(time.time()-t0,' seconds')
