
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
import pandas as pd
import numpy as np
from copy import deepcopy


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

    def to_frame(self):
        d = dict()
        max_len = -np.inf
        for k_1, v_1 in self.items():
            for k_2, v_2 in v_1.items():
                if len(v_2) > max_len:
                    max_len = len(v_2)
                d[(k_1, k_2)] = v_2

        for _, v in d.items():
            if len(v) < max_len:
                v.extend([np.nan]*(max_len-len(v)))

        return pd.DataFrame(d)

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
        from pymgrid.algos import ModelPredictiveControl
        MPC = ModelPredictiveControl(self.microgrid)
        self.mpc_output = MPC.run(verbose=verbose, **kwargs)
        self.has_mpc_benchmark = True
        self.outputs_dict[self.mpc_output.alg_name] = self.mpc_output

    def run_rule_based_benchmark(self):
        """
        Run the rule based benchmark and store the output in self.rule_based_output
        :return:
            None
        """
        from pymgrid.algos import RuleBasedControl
        RBC = RuleBasedControl(self.microgrid)
        self.rule_based_output = RBC.run_rule_based()
        self.has_rule_based_benchmark = True
        self.outputs_dict[self.rule_based_output.alg_name] = self.rule_based_output

    def run_saa_benchmark(self, preset_to_use=85, **kwargs):
        from pymgrid.algos.saa import SampleAverageApproximation
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
            t_vals.append(len(self.outputs_dict[key]['cost']['total_cost']))

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
                cost = round(np.sum(self.mpc_output['cost']['total_cost'][int(np.ceil(T*(1-test_ratio))):]), 2)
                print('Cost of the last {} steps ({} percent of all steps) using MPC: {}'.format(steps, percent, cost))

            if self.has_rule_based_benchmark and 'rbc' in algorithms:
                cost = round(np.sum(self.rule_based_output['cost']['total_cost'][int(np.ceil(T*(1-test_ratio))):]), 2)
                print('Cost of the last {} steps ({} percent of all steps) using rule-based control: {}'.format(steps, percent, cost))

            if self.has_saa_benchmark and 'saa' in algorithms:
                cost = round(np.sum(self.saa_output['cost']['total_cost'][int(np.ceil(T*(1-test_ratio))):]), 2)
                print('Cost of the last {} steps ({} percent of all steps) using sample-average MPC control: {}'.format(steps, percent, cost))

        else:

            if self.has_mpc_benchmark and 'mpc' in algorithms:
                cost_train = round(np.sum(self.mpc_output['cost']['total_cost'][:test_index]), 2)
                cost_test = round(np.sum(self.mpc_output['cost']['total_cost'][test_index:]), 2)

                print('Test set cost using MPC: {}'.format(cost_test))
                print('Train set cost using MPC: {}'.format(cost_train))

            if self.has_rule_based_benchmark and 'rbc' in algorithms:
                cost_train = round(np.sum(self.rule_based_output['cost']['total_cost'][:test_index]), 2)
                cost_test = round(np.sum(self.rule_based_output['cost']['total_cost'][test_index:]), 2)

                print('Test set cost using RBC: {}'.format(cost_test))
                print('Train set cost using RBC: {}'.format(cost_train))

            if self.has_saa_benchmark and 'saa' in algorithms:
                cost_train = round(np.sum(self.saa_output['cost']['total_cost'][:test_index]), 2)
                cost_test = round(np.sum(self.saa_output['cost']['total_cost'][test_index:]), 2)

                print('Test set cost using SAA: {}'.format(cost_test))
                print('Train set cost using SAA: {}'.format(cost_train))
