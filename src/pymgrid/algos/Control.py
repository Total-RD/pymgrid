
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
import sys
import operator


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

    def run_rule_based(self, priority_list=0, length=None):

        """ This function runs the rule based benchmark over the datasets (load and pv profiles) in the microgrid."""

        baseline_priority_list_action = deepcopy(self.microgrid._df_record_control_dict)
        baseline_priority_list_update_status = deepcopy(self.microgrid._df_record_state)
        baseline_priority_list_record_production = deepcopy(self.microgrid._df_record_actual_production)
        baseline_priority_list_cost = deepcopy(self.microgrid._df_record_cost)
        baseline_priority_list_co2 = deepcopy(self.microgrid._df_record_co2)

        if length is None or length >= self.microgrid._data_length:
            length = self.microgrid._data_length-1

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
        from pymgrid.algos.mpc.MPC import ModelPredictiveControl
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


if __name__ == '__main__':

    from src.pymgrid import MicrogridGenerator

    m_gen = MicrogridGenerator.MicrogridGenerator(nb_microgrid=25)
    m_gen.generate_microgrid(verbose=False)
    microgrid = m_gen.microgrids[4]

    benchmark = Benchmarks(microgrid)
    benchmark.run_saa_benchmark(preset_to_use=70)
    benchmark.describe_benchmarks(test_split=True, test_ratio=0.33)

