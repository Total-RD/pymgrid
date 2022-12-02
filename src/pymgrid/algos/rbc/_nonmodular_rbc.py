import operator
import sys
from copy import deepcopy

from pymgrid.algos.Control import ControlOutput


class NonModularRuleBasedControl:
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