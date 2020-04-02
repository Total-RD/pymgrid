import pandas as pd
import numpy as np
from copy import copy
import cvxpy as cp
import operator
import math

class Microgrid:

    def __init__(self, parameters):


        #list of parameters
        #this is a static dataframe: parameters of the microgrid that do not change with time
        self.parameters = parameters['parameters']
        self.architecture =  parameters['architecture']

        #different timeseries
        self.load=parameters['load']
        self.pv=parameters['pv']


        if parameters['architecture']['grid']==1:
            self.grid_status=parameters['grid_ts'] #time series of outages
            #todo if we move to time series of price
            self.grid_price_import=0
            self.grid_price_export=0


        #create timeseries of updated
        # those dataframe record what is hapepning at each time step
        self.df_actions=parameters['df_actions']
        self.df_status = parameters['df_status']
        self.df_actual_generation = parameters['df_actual_generation']
        self.df_cost = parameters['df_cost']

        self.horizon = 24

        self.data_length = min(self.load, self.pv)

        self.run_timestep = 0
        self.done = False


    def train_test_split(self, train_size=0.67, shuffle = False, ):
        self.limit_index = np.ceil(self.data_length*train_size)
        self.load_train = self.load.iloc[:self.limit_index]
        self.pv_train = self.pv.iloc[:self.limit_index]

        self.load_test = self.load.iloc[self.limit_index:]
        self.pv_test = self.pv.iloc[self.limit_index:]

        if self.architecture['grid'] == 1:
            self.grid_status_train = self.grid_status.iloc[:self.limit_index]
            self.grid_status_test = self.grid_status.iloc[self.limit_index:]


    def reset(self):
        #todo validate
        #todo mechanism to store history of what happened
        self.df_actions = self.df_actions[0:0]
        self.df_status = self.df_status[0:0]
        self.df_actual_generation = self.df_actual_generation[0:0]
        self.df_cost = self.df_cost[0:0]

        self.run_timestep = 0
        self.done = False

    def _generate_priority_list(self, architecture, parameters , grid_status=0,  ):

        # compute marginal cost of each ressource
        # construct priority list
        # should receive fuel cost and cost curve, price of electricity
        if architecture['battery'] != 0 and architecture['grid'] == 1:


            if parameters['grid_price_export'].values[0] / (parameters['battery_efficiency'].values[0]**2) < parameters['grid_price_import'].values[0]:

                # should return something like ['gen', starting at in MW]?
                priority_dict = {'PV': 1 * architecture['PV'],
                                 'battery': 2 * architecture['battery'],
                                 'grid': 3 * architecture['grid'] * grid_status,
                                 'genset': 4 * architecture['genset']}

            else:
                # should return something like ['gen', starting at in MW]?
                priority_dict = {'PV': 1 * architecture['PV'],
                                 'battery': 3 * architecture['battery'],
                                 'grid': 2 * architecture['grid'] * grid_status,
                                 'genset': 4 * architecture['genset']}

        else:
            priority_dict = {'PV': 1 * architecture['PV'],
                             'battery': 2 * architecture['battery'],
                             'grid': 3 * architecture['grid'] * grid_status,
                             'genset': 4 * architecture['genset']}

        return priority_dict

    #Todo later: add reserve for ploutos
    def _generate_genset_reserves(self, run_dict):

        # spinning=run_dict['next_load']*0.2

        nb_gen_min = int(math.ceil(run_dict['next_peak'] / self.genset_power_max))

        return nb_gen_min


    def _run_priority_based(self, load, pv, parameters, status, priority_dict):


        temp_load = load
        #todo add reserves to ploutos
        excess_gen = 0

        pCharge = 0
        pDischarge = 0
        pImport = 0
        pExport = 0
        pGenset = 0
        load_not_matched = 0
        pv_not_curtailed = 0
        self_consumed_pv = 0

        #todo consider min production of genset
        if self.architecture['genset'] == 1:
            min_load = self.parameters['genset_rater_power'].values[0] * self.parameters['genset_pmin'].values[0]
            temp_load=temp_load-min_load
        # for gen with prio i in 1:max(priority_dict)
        # we sort the priority list
        # probably we should force the PV to be number one, the min_power should be absorbed by genset, grid?
        sorted_priority = sorted(priority_dict.items(), key=operator.itemgetter(1))
        # print (sorted_priority)
        for gen, priority in sorted_priority:  # .iteritems():

            if priority > 0:

                if gen == 'PV':
                    temp_load_for_excess = copy(temp_load)
                    # print (temp_load * self.maximum_instantaneous_pv_penetration - run_dict['next_pv'])
                    self_consumed_pv = min(temp_load, pv) #self.maximum_instantaneous_pv_penetration,
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
                    if temp_load  > 0:
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
                        'load_not_matched': load_not_matched,
                        'pv_consummed': pv_not_curtailed,
                        'pv_curtailed':pv - pv_not_curtailed,
                            'load': load,
                            'pv':pv}
                        #'nb_gen_min': nb_gen_min}

        return control_dict

    def baseline_rule_based(self, priority_list=0, length = 8760):


        self.baseline_priority_list_action = copy(self.df_actions)
        self.baseline_priority_list_update_status = copy(self.df_status)
        self.baseline_priority_list_record_production = copy(self.df_actual_generation)
        self.baseline_priority_list_cost= copy(self.df_cost)


        for i in range(length-self.horizon):

            if self.architecture['grid']==1:
                priority_dict = self._generate_priority_list( self.architecture, self.parameters,self.grid_status.iloc[i].values[0])
            else:
                priority_dict = self._generate_priority_list(self.architecture, self.parameters)

            control_dict = self._run_priority_based(self.load.iloc[i].values[0], self.pv.iloc[i].values[0], self.parameters,
                                                   self.baseline_priority_list_update_status, priority_dict)

            self.baseline_priority_list_action = self._record_action(control_dict, self.baseline_priority_list_action)

            self.baseline_priority_list_record_production = self._record_production(control_dict,
                                                                                   self.baseline_priority_list_record_production, self.baseline_priority_list_update_status)


            self.baseline_priority_list_update_status = self._update_status(self.baseline_priority_list_record_production.iloc[-1,:].to_dict(),
                                                                           self.baseline_priority_list_update_status)


            self.baseline_priority_list_cost = self._record_cost(self.baseline_priority_list_record_production.iloc[-1,:].to_dict(), self.baseline_priority_list_cost)



    def _mpc_lin_prog_cvxpy(self, parameters, load, pv, grid, status, horizon=24):
        #todo switch to a matrix structure
        load = np.reshape(load, (horizon,))
        #todo mip to choose which generators are online
        fuel_cost_polynom = 0
        if self.architecture['genset'] == 1:
            fuel_cost_polynom = []
            fuel_cost_polynom_order = self.parameters['genset_polynom_order'].values[0]
            for i in range(fuel_cost_polynom_order):
                fuel_cost_polynom.append(self.parameters['genset_polynom_' + str(i)].values[0])

        # variables
        #if self.architecture['genset'] ==1:
        p_genset = cp.Variable((horizon,), pos=True)

        #if self.architecture['grid']==1:
        p_grid_import = cp.Variable((horizon,), pos=True)
        p_grid_export = cp.Variable((horizon,), pos=True)
        u_import = cp.Variable((horizon,),pos=True)# boolean=True)
        u_export = cp.Variable((horizon,), pos=True)#boolean=True)

        #if self.architecture['battery'] == 1:
        p_charge = cp.Variable((horizon,), pos=True)
        p_discharge = cp.Variable((horizon,), pos=True)
        u_charge = cp.Variable((horizon,), pos=True)#boolean=True)
        u_discharge = cp.Variable((horizon,),pos=True)# boolean=True)
        nb_battery_cycle = cp.Variable((horizon,), pos=True)

        #if self.architecture['pv']==1:
        p_curtail_pv = cp.Variable((horizon,), pos=True)

        p_loss_load = cp.Variable((horizon,), pos=True)

        # parameters
        fuel_cost = np.zeros(horizon)
        p_price_import = np.zeros(horizon)
        p_price_export = np.zeros(horizon)
        cost_battery_cycle = np.zeros(horizon)


        cost_loss_load = parameters['cost_loss_load'].values[0]*np.ones(horizon)



        # Constraints
        constraints = []
        total_cost = 0.0
        constraints += [p_loss_load <= load]
        if self.architecture['genset'] ==1:
            # p_genset_min = parameters['genset_pmin'].values[0] * np.ones(horizon)
            # p_genset_max = parameters['genset_pmax'].values[0] * np.ones(horizon)
            p_genset_min = parameters['genset_pmin'].values[0]*parameters['genset_rated_power'].values[0]
            p_genset_max = parameters['genset_pmax'].values[0]*parameters['genset_rated_power'].values[0]
            fuel_cost = parameters['fuel_cost'].values[0] * np.ones(horizon)

            for t in range(horizon):

                constraints += [p_genset[t] >= p_genset_min,
                                p_genset[t] <= p_genset_max]

                total_cost += (p_genset[t] * fuel_cost[t])

        else:
            for t in range(horizon):
                constraints += [p_genset[t] == 0]


        if self.architecture['grid']==1:


            grid = np.reshape(grid, (horizon,))
            p_grid_import_max = parameters['grid_power_import'].values[0]
            p_grid_export_max = parameters['grid_power_export'].values[0]
            p_price_import = parameters['grid_price_import'].values[0] * np.ones(horizon)
            p_price_export = parameters['grid_price_export'].values[0] * np.ones(horizon)

            for t in range(horizon):
                constraints += [p_grid_import[t] <= p_grid_import_max,
                                p_grid_export[t] <= p_grid_export_max,
                                ]

                total_cost += (p_grid_import[t] * p_price_import[t]
                                                 - p_grid_export[t] * p_price_export[t])




        else:
            for t in range(horizon):
                constraints += [p_grid_import[t] == 0,
                                p_grid_export[t] == 0]

        if self.architecture['battery'] == 1:
            nb_battery_cycle = cp.Variable((horizon,), pos=True)
            battery_soc = cp.Variable((horizon,), pos=True)

            cost_battery_cycle = parameters['battery_cost_cycle'].values[0] / (2*parameters['battery_capacity'].values[0])* np.ones(horizon)

            p_charge_max = parameters['battery_power_charge'].values[0]
            p_discharge_max = parameters['battery_power_discharge'].values[0]

            for t in range(horizon):

                constraints += [p_charge[t] <= p_charge_max ,
                                p_discharge[t] <= p_discharge_max ,
                                u_charge[t] + u_discharge[t] <= 1]

                constraints += [battery_soc[t] >= parameters['battery_soc_min'].values[0] ,
                                battery_soc[t] <= parameters['battery_soc_max'].values[0] ]


                total_cost += (p_charge[t]*cost_battery_cycle[t]+ p_discharge[t]*cost_battery_cycle[t])

            soc_0 = status.iloc[-1]['battery_soc']
            constraints += [battery_soc[0] == soc_0 + (p_charge[0] * parameters['battery_efficiency'].values[0]
                                                       - p_discharge[0] / parameters['battery_efficiency'].values[0]) /
                            parameters['battery_capacity'].values[0]]
            for t in range(1, horizon):
                constraints += [
                    battery_soc[t] == battery_soc[t - 1] + (p_charge[t] * parameters['battery_efficiency'].values[0]
                                                            - p_discharge[t] / parameters['battery_efficiency'].values[
                                                                0]) / parameters['battery_capacity'].values[0]]


        else:
            for t in range(horizon):
                constraints += [p_charge[t] == 0,
                                p_discharge[t] == 0 ]

        if self.architecture['PV']==1:
            pv = np.reshape(pv, (horizon,))
            for t in range(horizon):
                constraints += [p_curtail_pv[t] <= pv[t]]
        else:
            for t in range(horizon):
                constraints += [p_curtail_pv[t] == 0]


        #constraint balance of power

        for t in range(horizon):
            total_cost += p_loss_load[t] * cost_loss_load[t]
            constraints+=[p_genset[t]
                          + p_discharge[t]
                          - p_charge[t]
                          - p_curtail_pv[t]
                          + p_loss_load [t]
                          + p_grid_import [t] * grid[t]
                          - p_grid_export [t] * grid[t]
                           == load[t] - pv[t]]


        # Objective function
        obj = cp.Minimize(total_cost)

        #todo fuel cost to consider polynom

        # set fuel consumption times price, elect * price, battery degradation
        # if fuel_cost_polynom_order == 0:
        #     obj = cp.Minimize(cp.sum(p_grid_import*p_price_import-p_grid_export*p_price_export))
        #
        # if fuel_cost_polynom_order == 1:
        #     obj = cp.Minimize(cp.sum(fuel_cost_polynom[0] +fuel_cost_polynom[1] * p_genset +
        #                              p_grid_import * p_price_import - p_grid_export * p_price_export))
        #
        # if fuel_cost_polynom_order == 2:
        #     obj = cp.Minimize(cp.sum(fuel_cost_polynom[0] +fuel_cost_polynom[1] * p_genset +
        #                              fuel_cost_polynom[2] * p_genset**2 +
        #                              p_grid_import * p_price_import - p_grid_export * p_price_export))


        prob = cp.Problem(obj, constraints)
        prob.solve()#verbose=True)#, solver=cp.ECOS,)

        control_dict = {'battery_charge': p_charge.value[0],
                            'battery_discharge': p_discharge.value[0],
                            'genset': p_genset.value[0],
                            'grid_import': p_grid_import.value[0],
                            'grid_export': p_grid_export.value[0],
                            'load_not_matched': p_loss_load.value[0],
                            'pv_consummed': pv[0]-p_curtail_pv.value[0],
                            'pv_curtailed': p_curtail_pv.value[0],
                            'load': load[0],
                            'pv':pv[0]}

        return control_dict

    def baseline_linprog(self, forecast_error=0, length=8760):

        self.baseline_linprog_action = copy(self.df_actions)
        self.baseline_linprog_update_status = copy(self.df_status)
        self.baseline_linprog_record_production = copy(self.df_actual_generation)
        self.baseline_linprog_cost = copy(self.df_cost)


        for i in range(length-self.horizon):
            if self.architecture['grid'] == 0:
                temp_grid = np.zeros(self.horizon)
            else:
                temp_grid = self.grid_status.iloc[i:i+self.horizon].values
            control_dict = self._mpc_lin_prog_cvxpy(self.parameters, self.load.iloc[i:i+self.horizon].values,
                                               self.pv.iloc[i:i+self.horizon].values,
                                               temp_grid,
                                               self.baseline_linprog_update_status,
                                               self.horizon )

            self.baseline_linprog_action = self._record_action(control_dict, self.baseline_linprog_action)

            self.baseline_linprog_record_production = self._record_production(control_dict,
                                                                             self.baseline_linprog_record_production,
                                                                             self.baseline_linprog_update_status)

            self.baseline_linprog_update_status = self._update_status(self.baseline_linprog_record_production.iloc[-1,:].to_dict(),
                                                                           self.baseline_linprog_update_status)

            self.baseline_linprog_cost = self._record_cost(self.baseline_linprog_record_production.iloc[-1,:].to_dict(),
                                                                                   self.baseline_linprog_cost)


    def _record_action(self, control_dict, df):
        df = df.append(control_dict,ignore_index=True)

        return df


    def _update_status(self, control_dict, df):
        #self.df_status = self.df_status.append(self.new_row, ignore_index=True)

        #todo add capa to discharge, capa to charge

        new_soc =np.nan
        for col in df.columns:

            if col == 'battery_soc':
                new_soc = df['battery_soc'].iloc[-1] + (control_dict['battery_charge']*self.parameters['battery_efficiency'].values[0]
                                                        - control_dict['battery_discharge']/self.parameters['battery_efficiency'].values[0])/self.parameters['battery_capacity'].values[0]
            #if col == 'net_load':


        dict = {
            'battery_soc':new_soc,
            'net_load': control_dict['load']-control_dict['pv']
        }

        df = df.append(dict,ignore_index=True)

        #self.df_status['soc'].iloc[-1] =(self.df_status['battery_soc'].iloc[-2]
        #                                              + self._record_actions['battery_power_charge'].iloc[-1]*self.parameters['battery_efficiency']
        #                                              - self._record_actions['battery_power_discharge'].iloc[-1]/self.parameters['battery_efficiency'])



        return df


    #now we consider all the generators on all the time (mainly concern genset)
    
    def _check_constraints_genset(self, p_genset):
        if p_genset < 0:
            p_genset =0
            print('error, genset power cannot be lower than 0')
    
        if p_genset < self.parameters['genset_rater_power'].values[0] * self.parameters['genset_pmin'].values[0]:
            p_genset = self.parameters['genset_rater_power'].values[0] * self.parameters['genset_pmin'].values[0]
        
        if p_genset > self.parameters['genset_rater_power'].values[0] * self.parameters['genset_pmax'].values[0]:
            p_genset = self.parameters['genset_rater_power'].values[0] * self.parameters['genset_pmax'].values[0]
        
        return p_genset
        
    def _check_constraints_grid(self, p_import, p_export):
        if p_import < 0:
            p_import = 0

        if p_export <0:
            p_export = 0

        if p_import > 0 and p_export > 0:
            print ('cannot import and export at the same time')
            #todo how to deal with that?
            
        if p_import > self.parameters['grid_power_import'].values[0]:
            p_import = self.parameters['grid_power_import'].values[0]

        if p_export > self.parameters['grid_power_export'].values[0]:
            p_export = self.parameters['grid_power_export'].values[0]
        
        return p_import, p_export
        
    def _check_constraints_battery(self, p_charge, p_discharge, status):

        if p_charge < 0:
            p_charge = 0

        if p_discharge < 0:
            p_discharge = 0

        if p_charge > 0 and p_discharge > 0:
            print ('cannot import and export at the same time')
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

        #todo enforce constraints
        #todo make sure the control actions repect their respective constriants
        total_load = 0
        total_production = 0
        threshold = 0.001
        total_load = control_dict['load'] - control_dict['pv']


        #check the generator constraints


        try:
            total_production += control_dict['load_not_matched']
        except:
            control_dict['load_not_matched'] =0
        try:
            total_production -= control_dict['pv_curtailed']
        except:
            control_dict['pv_curtailed'] = 0

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
            df = df.append(control_dict, ignore_index=True)
            print('total_production > total_load')
            print(control_dict)

        elif total_production < total_load :
            control_dict['load_not_matched']+= total_load-total_production
            df = df.append(control_dict, ignore_index=True)
            print('total_production < total_load')
            print(control_dict)

        return df

    def _record_cost(self, control_dict, df):

        cost = control_dict['load_not_matched'] * self.parameters['cost_loss_load'].values[0]

        if self.architecture['genset'] == 1:
            cost += control_dict['genset'] * self.parameters['fuel_cost'].values[0]

        if self.architecture['grid'] ==1:


            cost +=( control_dict['grid_import'] * self.parameters['grid_price_import'].values[0]
                     - control_dict['grid_export'] * self.parameters['grid_price_export'].values[0])

        cost_dict= {'cost': cost}

        df = df.append(cost_dict, ignore_index=True)

        return df

    #if return whole pv and load ts, the time can be counted in notebook
    def run(self, control_dict):
        #todo internaliser le traqueur du pas de temps

        self.df_actions = self._record_action(control_dict, self.df_actions)

        self.df_status = self._update_status(control_dict, self.df_status)

        self.df_actual_generation = self._record_production(control_dict,
                                                                         self.df_actual_generation)

        self.df_cost = self._record_cost(self.df_actual_generation.iloc[-1,:].to_dict('list'),
                                                           self.df_cost)

        #self.check_control()

        self.run_timestep += 1
        if self.run_timestep == self.data_length - self.horizon:
            self.done = True

        return self.get_state()


    def get_state(self):

        mg_data = {
            'current_state': self.df_status,
            'PV': self.pv.iloc[self.run_timestep:self.run_timestep + self.horizon].values,
            'load': self.load.iloc[self.run_timestep:self.run_timestep + self.horizon].values,
            'parameters': self.parameters,
            'cost': self.df_cost.iloc[-1]
        }

        return mg_data



    ########RL utility
    #todo utility function to split the data between training and testing
    #todo add a forecasting function that add noise to the time series
    #todo forecasting function can be used for both mpc benchmart and rl loop


    #todo get load(i)
    #todo get pv(i)
    #todo get net load
    #todo get state


    #todo info state -> retourner les colonnes
    #todo info parameter -> retourner le nom des colonnes

    #todo parameter done pour traquer les pas de temps

    #todo verbose
