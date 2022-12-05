from pymgrid.modules import LoadModule, RenewableModule, BatteryModule, GridModule, GensetModule, UnbalancedEnergyModule
from copy import deepcopy
import pandas as pd
import numpy as np
from warnings import warn

_empty_params = dict(parameters=dict(),
                     df_actions=dict(),
                     architecture=dict(PV=0, battery=0, genset=0, grid=0),
                     df_status=dict(hour=[0]),
                     df_actual_generation=dict(loss_load=[], overgeneration=[]),
                     df_cost=dict(loss_load=[], overgeneration=[], total_cost=[]),
                     df_co2=dict(co2=[]),
                     pv=None,
                     load=None,
                     grid_ts=None,
                     control_dict=[],
                     grid_price_import=None,
                     grid_price_export=None,
                     grid_co2=None,
                    )


def get_empty_params():
    return deepcopy(_empty_params)


def check_viability(modular):
    classes = LoadModule, RenewableModule, BatteryModule, GridModule, GensetModule, UnbalancedEnergyModule
    classes_str = '\n'.join([str(x) for x in classes])
    n_modules_by_cls = dict(zip(classes, [0]*len(classes)))

    for name, module_list in modular.modules.iterdict():
        if len(module_list) > 1:
            raise ValueError('Cannot convert modular microgrid with multiple modules of same type. '
                         f'The following module name has multiple modules: {name}')

        module = module_list[0]
        if not isinstance(module, classes):
            raise TypeError(f'Unable to parse module {name} of type {module.__class__.__name__}.'
                            f'Must be one of:\n{classes_str}')

        n_modules_by_cls[module.__class__] += 1

    invalid = []
    for cls, n_modules in n_modules_by_cls.items():
        if n_modules > 1:
            invalid.append((cls, n_modules))

    if len(invalid):
        raise ValueError('Cannot convert modular microgrid with multiple modules of same type. '
                         f'The following module types conflict: {invalid}')

    if n_modules_by_cls[LoadModule] != 1:
        raise ValueError('Cannot convert modular microgrid that has no LoadModule.')


def finalize_params(params_dict):
    params_dict['parameters'] = pd.DataFrame(params_dict['parameters'])


def add_params_from_module(module, params_dict):
    if isinstance(module, LoadModule):
        add_load_params(module, params_dict)
    elif isinstance(module, RenewableModule):
        add_pv_params(module, params_dict)
    elif isinstance(module, BatteryModule):
        add_battery_params(module, params_dict)
    elif isinstance(module, GridModule):
        add_grid_params(module, params_dict)
    elif isinstance(module, GensetModule):
        add_genset_params(module, params_dict)
    elif isinstance(module, UnbalancedEnergyModule):
        add_unbalanced_energy_params(module, params_dict)
    else:
        raise ValueError(f'Cannot parse module {module}.')


def add_load_params(load_module, params_dict):
    params_dict['load'] = pd.DataFrame(load_module.time_series)
    _add_to_parameters(params_dict,
                       load=-1 * load_module.min_act)
    _add_to_control_dict(params_dict, 'load')
    _add_to_df_actual_generation(params_dict, 'loss_load')
    _add_to_df_actions(params_dict, 'load')
    _add_to_df_status(params_dict, load=round(load_module.current_load, 1))


def add_pv_params(pv_module, params_dict):
    params_dict['pv'] = pd.DataFrame(pv_module.time_series)
    _add_to_architecture(params_dict, 'PV')
    _add_to_parameters(params_dict, PV_rated_power=pv_module.max_act)
    _add_to_df_actions(params_dict, 'pv_consummed','pv_curtailed','pv')
    _add_to_df_status(params_dict, pv=[pv_module.current_renewable])
    _add_to_df_actual_generation(params_dict, 'pv_consummed','pv_curtailed')
    _add_to_control_dict(params_dict, 'pv_consummed', 'pv_curtailed', 'pv')


def add_battery_params(battery_module, params_dict):
    _add_to_architecture(params_dict, 'battery')
    _add_to_parameters(params_dict,
                       battery_soc_0=battery_module.soc,
                       battery_power_charge=battery_module.max_charge,
                       battery_power_discharge=battery_module.max_discharge,
                       battery_capacity=battery_module.max_capacity,
                       battery_efficiency=battery_module.efficiency,
                       battery_soc_min=battery_module.min_soc,
                       battery_soc_max=battery_module.max_soc,
                       battery_cost_cycle=battery_module.battery_cost_cycle)
    _add_to_df_actions(params_dict, 'battery_charge', 'battery_discharge')
    _add_to_df_status(params_dict,
                      battery_soc=battery_module.soc,
                      capa_to_charge=round((battery_module.max_soc-battery_module.soc) *
                                     battery_module.max_capacity/battery_module.efficiency, 1),
                      capa_to_discharge=round((battery_module.soc-battery_module.min_soc) *
                                        battery_module.max_capacity/battery_module.efficiency, 1)
                      )
    _add_to_df_actual_generation(params_dict, 'battery_charge', 'battery_discharge')
    _add_to_df_cost(params_dict, 'battery')
    _add_to_control_dict(params_dict, 'battery_charge','battery_discharge')


def add_grid_params(grid_module, params_dict):
    time_series_df = pd.DataFrame(grid_module.time_series,
                                  columns=['grid_price_import', 'grid_price_export', 'grid_co2', 'grid_status'])
    params_dict['grid_price_import'] = time_series_df['grid_price_import'].to_frame()
    params_dict['grid_price_export'] = time_series_df['grid_price_export'].to_frame()
    params_dict['grid_co2'] = time_series_df['grid_co2'].to_frame()
    params_dict['grid_ts'] = time_series_df['grid_status'].to_frame()
    _add_to_architecture(params_dict, 'grid')
    _add_to_parameters(params_dict,
                       grid_weak=(time_series_df['grid_status'].min() < 1).item(),
                       grid_power_import=grid_module.max_import,
                       grid_power_export=grid_module.max_export)
    _add_to_df_actions(params_dict, 'grid_import','grid_export')
    _add_to_df_status(params_dict,
                      grid_status=time_series_df['grid_status'].iloc[0],
                      grid_co2=time_series_df['grid_co2'].iloc[0],
                      grid_price_import=time_series_df['grid_price_import'].iloc[0],
                      grid_price_export=time_series_df['grid_price_export'].iloc[0]
                      )
    _add_to_df_actual_generation(params_dict, 'grid_import', 'grid_export')
    _add_to_df_cost(params_dict, 'grid_import', 'grid_export')
    _add_to_control_dict(params_dict, 'grid_import', 'grid_export')
    _add_cost_co2(params_dict, grid_module.cost_per_unit_co2)


def add_genset_params(genset_module, params_dict):
    warn('GensetModules does not contain separate rated_power and p_max information.'
         'Assuming p_max=0.9.')
    genset_pmax=0.9
    genset_rated_power = genset_module.running_max_production/genset_pmax
    _add_to_architecture(params_dict, 'genset')
    _add_genset_polynom(params_dict)

    if genset_rated_power == 0:
        raise RuntimeError

    _add_to_parameters(params_dict,
                       genset_rated_power=genset_rated_power,
                       genset_pmin=genset_module.running_min_production/genset_rated_power,
                       genset_pmax=genset_pmax,
                       fuel_cost=genset_module.genset_cost,
                       genset_co2=genset_module.co2_per_unit)
    _add_to_df_actions(params_dict, 'genset')
    _add_to_df_actual_generation(params_dict, 'genset')
    _add_to_df_cost(params_dict, 'genset')
    _add_to_control_dict(params_dict, 'genset')
    _add_cost_co2(params_dict, genset_module.cost_per_unit_co2)


def add_unbalanced_energy_params(unbalanced_energy_module, params_dict):
    _add_to_parameters(params_dict,
                       cost_overgeneration=unbalanced_energy_module.overgeneration_cost,
                       cost_loss_load=unbalanced_energy_module.loss_load_cost
                       )
    _add_to_df_actual_generation(params_dict, 'overgeneration')
    _add_to_df_cost(params_dict, 'overgeneration')


def _add_empty(params_dict, subdict_name, *keys):
    params_dict[subdict_name].update({k: [] for k in keys})


def _add_to_architecture(params_dict, component):
    if component not in params_dict['architecture']:
        raise NameError(f'Component {component} not viable member of architecture')
    params_dict['architecture'][component] = 1


def _add_to_parameters(params_dict, **parameters):
    params_dict['parameters'].update({key: [value] if not isinstance(value, list) else value
                                     for key, value in parameters.items()})


def _add_to_df_actions(params_dict, *keys):
    _add_empty(params_dict, 'df_actions', *keys)


def _add_to_df_status(params_dict, **init_status_values):
    params_dict['df_status'].update({key: [value] if not isinstance(value, list) else value
                                     for key, value in init_status_values.items()})


def _add_to_df_actual_generation(params_dict, *keys):
    _add_empty(params_dict, 'df_actual_generation', *keys)


def _add_to_df_cost(params_dict, *keys):
    _add_empty(params_dict, 'df_cost', *keys)


def _add_to_control_dict(params_dict, *keys):
    params_dict['control_dict'].extend(list(keys))


def _add_cost_co2(params_dict, cost_co2):
    from warnings import warn
    if 'cost_co2' in params_dict['parameters']:
        existing_cost_co2 = params_dict['parameters']['cost_co2']
        if cost_co2 != existing_cost_co2:
            warn(f'cost_co2 value {cost_co2} being added is different from existing cost_co2 value {existing_cost_co2}. Using mean.')
            params_dict['parameters']['cost_co2'] = np.mean([cost_co2, existing_cost_co2])
    else:
        params_dict['parameters']['cost_co2'] = cost_co2
    _add_to_df_cost(params_dict, 'co2')


def _add_genset_polynom(params_dict):
    np.random.seed(0)
    warn('Getting genset_polynom parameters randomly')
    polynom = [np.random.rand() * 10, np.random.rand(), np.random.rand() / 10]

    to_add = dict(genset_polynom_order= len(polynom))
    to_add.update({f'genset_polynom_{i}': pn for i, pn in enumerate(polynom)})
    _add_to_parameters(params_dict, **to_add)
