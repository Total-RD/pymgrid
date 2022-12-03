import pandas as pd
from pymgrid.modules import LoadModule, RenewableModule, BatteryModule, GridModule, GensetModule, UnbalancedEnergyModule


def get_module(component, nonmodular, raise_errors):
    if component == 'load':
        return get_load_module(nonmodular, raise_errors)
    elif component == 'PV':
        return get_pv_module(nonmodular, raise_errors)
    elif component == 'battery':
        return get_battery_module(nonmodular, raise_errors)
    elif component == 'genset':
        return get_genset_module(nonmodular, raise_errors)
    elif component == 'grid':
        return get_grid_module(nonmodular, raise_errors)
    elif component == 'unbalanced_energy':
        return get_unbalanced_energy_module(nonmodular, raise_errors)
    else:
        raise ValueError(f'Cannot parse component {component}.')


def get_load_module(nonmodular, raise_errors):
    time_series = nonmodular._load_ts
    return LoadModule(time_series=time_series,
                      forecaster='oracle',
                      forecast_horizon=nonmodular.horizon-1,
                      raise_errors=raise_errors)


def get_pv_module(nonmodular, raise_errors):
    time_series = nonmodular._pv_ts
    return RenewableModule(time_series=time_series,
                           raise_errors=raise_errors,
                           forecaster='oracle',
                           forecast_horizon=nonmodular.horizon-1
                           )


def get_battery_module(nonmodular, raise_errors):
    battery = nonmodular.battery
    max_capacity = battery.capacity
    min_capacity = max_capacity*battery.soc_min
    max_charge = battery.p_charge_max
    max_discharge = battery.p_discharge_max
    efficiency = battery.efficiency
    battery_cost_cycle = battery.cost_cycle
    init_soc = battery.soc
    return BatteryModule(min_capacity=min_capacity,
                         max_capacity=max_capacity,
                         max_charge=max_charge,
                         max_discharge=max_discharge,
                         efficiency=efficiency,
                         battery_cost_cycle=battery_cost_cycle,
                         init_soc=init_soc,
                         raise_errors=raise_errors)


def get_genset_module(nonmodular, raise_errors):
    genset = nonmodular.genset
    min_production = genset.p_min*genset.rated_power
    max_production = genset.p_max*genset.rated_power
    genset_cost = genset.fuel_cost
    co2_per_unit = nonmodular.parameters.genset_co2.item()
    cost_per_unit_co2 = nonmodular.parameters.cost_co2.item()
    return GensetModule(running_min_production=min_production,
                        running_max_production=max_production,
                        genset_cost=genset_cost,
                        co2_per_unit=co2_per_unit,
                        cost_per_unit_co2=cost_per_unit_co2,
                        start_up_time=0,
                        wind_down_time=0,
                        raise_errors=raise_errors)


def get_grid_module(nonmodular, raise_errors):
    max_import = nonmodular.grid.power_import
    max_export = nonmodular.grid.power_export
    cost_per_unit_co2 = nonmodular.parameters.cost_co2.item()

    cost_import = nonmodular._grid_price_import.squeeze()
    cost_import.name = 'cost_import'
    cost_export = nonmodular._grid_price_export.squeeze()
    cost_export.name = 'cost_export'
    co2_per_unit = nonmodular._grid_co2.squeeze()
    co2_per_unit.name = 'co2_per_unit_production'
    grid_status = nonmodular._grid_status_ts.squeeze()
    grid_status.name = 'grid_status'
    time_series = pd.concat([cost_import, cost_export, co2_per_unit, grid_status], axis=1)

    return GridModule(max_import=max_import,
                      max_export=max_export,
                      time_series=time_series,
                      forecaster='oracle',
                      forecast_horizon=nonmodular.horizon - 1,
                      cost_per_unit_co2=cost_per_unit_co2,
                      raise_errors=raise_errors)


def get_unbalanced_energy_module(nonmodular, raise_errors):
    return UnbalancedEnergyModule(raise_errors=raise_errors,
                                  loss_load_cost=nonmodular.parameters['cost_loss_load'].item(),
                                  overgeneration_cost=nonmodular.parameters['cost_overgeneration'].item()
                                  )