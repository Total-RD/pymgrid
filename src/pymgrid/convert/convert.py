from pymgrid._deprecated.non_modular_microgrid import NonModularMicrogrid
from pymgrid.microgrid.microgrid import Microgrid
from pymgrid.convert.get_module import get_module
from pymgrid.convert.to_nonmodular_ops import check_viability, add_params_from_module, get_empty_params, finalize_params


def to_modular(nonmodular, raise_errors=False):
    modules = [('load', get_module('load', nonmodular, raise_errors)),
               ('unbalanced_energy', get_module('unbalanced_energy', nonmodular, raise_errors))]
    for component, exists in nonmodular.architecture.items():
        if exists:
            module = get_module(component, nonmodular, raise_errors)
            modules.append((component, module))
    return Microgrid(modules, add_unbalanced_module=False)


def to_nonmodular(modular):
    """
    microgrid_params needs to contain the following:
        parameters: ('DataFrame', (1, 16))
        df_actions: ('dict', {'load': [], 'pv_consummed': [], 'pv_curtailed': [], 'pv': [], 'battery_charge': [], 'battery_discharge': [], 'grid_import': [], 'grid_export': []})
        architecture: ('dict', {'PV': 1, 'battery': 1, 'genset': 0, 'grid': 1})
        df_status: ('dict', {'load': [304.4], 'hour': [0], 'pv': [0.0], 'battery_soc': [0.2], 'capa_to_charge': [1290.7], 'capa_to_discharge': [0.0], 'grid_status': [1.0], 'grid_co2': [0.23975790800000002], 'grid_price_import': [0.22], 'grid_price_export': [0.0]})
        df_actual_generation: ('dict', {'loss_load': [], 'overgeneration': [], 'pv_consummed': [], 'pv_curtailed': [], 'battery_charge': [], 'battery_discharge': [], 'grid_import': [], 'grid_export': []})
        grid_spec: ('int', 0)
        df_cost: ('dict', {'loss_load': [], 'overgeneration': [], 'co2': [], 'battery': [], 'grid_import': [], 'grid_export': [], 'total_cost': []})
        df_co2: ('dict', {'co2': []})
        pv: ('DataFrame', (8760, 1))
        load: ('DataFrame', (8760, 1))
        grid_ts: ('DataFrame', (8760, 1))
        control_dict: ('list', ['load', 'pv_consummed', 'pv_curtailed', 'pv', 'battery_charge', 'battery_discharge', 'grid_import', 'grid_export'])
        grid_price_import: ('DataFrame', (8760, 1))
        grid_price_export: ('DataFrame', (8760, 1))
        grid_co2: ('DataFrame', (8760, 1))
    :param modular:
    :return:
    """
    check_viability(modular)
    microgrid_params = get_empty_params()
    for _, module_list in modular.modules.iterdict():
        add_params_from_module(module_list[0], microgrid_params)
    finalize_params(microgrid_params)
    return NonModularMicrogrid(parameters=microgrid_params, horizon=modular.get_forecast_horizon() + 1)
