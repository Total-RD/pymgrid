import pandas as pd

def normalize_environment_states(mg):
    max_values = {}
    for keys in mg._df_record_state:
        if keys == 'hour':
            max_values[keys] = 24
        elif keys == 'capa_to_charge' or keys == 'capa_to_discharge' :
            max_values[keys] = mg.parameters.battery_capacity.values[0]
        elif keys == 'grid_status' or keys == 'battery_soc':
            max_values[keys] = 1
        elif keys == 'grid_co2':
            max_values[keys] = max(mg._grid_co2.values[0])
        elif keys == 'grid_price_import':
            max_values[keys] = max(mg._grid_price_import.values[0]) 
        elif keys == 'grid_price_export':
            max_values[keys] = max(mg._grid_price_import.values[0]) 
        elif keys == 'load':
            max_values[keys] = mg.parameters.load.values[0]
        elif keys == 'pv':
            max_values[keys] = mg.parameters.PV_rated_power.values[0]
        else:
            max_values[keys] = mg.parameters[keys].values[0] 
            
    return max_values

def sample_reset(has_grid, saa, microgrid, sampling_args=None):
    """
    Generates a new sample using an instance of SampleAverageApproximation and
    :param has_grid: bool, whether the microgrid has a grid.
    :param saa:, SampleAverageApproximation
    :param microgrid: Microgrid
    :param sampling_args: arguments to be passed to saa.sample_from_forecasts().
    :return:
    """
    if sampling_args is None:
        sampling_args = dict()

    sample = saa.sample_from_forecasts(n_samples=1, **sampling_args)
    sample = sample[0]

    microgrid._load_ts = pd.DataFrame(sample['load'])
    microgrid._pv_ts = pd.DataFrame(sample['pv'])
    microgrid._df_record_state['load'] = [sample['load'].iloc[0].squeeze()]
    microgrid._df_record_state['pv'] = [sample['pv'].iloc[0].squeeze()]
    if has_grid:
        microgrid._grid_status_ts = pd.DataFrame(sample['grid'])
        microgrid._df_record_state['grid_status'] = [sample['grid'].iloc[0].squeeze()]