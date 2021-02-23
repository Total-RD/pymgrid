

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