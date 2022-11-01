from pymgrid.microgrid.modules import GensetModule


default_params = dict(running_min_production=10,
                              running_max_production=100,
                              genset_cost=1,
                              start_up_time=0,
                              wind_down_time=0,
                              allow_abortion=True,
                              init_start_up=True,
                              raise_errors=True)


def get_genset(default_parameters=None, **new_params):
    params = default_parameters.copy() if default_parameters is not None else default_params.copy()
    params.update(new_params)
    return GensetModule(**params), params


def normalize_production(production, max_production=None):
    max_production = max_production if max_production else default_params['running_max_production']
    return production/max_production
