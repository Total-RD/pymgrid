from pymgrid.microgrid.modular_microgrid.modular_microgrid import ModularMicrogrid
from pymgrid.microgrid.modules import *
import numpy as np


def get_modular_microgrid(remove_modules=(), additional_modules=None,):

    modules = dict(
        genset=GensetModule(running_min_production=10, running_max_production=50, genset_cost=0.5),

        battery=BatteryModule(min_capacity=0,
                              max_capacity=100,
                              max_charge=50,
                              max_discharge=50,
                              efficiency=1.0,
                              init_soc=0.5),

        pv=RenewableModule(time_series=50*np.ones(100)),

        load=LoadModule(time_series=60*np.ones(100),
                        loss_load_cost=10),

        grid=GridModule(max_import=100,
                        max_export=0,
                        time_series_cost_co2=np.ones((100, 3)),
                        raise_errors=True)
        )

    for module in remove_modules:
        try:
            modules.pop(module)
        except KeyError:
            raise NameError(f"Module {module} not one of default modules {list(module.keys())}.")

    modules = list(modules.values())
    modules.extend(additional_modules if additional_modules else [])

    return ModularMicrogrid(modules)
