import numpy as np

from pymgrid import Microgrid

from pymgrid.modules import (
    BatteryModule,
    GensetModule,
    GridModule,
    LoadModule,
    RenewableModule
)


def get_modular_microgrid(remove_modules=(), retain_only=None, additional_modules=None, add_unbalanced_module=True):

    modules = dict(
        genset=GensetModule(running_min_production=10, running_max_production=50, genset_cost=0.5),

        battery=BatteryModule(min_capacity=0,
                              max_capacity=100,
                              max_charge=50,
                              max_discharge=50,
                              efficiency=1.0,
                              init_soc=0.5),

        pv=RenewableModule(time_series=50*np.ones(100)),

        load=LoadModule(time_series=60*np.ones(100)),

        grid=GridModule(max_import=100, max_export=0, time_series=np.ones((100, 3)), raise_errors=True)
        )

    if retain_only is not None:
        modules = {k: v for k, v in modules.items() if k in retain_only}
        if remove_modules:
            raise RuntimeError('Can pass either remove_modules or retain_only, but not both.')
    else:
        for module in remove_modules:
            try:
                modules.pop(module)
            except KeyError:
                raise NameError(f"Module {module} not one of default modules {list(modules.keys())}.")

    modules = list(modules.values())
    modules.extend(additional_modules if additional_modules else [])

    return Microgrid(modules, add_unbalanced_module=add_unbalanced_module)
