from src.pymgrid.microgrid.modules.base import BaseTimeSeriesMicrogridModule
import numpy as np


class GridModule(BaseTimeSeriesMicrogridModule):
    module_type = ('grid', 'fixed')

    def __init__(self,
                 max_import,
                 max_export,
                 time_series_cost_co2,
                 cost_per_unit_co2=0,
                 raise_errors=False):

        self._check_params(max_import, max_export, time_series_cost_co2)
        self.max_import, self.max_export = max_import, max_export
        self.cost_per_unit_co2 = cost_per_unit_co2
        self.name = ('grid', None)
        super().__init__(time_series_cost_co2,
                         raise_errors,
                         provided_energy_name='grid_import',
                         absorbed_energy_name='grid_export')

    def _check_params(self, max_import, max_export, time_series):
        if max_import < 0:
            raise ValueError('parameter max_import must be non-negative.')
        if max_export < 0:
            raise ValueError('parameter max_export must be non-negative.')
        if time_series.shape[1] != 3:
            raise ValueError('Time series must be two dimensional with three columns: import costs, export costs, '
                             'and co2 production per kWh.')

        if (time_series < 0).any().any():
            raise ValueError('Time series must be non-negative.')

    def get_bounds(self):
        min_obs = self._time_series.min(axis=0)
        max_obs = self._time_series.max(axis=0)
        assert len(min_obs) == 3

        min_act, max_act = -1 * self.max_export, self.max_import

        return min_obs, max_obs, min_act, max_act

    def update(self, external_energy_change, as_source=False, as_sink=False):
        assert as_source + as_sink == 1, 'Must act as either source or sink but not both or neither.'

        try:
            next_costs = self._time_series[self.current_step+1]
            done = False
        except IndexError:
            next_costs = np.array([np.nan]*3)
            done = True

        reward = self.get_cost(external_energy_change, as_source, as_sink)
        info_key = 'provided_energy' if as_source else 'absorbed_energy'
        info = {info_key: external_energy_change,
                'co2_production': self.get_co2_production(external_energy_change, as_source, as_sink)}
        return reward, done, info

    def get_cost(self, import_export, as_source, as_sink):
        if as_source:                                               # Import
            import_cost = self._time_series[self.current_step, 0]
            return -1 * import_cost*import_export + self.get_co2_cost(import_export, as_source, as_sink)
        elif as_sink:                                               # Export
            export_cost = self._time_series[self.current_step, 1]
            return export_cost * import_export + self.get_co2_cost(import_export, as_source, as_sink)
        else:
            raise RuntimeError

    def get_co2_cost(self, import_export, as_source, as_sink):
        return -1.0 * self.cost_per_unit_co2*self.get_co2_production(import_export, as_source, as_sink)

    def get_co2_production(self, import_export, as_source, as_sink):
        if as_source:                                               # Import
            co2_prod_per_kWh = self._time_series[self.current_step, 2]
            co2 = import_export*co2_prod_per_kWh
            return co2
        elif as_sink:                                               # Export
            return 0.0
        else:
            raise RuntimeError

    def as_flex(self):
        self.__class__.module_type = (self.__class__.module_type[0], 'flex')

    def as_fixed(self):
        self.__class__.module_type = (self.__class__.module_type[0], 'fixed')

    @property
    def state_components(self):
        return np.array(['import_cost', 'export_cost', 'co2_per_kwh'], dtype=object)

    @property
    def max_production(self):
        return self.max_import

    @property
    def max_consumption(self):
        return self.max_export

    @property
    def is_source(self):
        return True

    @property
    def is_sink(self):
        return True

    def __repr__(self):
        return f'GridModule(max_import={self.max_import}, max_export={self.max_export}'
