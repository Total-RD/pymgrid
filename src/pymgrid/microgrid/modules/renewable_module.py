import numpy as np
from src.pymgrid.microgrid.modules.base_module import BaseTimeSeriesMicrogridModule


class RenewableModule(BaseTimeSeriesMicrogridModule):
    module_type = ('renewable', 'flex')

    def __init__(self, time_series,
                 raise_errors=False,
                 provided_energy_name='renewable_used'):
        super().__init__(time_series,
                         raise_errors,
                         provided_energy_name=provided_energy_name,
                         absorbed_energy_name=None)
        self.name = ('renewable', None)

    def update(self, external_energy_change, as_source=False, as_sink=False):
        assert as_source, f'Class {self.__class__.__name__} can only be used as a source.'
        assert external_energy_change <= self.current_renewable, f'Cannot provide more than {self.current_renewable}'
        try:
            next_renewable = self._time_series[self.current_step+1]
            done = False
        except IndexError:
            next_renewable = np.nan
            done = True

        info = {'provided_energy': external_energy_change, 'curtailment': self.current_renewable-external_energy_change}

        return np.array(next_renewable), 0.0, done, info

    def state_dict(self):
        return dict(current_renewable=self.current_renewable.item())

    @property
    def max_production(self):
        return self.current_renewable

    @property
    def current_renewable(self):
        return self._time_series[self._current_step]

    @property
    def is_source(self):
        return True
