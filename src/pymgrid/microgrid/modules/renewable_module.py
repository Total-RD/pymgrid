import numpy as np
import yaml

from pymgrid.microgrid.modules.base import BaseTimeSeriesMicrogridModule


class RenewableModule(BaseTimeSeriesMicrogridModule):
    module_type = ('renewable', 'flex')
    yaml_tag = u"!RenewableModule"
    yaml_loader = yaml.SafeLoader
    yaml_dumper = yaml.SafeDumper

    def __init__(self, time_series,
                 raise_errors=False,
                 forecaster=None,
                 forecast_horizon=24,
                 forecaster_increase_uncertainty=False,
                 provided_energy_name='renewable_used'):
        super().__init__(time_series,
                         raise_errors,
                         forecaster=forecaster,
                         forecast_horizon=forecast_horizon,
                         forecaster_increase_uncertainty=forecaster_increase_uncertainty,
                         provided_energy_name=provided_energy_name,
                         absorbed_energy_name=None)
        self.name = ('renewable', None)

    def update(self, external_energy_change, as_source=False, as_sink=False):
        assert as_source, f'Class {self.__class__.__name__} can only be used as a source.'
        assert external_energy_change <= self.current_renewable, f'Cannot provide more than {self.current_renewable}'

        info = {'provided_energy': external_energy_change, 'curtailment': self.current_renewable.item()-external_energy_change}

        return 0.0, self._done(), info

    @property
    def state_components(self):
        return np.array(["renewable"], dtype=object)

    @property
    def max_production(self):
        return self.current_renewable

    @property
    def current_renewable(self):
        return self._time_series[self._current_step]

    @property
    def is_source(self):
        return True
