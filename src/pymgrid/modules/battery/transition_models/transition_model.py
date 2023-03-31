import inspect
import yaml

from abc import abstractmethod


class BatteryTransitionModel(yaml.YAMLObject):
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader
    yaml_tag = u"!BatteryTransitionModel"

    @abstractmethod
    def __call__(self,
                 external_energy_change,
                 min_capacity,
                 max_capacity,
                 max_charge,
                 max_discharge,
                 efficiency,
                 battery_cost_cycle,
                 state_dict):
        if external_energy_change < 0:
            return external_energy_change / efficiency
        else:
            return external_energy_change * efficiency

    def __repr__(self):
        params = inspect.signature(self.__init__).parameters
        formatted_params = ', '.join([f'{p}={getattr(self, p)}' for p in params])
        return f'{self.__class__.__name__}({formatted_params})'

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return repr(self) == repr(other)
