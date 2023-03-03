import inspect
import yaml

from abc import abstractmethod


class BaseTrajectory(yaml.YAMLObject):
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    @abstractmethod
    def __call__(self, initial_step, final_step):
        pass

    def __repr__(self):
        params = inspect.signature(self.__init__).parameters
        formatted_params = ', '.join([f'{p}={getattr(self, p)}' for p in params])
        return f'{self.__class__.__name__}({formatted_params})'

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return repr(self) == repr(other)
