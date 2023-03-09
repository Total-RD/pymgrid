import yaml

from abc import abstractmethod


class BaseRewardShaper(yaml.YAMLObject):
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    @staticmethod
    def sum_module_val(info, module_name, module_attr):
        try:
            module_info = info[module_name]
            return sum([d[module_attr] for d in module_info])
        except KeyError:
            return 0.0

    @abstractmethod
    def __call__(self, step_info, cost_info):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}()'
