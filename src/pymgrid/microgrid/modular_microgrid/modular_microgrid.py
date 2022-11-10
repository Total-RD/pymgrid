import numpy as np
import pandas as pd
import yaml

from copy import deepcopy
from warnings import warn

from pymgrid.microgrid.modules import *
from pymgrid.microgrid.modules.module_container import ModuleContainer
from pymgrid.microgrid.utils.logger import ModularLogger
from pymgrid.microgrid.utils.step import MicrogridStep
from pymgrid.microgrid.utils.serialize import add_numpy_pandas_representers, add_numpy_pandas_constructors, dump_data

DEFAULT_HORIZON = 23


class Microgrid(yaml.YAMLObject):
    yaml_tag = u"!Microgrid"
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    def __init__(self,
                 modules,
                 add_unbalanced_module=True,
                 loss_load_cost=10,
                 overgeneration_cost=2):
        """

        :param modules: list-like. List of _modules or tuples. Latter case: tup(str, Module); str to define name of module
            and second element is the module.

        """

        self._modules = self._get_module_container(modules,
                                                    add_unbalanced_module,
                                                    loss_load_cost,
                                                    overgeneration_cost)
        self._balance_logger = ModularLogger()

    def _get_unbalanced_energy_module(self,
                                      loss_load_cost,
                                      overgeneration_cost):

        return UnbalancedEnergyModule(raise_errors=False,
                                      loss_load_cost=loss_load_cost,
                                      overgeneration_cost=overgeneration_cost
                                      )

    def _get_module_container(self, modules, add_unbalanced_module, loss_load_cost, overgeneration_cost):
        """
        Types of _modules:
        Fixed source: provides energy to the microgrid.
            When queried, the microgrid must absorb said energy.
            Example: battery (when discharging), grid (when importing from)

        Flex source: provides energy to the microgrid.
            Can retain excess energy and not send to microgrid.
            Queried last, in the hopes of balancing other _modules.
            Example: pv with ability to curtail pv

        Fixed sink: absorbs energy from the microgrid.
            When queried, the microgrid must send it said energy.
            Example: load, grid (when exporting to)

        Flexible sink: absorbs energy from the microgrid.
            Can absorb excess energy from the microgrid.
            Queried last, in the hopes of balancing other _modules.
            Example: dispatchable load

        Note that _modules can act as both sources and sinks (batteries, grid), but cannot be both
            fixed and flexible.

        :return:
        """
        modules = deepcopy(modules)

        if not pd.api.types.is_list_like(modules):
            raise TypeError("modules must be list-like of modules.")

        if add_unbalanced_module:
            modules.append(self._get_unbalanced_energy_module(loss_load_cost, overgeneration_cost))

        return ModuleContainer(modules)

    def reset(self):
        return {
            **{name: [module.reset() for module in module_list] for name, module_list in self.modules.iterdict()},
            **{"balance": self._balance_logger.flush()}
        }

    def run(self, control, normalized=True):
        """
        control must contain controls for all fixed _modules in the microgrid.
        Flex _modules are consumed/deployed in the order passed to the microgrid (maybe this should be changed?)
        :param control: dict. keys are names of all fixed _modules
        :return:
        """
        control_copy = control.copy()
        microgrid_step = MicrogridStep()

        for name, modules in self.fixed.iterdict():
            try:
                module_controls = control_copy.pop(name)
            except KeyError:
                raise ValueError(f'Control for module "{name}" not found. Available controls:\n\t{control.keys()}')
            else:
                try:
                    _zip = zip(modules, module_controls)
                except TypeError:
                    _zip = zip(modules, [module_controls])

            for module, _control in _zip:
                module_step = module.step(_control, normalized=normalized) # obs, reward, done, info.
                microgrid_step.append(name, *module_step)

        provided, consumed, _ = microgrid_step.balance()
        difference = provided - consumed                # if difference > 0, have an excess. Try to use flex sinks to dissapate
                                                        # otherwise, insufficient. Use flex sources to make up
        log_dict = self._get_log_dict(provided, consumed, prefix='fixed')

        if len(control_copy) > 0:
            warn(f'\nIgnoring the following keys in passed control:\n {list(control_copy.keys())}')

        if difference > 0:
            energy_excess = difference
            for name, modules in self.flex.iterdict():
                for module in modules:
                    if not module.is_sink:
                        sink_amt = 0.0
                    elif module.max_consumption < energy_excess: # module cannot dissapate all excess energy
                        sink_amt = -1.0*module.max_consumption
                    else:
                        sink_amt = -1.0 * energy_excess

                    module_step = module.step(sink_amt, normalized=False)
                    microgrid_step.append(name, *module_step)
                    energy_excess += sink_amt

        else:
            energy_needed = - difference
            for name, modules in self.flex.iterdict():
                for module in modules:
                    if not module.is_source:
                        source_amt = 0.0
                    elif module.max_production < energy_needed: # module cannot provide sufficient energy
                        source_amt = module.max_production
                    else:
                        source_amt = energy_needed

                    module_step = module.step(source_amt, normalized=False)
                    microgrid_step.append(name, *module_step)
                    energy_needed -= source_amt

        provided, consumed, reward = microgrid_step.balance()
        log_dict = self._get_log_dict(provided, consumed, log_dict=log_dict, prefix='overall')

        self._balance_logger.log(reward=reward, **log_dict)

        if not np.isclose(provided, consumed):
            raise RuntimeError('Microgrid modules unable to balance energy production with consumption.\n'
                               '')

        return microgrid_step.output()

    def _get_log_dict(self, provided_energy, absorbed_energy, log_dict=None, prefix=None):
        _log_dict = dict(provided_to_microgrid=provided_energy, absorbed_by_microgrid=absorbed_energy)
        _log_dict = {(prefix + '_' + k if prefix is not None else k): v for k, v in _log_dict.items()}
        if log_dict:
            _log_dict.update(log_dict)
        return _log_dict

    def sample_action(self, strict_bound=False, sample_flex_modules=False):
        module_iterator = self._modules.module_dict() if sample_flex_modules else self._modules.fixed.module_dict()
        return {module_name: [module.sample_action(strict_bound=strict_bound) for module in module_list] for module_name, module_list in module_iterator.items()}

    def get_empty_action(self, sample_flex_modules=False):
        module_iterator = self._modules.module_dict() if sample_flex_modules else self._modules.fixed.module_dict()

        return {module_name: [None]*len(module_list) for module_name, module_list in module_iterator.items()}


    def to_normalized(self, data_dict, act=False, obs=False):
        assert act + obs == 1, 'One of act or obs must be True but not both.'
        return {module_name: [module.to_normalized(value, act=act, obs=obs) for module, value in zip(module_list, data_dict[module_name])]
                for module_name, module_list in self._modules.iterdict() if module_name in data_dict}

    def from_normalized(self, data_dict, act=False, obs=False):
        assert act + obs == 1, 'One of act or obs must be True but not both.'
        return {module_name: [module.from_normalized(value, act=act, obs=obs) for module, value in zip(module_list, data_dict[module_name])]
                for module_name, module_list in self._modules.iterdict() if module_name in data_dict}

    def get_log(self, as_frame=True, drop_singleton_key=False):
        _log_dict = dict()
        for name, modules in self._modules.iterdict():
            for j, module in enumerate(modules):
                for key, value in module.log_dict().items():
                    _log_dict[(name, j, key)] = value

        for key, value in self._balance_logger.to_dict().items():
            _log_dict[('balance', 0, key)] = value

        if hasattr(self, 'log_dict'):
            for key, value in self.log_dict.items():
                _log_dict[(key, '', '')] = value

        if drop_singleton_key:
            keys_arr = np.array(list(_log_dict.keys()))
            module_counters = keys_arr[:, 1].astype(np.int64)
            if module_counters.min() == module_counters.max():
                _log_dict = {(key[0], key[2]): value for key, value in _log_dict.items()}

        if as_frame:
            return pd.DataFrame(_log_dict)
        return _log_dict

    def get_forecast_horizon(self):
        horizons = []
        for module in self.iterlist():
            try:
                horizons.append(module.forecast_horizon)
            except AttributeError:
                pass

        if len(horizons) == 0:
            warn(f"No forecast horizon found in microgrid.modules. Using default horizon {DEFAULT_HORIZON}")
            return DEFAULT_HORIZON
        elif not np.min(horizons) == np.max(horizons):
                raise ValueError(f"Mismatched forecast_horizons found: {horizons}")

        return horizons[0]

    @property
    def modules(self):
        return self._modules

    @property
    def fixed_modules(self):
        return self._modules.fixed

    @property
    def flex_modules(self):
        return self._modules.flex

    @property
    def flat_modules(self):
        raise AttributeError('Getting attribute flat_modules has been deprecated. Call .modules_dict() instead.')

    @property
    def module_list(self):
        return self._modules.module_list()

    @property
    def n_modules(self):
        return len(self._modules)

    def dump(self, stream=None):
        return yaml.safe_dump(self, stream=stream)

    @classmethod
    def load(cls, stream):
        return yaml.safe_load(stream)

    @classmethod
    def to_yaml(cls, dumper, data):
        add_numpy_pandas_representers()
        return dumper.represent_mapping(cls.yaml_tag, data.serialize(dumper.stream), flow_style=cls.yaml_flow_style)

    @classmethod
    def from_yaml(cls, loader, node):
        add_numpy_pandas_constructors()
        mapping = loader.construct_mapping(node, deep=True)
        instance = cls(mapping["modules"], add_unbalanced_module=False)
        instance._balance_logger = instance._balance_logger.from_raw(mapping["balance_log"])
        return instance

    def serialize(self, dumper_stream):
        data = {"modules": self._modules.module_tuples(),
                "balance_log": self._balance_logger.serialize()}
        return dump_data(data, dumper_stream, self.yaml_tag)

    @classmethod
    def from_nonmodular(cls, nonmodular):
        from pymgrid.microgrid.convert.convert import to_modular
        return to_modular(nonmodular)

    def to_nonmodular(self):
        from pymgrid.microgrid.convert.convert import to_nonmodular
        return to_nonmodular(self)

    @classmethod
    def from_scenario(cls, microgrid_number=0, scenario="pymgrid25"):
        # TODO use better serialization
        import pickle
        from pymgrid import PROJECT_PATH
        with open(PROJECT_PATH / f"data/scenario/{scenario}.pkl", "rb") as f:
            mgen = pickle.load(f)
        return cls.from_nonmodular(mgen.microgrids[microgrid_number])

    def __getnewargs__(self):
        return (self.module_tuples(), )

    def __len__(self):
        """
        Length of available underlying data.
        :return:
        """
        l = []
        for module in self.modules.iterlist():
            try:
                l.append(len(module))
            except TypeError:
                pass

        return min(l)

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented

        return self.modules == other.modules and self._balance_logger == other._balance_logger

    def __repr__(self):
        module_str = [name + ' x ' + str(len(modules)) for name, modules in self._modules.iterdict()]
        module_str = ', '.join(module_str)
        return f'Microgrid([{module_str}])'

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError
        elif item == "_modules":
            raise RuntimeError

        try:
            return getattr(self._modules, item)
        except AttributeError:
            names = ", ".join([f'"{x}"' for x in self.modules.names()])
            raise AttributeError(f'ModularMicrogrid has no attribute "{item}". '
                                 f'Did you mean one of the modules {names}?')
