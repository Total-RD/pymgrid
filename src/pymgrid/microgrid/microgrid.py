import numpy as np
import pandas as pd
import yaml

from copy import deepcopy
from warnings import warn

from pymgrid.microgrid import DEFAULT_HORIZON
from pymgrid.modules import ModuleContainer, UnbalancedEnergyModule
from pymgrid.utils.logger import ModularLogger
from pymgrid.microgrid.utils.step import MicrogridStep
from pymgrid.utils.serialize import add_numpy_pandas_representers, add_numpy_pandas_constructors, dump_data


class Microgrid(yaml.YAMLObject):
    """
    Microgrid class, used to define and simulate an environment with a variety of modules.

    Parameters
    ----------
    modules : List[Union[Tuple[str, BaseMicrogridModule], BaseMicrogridModule]]
        List of modules that define the microgrid. The list can contain either/both microgrid modules -- subclasses of
        ``BaseMicrogridModule`` -- and tuples of length two, which must contain a string defining the name of the module
        followed by the module.

        ``Microgrid`` groups modules into lists based on their names. If no name is given (e.g. an element in ``modules``
        is a subclass of ``BaseMicrogridModule`` and not a tuple, then the name is defined to be
        ``module.__class__.name[0]``. Modules are then exposed (within lists) by name as attributes to the microgrid.
        See below for an example.

        .. note::
        The constructor copies modules passed to it.

    add_unbalanced_module : bool, default True.
        Whether to add an unbalanced energy module to your microgrid. Such a module computes and attributes
        costs to any excess supply or demand.
        Set to True unless ``modules`` contains an ``UnbalancedEnergyModule``.

    loss_load_cost : float, default 10.0
        Cost per unit of unmet demand. Ignored if ``add_unbalanced_module=False``.

    overgeneration_cost : float, default 2.0
        Cost per unit of excess generation.  Ignored if ``add_unbalanced_module=False``.

    Examples
    --------
    >>> from pymgrid import Microgrid
    >>> from pymgrid.modules import LoadModule, RenewableModule, GridModule, BatteryModule
    >>> timesteps = 10
    >>> load = LoadModule(10*np.random.rand(timesteps), loss_load_cost=10.)
    >>> pv = RenewableModule(10*np.random.rand(timesteps))
    >>> grid = GridModule(max_import=100, max_export=10, time_series=np.random.rand(timesteps, 3))
    >>> battery_0 = BatteryModule(min_capacity=0, \
                                  max_capacity=100, \
                                  max_charge=1,\
                                  max_discharge=10, \
                                  efficiency=0.9, \
                                  init_soc=0.5)
    >>> battery_1 = BatteryModule(min_capacity=1, \
                                  max_capacity=20, \
                                  max_charge=5, \
                                  max_discharge=10, \
                                  efficiency=0.9, \
                                  init_soc=0.5)

    >>> microgrid = Microgrid(modules=[load, ('pv', pv), grid, battery_0, battery_1])
    >>> # The modules are now available as attributes. The exception to this is `load`, which is an exposed method.
    >>> print(microgrid.pv)
    [RenewableModule(time_series=<class 'numpy.ndarray'>, raise_errors=False, forecaster=NoForecaster, forecast_horizon=0, forecaster_increase_uncertainty=False, provided_energy_name=renewable_used)]
    >>> print(microgrid.grid)
    [GridModule(max_import=100, max_export=10)]
    >>> print(microgrid.grid.item()) # Return the module instead of a list containing the module, if list has one item.
    GridModule(max_import=100, max_export=10)
    >>> for j, battery in enumerate(microgrid.battery):
    >>>     print(f"Battery {j}: {battery}")
    Battery 0: BatteryModule(min_capacity=0, max_capacity=100, max_charge=1, max_discharge=10, efficiency=0.9, battery_cost_cycle=0.0, battery_transition_model=None, init_charge=None, init_soc=0.5, raise_errors=False)
    Battery 1: BatteryModule(min_capacity=1, max_capacity=20, max_charge=5, max_discharge=10, efficiency=0.9, battery_cost_cycle=0.0, battery_transition_model=None, init_charge=None, init_soc=0.5, raise_errors=False)

    """

    yaml_tag = u"!Microgrid"
    """Tag used for yaml serialization."""
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    def __init__(self,
                 modules,
                 add_unbalanced_module=True,
                 loss_load_cost=10.,
                 overgeneration_cost=2.):
        self._modules = self._get_module_container(modules,
                                                   add_unbalanced_module,
                                                   loss_load_cost,
                                                   overgeneration_cost)

        self._balance_logger = ModularLogger()
        self._microgrid_logger = ModularLogger()  # log additional information.

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
        """
        Reset the microgrid and flush the log.

        Returns
        -------
        dict[str, list[float]]
            Observations from resetting the modules as well as the flushed balance log.
        """
        return {
            **{name: [module.reset() for module in module_list] for name, module_list in self.modules.iterdict()},
            **{"balance": self._balance_logger.flush(),
               "other": self._microgrid_logger.flush()}
        }

    def run(self, control, normalized=True):
        """

        Run the microgrid for a single step.

        Parameters
        ----------
        control : dict[str, list[float]]
            Actions to pass to each fixed module.
        normalized : bool, default True
            Whether ``control`` is a normalized value or not. If not, each module de-normalizes its respective action.

        Returns
        -------
        observation : dict[str, list[float]]
            Observations of each module after using the passed ``control``.
        reward : float
            Reward/cost of running the microgrid. A positive value implies revenue while a negative
            value is a cost.
        done : bool
            Whether the microgrid terminates.
        info : dict
            Additional information from this step.

        """
        control_copy = control.copy()
        microgrid_step = MicrogridStep()

        for name, modules in self.fixed.iterdict():
            for module in modules:
                microgrid_step.append(name, *module.step(0.0, normalized=False))

        fixed_provided, fixed_consumed, _ = microgrid_step.balance()
        log_dict = self._get_log_dict(fixed_provided, fixed_consumed, prefix='fixed')

        for name, modules in self.controllable.iterdict():
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
                module_step = module.step(_control, normalized=normalized)  # obs, reward, done, info.
                microgrid_step.append(name, *module_step)

        provided, consumed, _ = microgrid_step.balance()
        difference = provided - consumed                # if difference > 0, have an excess. Try to use flex sinks to dissapate
                                                        # otherwise, insufficient. Use flex sources to make up

        log_dict = self._get_log_dict(provided-fixed_provided, consumed-fixed_consumed, log_dict=log_dict, prefix='controllable')

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
        _log_dict = dict(provided_to_microgrid=provided_energy, absorbed_from_microgrid=absorbed_energy)
        _log_dict = {(prefix + '_' + k if prefix is not None else k): v for k, v in _log_dict.items()}
        if log_dict:
            _log_dict.update(log_dict)
        return _log_dict

    def sample_action(self, strict_bound=False, sample_flex_modules=False):
        """
        Get a random action within the microgrid's action space.

        Parameters
        ----------
        strict_bound : bool, default False
            If True, choose actions that is guaranteed to satisfy self.max_consumption and
            self.max_production bounds. Otherwise selects action from min_act and min_act, which may not satisfy
            instantaneous bounds.
        sample_flex_modules : bool, default false
            Whether to sample the flex modules in the microgrid.
            ``run`` does not expect actions for flex modules.

        Returns
        -------

        dict[str, list[float]]
            Random action in the action space.

        """

        module_iterator = self._modules.to_dict() if sample_flex_modules else self._modules.controllable.to_dict()
        return {module_name: [module.sample_action(strict_bound=strict_bound) for module in module_list]
                for module_name, module_list in module_iterator.items()
                if module_list[0].action_space.shape[0]}

    def get_empty_action(self, sample_flex_modules=False):
        """
        Get an action for the microgrid with no values set.

        Values are all ``None``; every ``None`` value should be replaced before passing an action to ``run``.

        Parameters
        ----------
        sample_flex_modules : bool, default false
            Whether to sample the flex modules in the microgrid.
            ``run`` does not expect actions for flex modules.

        Returns
        -------

        dict[str, list[None]]
            Empty action.

        """
        module_iterator = self._modules.to_dict() if sample_flex_modules else self._modules.controllable.to_dict()

        return {module_name: [None]*len(module_list) for module_name, module_list in module_iterator.items()
                if module_list[0].action_space.shape[0]}

    def to_normalized(self, data_dict, act=False, obs=False):
        """
        Normalize an action or observation.

        Parameters
        ----------
        data_dict : dict[str, list[int]]
            Action or observation to normalize. Dictionary keys are names of the modules while dictionary values
            are lists containing an action corresponding to all modules with that name.
        act : bool, default False
            Set to True if you are normalizing an action.
        obs : bool, default False
            Set to True if you are normalizing an observation.

        Returns
        -------
        dict[str, list[float]]
            Normalized action.
        """
        assert act + obs == 1, 'One of act or obs must be True but not both.'
        return {module_name: [module.to_normalized(value, act=act, obs=obs) for module, value in zip(module_list, data_dict[module_name])]
                for module_name, module_list in self._modules.iterdict() if module_name in data_dict}

    def from_normalized(self, data_dict, act=False, obs=False):
        """
        De-normalize an action or observation.

        Parameters
        ----------
        data_dict : dict[str, list[int]]
            Action or observation to de-normalize. Dictionary keys are names of the modules while dictionary values
            are lists containing an action corresponding to all modules with that name.
        act : bool, default False
            Set to True if you are de-normalizing an action.
        obs : bool, default False
            Set to True if you are de-normalizing an observation.

        Returns
        -------
        dict[str, list[float]]
            De-normalized action.
        """
        assert act + obs == 1, 'One of act or obs must be True but not both.'
        return {module_name: [module.from_normalized(value, act=act, obs=obs) for module, value in zip(module_list, data_dict[module_name])]
                for module_name, module_list in self._modules.iterdict() if module_name in data_dict}

    def get_log(self, as_frame=True, drop_singleton_key=False):
        """

        Collect a log of controls and responses of the microgrid.

        Parameters
        ----------
        as_frame : bool, default True
            Whether to return the log as a pd.DataFrame. If False, returns a nested dict.
        drop_singleton_key : bool, default False
            Whether to drop index level enumerating the modules by name if each module name has only one module.
            Ignored otherwise.

        Returns
        -------
        pd.DataFrame or dict

        """
        _log_dict = dict()
        for name, modules in self._modules.iterdict():
            for j, module in enumerate(modules):
                for key, value in module.log_dict().items():
                    _log_dict[(name, j, key)] = value

        for key, value in self._balance_logger.to_dict().items():
            _log_dict[('balance', 0, key)] = value

        for key, value in self._microgrid_logger.items():
            _log_dict[(key, 0, '')] = value

        col_names = ['module_name', 'module_number', 'field']

        df = pd.DataFrame(_log_dict)
        df.columns = pd.MultiIndex.from_tuples(df.columns.to_list(), names=col_names)

        if drop_singleton_key:
            df.columns = df.columns.remove_unused_levels()

        if as_frame:
            return df

        return df.to_dict()

    def get_forecast_horizon(self):
        """
        Get the forecast horizon of timeseries modules contained in the microgrid.

        Returns
        -------
        int
            The forecast horizon.

        Raises
        ------
        ValueError
            If horizons between modules are inconsistent.

        """
        horizons = []
        for module in self._modules.iterlist():
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
        """
        View of the module container.

        Returns
        -------
        modules : :class:`pymgrid.modules.module_container.ModuleContainer`
            View of the container.

        """
        return self._modules

    @property
    def state_dict(self):
        """
        State of the microgrid as a dict.

        Keys are module names and values are lists of state dicts for all modules with said name.

        Returns
        -------
        state_dict : dict[str, list[dict]]
            State of the microgrid as a nested dict.

        """
        return {name: [module.state_dict for module in modules] for name, modules in self._modules.iterdict()}

    @property
    def log(self):
        """
        Microgrid's log as a DataFrame.

        This is equivalent to `:meth:`get_log`.
        Returns
        -------
        log : pd.DataFrame
            The log of the microgrid.

        """
        return self.get_log()

    @property
    def state_series(self):
        """
        State of the microgrid as a pandas Series.

        Three are three levels in the MultiIndex: ``microgrid_name``, ``microgrid_number``
        (relative to each ``microgrid_name``) and state key name.

        Returns
        -------
        state : pd.Series
            State of the microgrid as a pandas Series..

        """
        return pd.Series(
            {
                (name, num, key): value
                for name, sd_list in self.state_dict.items()
                for num, sd in enumerate(sd_list)
                for key, value in sd.items()
            }
        )

    @property
    def fixed(self):
        """
        List of all fixed modules in the microgrid.

        Returns
        -------
        list of modules
        """
        return self._modules.fixed

    @property
    def flex(self):
        """
        List of all flex modules in the microgrid.

        Returns
        -------
        list
            The list of modules
        """
        return self._modules.flex

    @property
    def controllable(self):
        """
        List of all controllable modules in the microgrid.

        Returns
        -------
        list of modules
        """
        return self._modules.controllable

    @property
    def module_list(self):
        """
        List of all modules in the microgrid.

        Returns
        -------
        list
            The list of modules

        """
        return self._modules.to_list()

    @property
    def n_modules(self):
        """
        Number of modules in the microgrid.

        Returns
        -------
        int
        """
        return len(self._modules)

    def dump(self, stream=None):
        """
        Save a microgrid to a YAML buffer.

        Supports both strings of YAML or storing YAML in a path-like object.

        Parameters
        ----------
        stream : file-like object or None, default None
            Stream to save the YAML document. If None, returns the document instead.

        Returns
        -------
        str or None :
            Returns the YAMl document as a string if ``stream=None``. Returns None otherwise

        .. note::

            ``dump`` handles the serialization of array-like objects (e.g. time series and logs) differently depending
            on the value of ``stream``.  If ``stream is None``, array-like objects are serialized inline. If ``stream`` is
            a stream to a file-like object, however, array-like objects will be serialized as `.csv.gz` files in a
            directory relative to ``stream``, and the relative locations stored inline in the YAML file. For an example of
            this behavior, see `data/scenario/pymgrid25/microgrid_0`.

        """
        return yaml.safe_dump(self, stream=stream)

    @classmethod
    def load(cls, stream):
        """
        Load a microgrid from a yaml buffer.

        Supports both strings of YAML or YAML stored in a path-like object.

        Parameters
        ----------
        stream : str or file-like object
            YAML document. Can be either a string of loaded YAML or a stream to a local file containing a YAML document.

        Returns
        -------
        Microgrid : the loaded microgrid.

        """
        return yaml.safe_load(stream)

    @classmethod
    def to_yaml(cls, dumper, data):
        """
        :meta private:
        """
        add_numpy_pandas_representers()
        return dumper.represent_mapping(cls.yaml_tag, data.serialize(dumper.stream), flow_style=cls.yaml_flow_style)

    @classmethod
    def from_yaml(cls, loader, node):
        """
        :meta private:
        """
        add_numpy_pandas_constructors()
        mapping = loader.construct_mapping(node, deep=True)
        instance = cls(mapping["modules"], add_unbalanced_module=False)
        instance._balance_logger = instance._balance_logger.from_raw(mapping["balance_log"])
        return instance

    def serialize(self, dumper_stream):
        """
        :meta private:
        """
        data = {"modules": self._modules.to_tuples(),
                "balance_log": self._balance_logger.serialize()}
        return dump_data(data, dumper_stream, self.yaml_tag)

    @classmethod
    def from_nonmodular(cls, nonmodular):
        """
        Convert to Microgrid from old-style NonModularMicrogrid.

        Parameters
        ----------
        nonmodular : pymgrid.NonModularMicrogrid
            Non-modular (old-style) microgrid to be converted.

        Returns
        -------
        converted : pymgrid.Microgrid
            New-style modular microgrid.

        See Also
        --------
        pymgrid.Microgrid.to_nonmodular : Converter from new-style to old-style.

        .. warning::

            Any logs that have accumulated will be lost in conversion.

        """
        from pymgrid.convert.convert import to_modular
        return to_modular(nonmodular)

    def to_nonmodular(self):
        """
        Convert Microgrid to old-style NonModularMicrogrid.

        Returns
        -------
        converted : pymgrid.NonModularMicrogrid
            Old-style microgrid.

        See Also
        --------
        :meth:`Microgrid.to_modular` : Converter from old-style to new-style.

        .. warning::

            Any logs that have accumulated will be lost in conversion.

        """
        from pymgrid.convert.convert import to_nonmodular
        return to_nonmodular(self)

    @classmethod
    def from_scenario(cls, microgrid_number=0):
        """
        Load one of the *pymgrid25* benchmark microgrids.

        Parameters
        ----------
        microgrid_number : int, default 0
            Number of the microgrid to return. ``0<=microgrid_number<25``.

        Returns
        -------
        scenario : pymgrid.Microgrid
            The loaded microgrid.
        """
        from pymgrid import PROJECT_PATH
        n = microgrid_number

        if n not in np.arange(25):
            raise TypeError(f'Invalid microgrid_number {n}, must be an integer in the range [0, 25).')

        with open(PROJECT_PATH / f"data/scenario/pymgrid25/microgrid_{n}/microgrid_{n}.yaml", "r") as f:
            return cls.load(f)

    def __getnewargs__(self):
        return (self.modules.to_tuples(), )

    def __len__(self):
        """
        Length of available underlying data.
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
                                 f'Did you mean one of the modules {names}?').with_traceback(None)
