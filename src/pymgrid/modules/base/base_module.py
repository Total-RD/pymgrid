from abc import abstractmethod
import inspect
import logging
import yaml
import numpy as np

from warnings import warn

from pymgrid.utils.logger import ModularLogger
from pymgrid.utils.space import ModuleSpace
from pymgrid.utils.serialize import add_numpy_pandas_representers, add_numpy_pandas_constructors, dump_data


script_logger = logging.getLogger(__name__)


class BaseMicrogridModule(yaml.YAMLObject):
    """
    Base class for all microgrid _modules.
    All values passed to step(self) that result in non-negative
    """
    module_type = None
    """
    Type of the module.

    Returns : tuple[str, {'fixed', 'flex'}]
        length-two tuple denoting the name of the module class and whether it is a fixed or flex module.

    """
    yaml_tag = None
    """
    Tag used for yaml serialization.
    """

    _energy_pos = 0

    def __init__(self,
                 raise_errors,
                 provided_energy_name='provided_energy',
                 absorbed_energy_name='absorbed_energy'
                 ):

        self.raise_errors = raise_errors
        self._current_step = 0
        self._action_space = self._get_action_spaces()
        self._observation_space = self._get_observation_spaces()
        self.provided_energy_name, self.absorbed_energy_name = provided_energy_name, absorbed_energy_name
        self._logger = ModularLogger()
        self.name = (None, None)  # set by ModularMicrogrid

    def _get_action_spaces(self):
        unnormalized_low = self.min_act if isinstance(self.min_act, np.ndarray) else np.array([self.min_act])
        unnormalized_high = self.max_act if isinstance(self.max_act, np.ndarray) else np.array([self.max_act])
        return ModuleSpace(unnormalized_low=unnormalized_low,
                           unnormalized_high=unnormalized_high)

    def _get_observation_spaces(self):
        unnormalized_low = self.min_obs if isinstance(self.min_obs, np.ndarray) else np.array([self.min_obs])
        unnormalized_high = self.max_obs if isinstance(self.max_obs, np.ndarray) else np.array([self.max_obs])
        return ModuleSpace(unnormalized_low=unnormalized_low,
                           unnormalized_high=unnormalized_high)

    def reset(self):
        """
        Reset the module to step zero and flush the log.

        Returns
        -------
        np.ndarray
            Normalized observation after resetting.

        """
        self._update_step(reset=True)
        self._logger.flush()
        return self.to_normalized(self.state, obs=True)

    def _raise_error(self, ask_value, available_value, as_source=False, as_sink=False, lower_bound=False):
        assert as_source + as_sink == 1, 'Must act as either source or sink but not both or neither.'
        name = self.__class__.__name__
        ask_v = round(ask_value, 2)
        available_v = round(available_value, 2)

        if as_source:
            if lower_bound:
                raise ValueError(f'Module {name} unable to supply requested value {ask_v} as a source. '
                             f'Must supply at least: {available_v}.')
            raise ValueError(f'Module {name} unable to supply requested value {ask_v} as a source. '
                             f'Max currently available: {available_v}.')
        else:
            raise ValueError(f'Module {name} unable to absorb requested value {ask_v} as a sink. '
                             f'Max currently capable of absorbing: {available_v}.')

    def step(self, action, normalized=True):
        """
        Take one step in the module, attempting to draw or send ``action`` amount of energy.

        Parameters
        ----------
        action : float or np.ndarray, shape (1,)
            The amount of energy to draw or send.

            If ``normalized``, the action is assumed to be normalized and is un-normalized into the range
            [:attr:`.BaseModule.min_act`, :attr:`.BaseModule.max_act`].

            If the **unnormalized** action is positive, the module acts as a source and provides energy to the
            microgrid. Otherwise, the module acts as a sink and absorbs energy.

            If the unnormalized action implies acting as a sink and ``is_sink`` is False -- or the converse -- an
            ``AssertionError`` is raised.

        normalized : bool, default True
            Whether ``action`` is normalized. If True, action is assumed to be normalized and is un-normalized into the
            range [:attr:`.BaseModule.min_act`, :attr:`.BaseModule.max_act`].

        Raises
        ------
        AssertionError
            If action implies acting as a source and module is not a source. Likewise if action implies acting as a
            sink and module is not a sink.

        Returns
        -------
        observation : np.ndarray
            State of the module after taking action ``action``.
        reward : float
            Reward/cost after taking the action.
        done : bool
            Whether the module terminates.
        info : dict
            Additional information from this step.
            Will include either``provided_energy`` or ``absorbed_energy`` as a key, denoting the amount of energy
            this module provided to or absorbed from the microgrid.

        """

        denormalized_action = self._action_space.denormalize(action) if normalized else action

        try:
            denormalized_action = denormalized_action[self._energy_pos]
        except (IndexError, TypeError):
            if not isinstance(denormalized_action, (float, int)):
                try:
                    flat_dim = np.product(denormalized_action.shape)
                    assert flat_dim == 0
                except (AttributeError, AssertionError):
                    raise ValueError(f'Bad action {denormalized_action}')
                else:
                    denormalized_action = 0.0

        state_dict = self.state_dict
        reward, done, info = self._unnormalized_step(denormalized_action)
        self._log(state_dict, reward=reward, **info)
        self._update_step()

        obs = self.to_normalized(self.state, obs=True)

        return obs, reward, done, info

    def _unnormalized_step(self, unnormalized_action):
        if unnormalized_action > 0:
            return self.as_source(unnormalized_action)
        elif unnormalized_action < 0:
            return self.as_sink(-1.0*unnormalized_action)
        else:
            if self.is_source:
                return self.as_source(unnormalized_action)
            else:
                assert self.is_sink
                return self.as_sink(-1.0 * unnormalized_action)

    def as_source(self, energy_demand):
        """
        Act as an energy source to the microgrid.

        Microgrid will attempt to provide ``energy_demand`` amount of energy.
        Examples of this include discharging a battery, importing from a grid, or using renewables.

        It is assumed that ``energy_demand>=0``.

        Parameters
        ----------
        energy_demand : float
            Amount of energy that the microgrid is requesting. Must be non-negative.

        Returns
        -------
        reward : float
            Reward/cost after attempting the satisfy the energy demand.
        done : bool
            Whether the module terminates.
        info : dict
            Additional information from this step.
            Will include``provided_energy`` as a key, denoting the amount of energy
            this module provided to the microgrid.

        Raises
        ------
        AssertionError
            If ``energy_demand<0`` or the module is not a source.

        """

        assert energy_demand >= 0
        assert self.is_source, f'step() was called with positive energy (source) for module {self} but ' \
                               f'module is not a source and ' \
                               f'can only be called with negative energy.'

        if self.module_type[-1] == 'fixed':
            return self.update(None, as_source=True)

        if energy_demand > self.max_production:
            if self.raise_errors:
                self._raise_error(energy_demand, self.max_production, as_source=True)
            provided_energy = self.max_production

        elif energy_demand < self.min_production:
            if self.raise_errors:
                self._raise_error(energy_demand, self.min_production, as_source=True, lower_bound=True)
            provided_energy = self.min_production

        else:
            provided_energy = energy_demand

        return self.update(provided_energy, as_source=True)

    def as_sink(self, energy_excess):
        """
        Act as an energy sink to the microgrid.

        Microgrid will attempt to provide ``energy_excess`` amount of energy.
        Examples of this include charging a battery, exporting from a grid, or meeting a load.

        It is assumed that ``energy_excess>=0``.

        Parameters
        ----------
        energy_excess : float
            Amount of energy that the microgrid is attempting to dissipate. Must be non-negative.

        Returns
        -------
        reward : float
            Reward/cost after attempting the absorb the energy excess.
        done : bool
            Whether the module terminates.
        info : dict
            Additional information from this step.
            Will include``absorbed_energy`` as a key, denoting the amount of energy
            this module provided to the microgrid.

        Raises
        ------
        AssertionError
            If ``energy_excess<0`` or the module is not a sink.

        """

        assert energy_excess >= 0

        if self.module_type[-1] == 'fixed':
            return self.update(None, as_sink=True)

        if energy_excess > self.max_consumption:
            if self.raise_errors:
                self._raise_error(energy_excess, self.max_consumption, as_sink=True)
            absorbed_energy = self.max_consumption
        else:
            absorbed_energy = energy_excess

        assert absorbed_energy >= 0

        return self.update(absorbed_energy, as_sink=True)

    def _log(self, state_dict_pre_step, provided_energy=None, absorbed_energy=None, **info):
        _info = info.copy()

        if self.provided_energy_name is not None:
            _info[self.provided_energy_name] = provided_energy if provided_energy is not None else 0.0
        else:
            assert provided_energy is None, 'Cannot log provided_energy with NoneType provided_energy_name.'

        if self.absorbed_energy_name is not None:
            _info[self.absorbed_energy_name] = absorbed_energy if absorbed_energy is not None else 0.0
        else:
            assert absorbed_energy is None, 'Cannot log absorbed_energy with NoneType absorbed_energy_name.'

        _info.update(state_dict_pre_step)
        self._logger.log(**_info)

    def _update_step(self, reset=False):
        if reset:
            self._current_step = 0
        self._current_step += 1

    @abstractmethod
    def update(self, external_energy_change, as_source=False, as_sink=False):
        """
        Update the state of the module given an energy request.

        Parameters
        ----------
        external_energy_change : float or None
            Amount of energy to provide or absorb.
        as_source : bool
            Whether the module is acting as a source.
        as_sink
            Whether the module is acting as a sink.

        Returns
        -------
        reward : float
            Reward/cost after attempting the absorb the energy excess.
        done : bool
            Whether the module terminates.
        info : dict
            Additional information from this step.
            Will include``absorbed_energy`` as a key, denoting the amount of energy
            this module provided to the microgrid.
        """

        pass

    def sample_action(self, strict_bound=False):
        """
        Sample an action from the module's action space.

        Parameters
        ----------
        strict_bound : bool, default False
            If True, choose action that is guaranteed to satisfy self.max_consumption and
            self.max_production bounds. Otherwise select action from min_act and min_act, which may not satisfy
            instantaneous bounds.

        Returns
        -------
        float
            An action within the action space for this module.

        """

        min_bound, max_bound = 0, 1

        if strict_bound:
            if self.is_sink:
                min_bound = self._action_space.normalize(-1 * self.max_consumption)
                if np.isnan(min_bound):
                    min_bound = 0

            if self.is_source:
                max_bound = self._action_space.normalize(self.max_production)
                if np.isnan(max_bound):
                    max_bound = 0
        return np.random.rand()*(max_bound-min_bound) + min_bound

    def to_normalized(self, value, act=False, obs=False):
        """
        Normalize an action or observation.

        Parameters
        ----------
        value : scalar or array-like
            Action or observation to normalize.
        act : bool, default False
            Set to True if you are normalizing an action.
        obs : bool, default False
            Set to True if you are normalizing an observation.

        Returns
        -------
        np.ndarray
            Normalized action.

        """
        assert act + obs == 1, 'One of act or obs must be True but not both.'
        if act:
            return self._action_space.normalize(value)
        else:
            return self._observation_space.normalize(value)

    def from_normalized(self, value, act=False, obs=False):
        """
        Un-normalize an action or observation.

        Parameters
        ----------
        value : scalar or array-like
            Action or observation to un-normalize.
        act : bool, default False
            Set to True if you are un-normalizing an action.
        obs : bool, default False
            Set to True if you are un-normalizing an observation.

        Returns
        -------
        np.ndarray
            Un-normalized action.

        """
        assert act + obs == 1, 'One of act or obs must be True but not both.'
        if act:
            return self._action_space.denormalize(value)
        if obs:
            return self._observation_space.denormalize(value)

    def log_dict(self):
        """
        Module's log as a dict.

        Returns
        -------
        dict

        """
        return self._logger.to_dict()

    def log_frame(self):
        """
        Module's log as a DataFrame.

        Returns
        -------
        log : pd.DataFrame

        """
        return self._logger.to_frame()

    @property
    def log(self):
        """
        Module's log as a DataFrame.

        Equivalent to :meth:`log_frame`.

        Returns
        -------
        log : pd.DataFrame

        """

    @property
    def logger(self):
        """
        The module's logger.

        Returns
        -------
        logger : ModularLogger

        """
        return self._logger

    @property
    def logger_last(self):
        """
        The most recent entry in the log.

        Returns
        -------
        entry : dict

        """
        return {k: v[-1] for k, v in self._logger}

    @logger.setter
    def logger(self, logger):
        assert isinstance(logger, ModularLogger)
        self._logger = logger

    @property
    @abstractmethod
    def state_dict(self):
        """
        Current state of the module as a dictionary.

        Returns
        -------
        dict

        """
        return NotImplemented

    @property
    def state(self):
        """
        Current state of the module as a vector.

        Equivalent to the values of ``state_dict``.

        Returns
        -------
        np.ndarray

        """
        return np.array([*self.state_dict.values()])

    @property
    def current_step(self):
        """
        Current step of the module.

        Returns
        -------
        int

        """
        return self._current_step

    @property
    @abstractmethod
    def min_obs(self):
        """
        Minimum observation that the module gives.

        Used in normalization and to define observation spaces.

        Returns
        -------
        float or np.ndarray
            Scalar or vector minimum observation.

        """
        raise NotImplementedError('Must define min_obs (along with the other three bounds) '
                                  'before calling super().__init__().')

    @property
    @abstractmethod
    def max_obs(self):
        """
        Maximum observation that the module gives.

        Used in normalization and to define observation spaces.

        Returns
        -------
        float or np.ndarray
            Scalar or vector maximum observation.

        """
        raise NotImplementedError('Must define max_obs (along with the other three bounds) '
                                  'before calling super().__init__().')

    @property
    @abstractmethod
    def min_act(self):
        """
        Minimum action that the module allows.

        Used in normalization and to define action spaces.

        Returns
        -------
        float or np.ndarray
            Scalar or vector minimum action.

        """
        raise NotImplementedError('Must define min_act (along with the other three bounds) '
                                  'before calling super().__init__().')

    @property
    @abstractmethod
    def max_act(self):
        """
        Maximum action that the module allows.

        Used in normalization and to define action spaces.

        Returns
        -------
        float or np.ndarray
            Scalar or vector maximum action.

        """
        raise NotImplementedError('Must define max_act (along with the other three bounds) '
                                  'before calling super().__init__().')

    @property
    def min_production(self):
        """
        Minimum amount of production at the current time step.

        In general, this value is zero. Some modules, such as ``GensetModule``,
        must produce a minimum amount of energy in some cases, and this value will be positive.

        Must be defined in any child class that is a source.
        If the module is not a source, this value is irrelevant.

        Returns
        -------
        float

        """
        return 0

    @property
    def max_production(self):
        """
        Maximum amount of production at the current time step.

        Must be defined in any child class that is a source.
        If the module is not a source, this value is irrelevant.

        Returns
        -------
        float

        """
        return NotImplemented

    @property
    def max_consumption(self):
        """
        Maximum amount of consumption at the current time step.

        Must be defined in any child class that is a sink.
        If the module is not a sink, this value is irrelevant.

        Returns
        -------
        float

        """
        return NotImplemented

    @property
    def marginal_cost(self):
        """
        Average marginal cost of producing with the module.

        Returns
        -------
        marginal_cost : float
            Average marginal cost.

        """
        return NotImplemented

    @property
    def action_space(self):
        """
        Action spaces of the module.

        Contains both normalized and un-normalized action space.

        Returns
        -------
        action_space : :class:`ModuleSpace <pymgrid.utils.space.ModuleSpace>`
            The action space.
        """
        return self._action_space

    @property
    def observation_space(self):
        """
        Observation space of the module.

        Contains both normalized and un-normalized observation spaces.

        Returns
        -------
        observation_space : :class:`ModuleSpace <pymgrid.utils.space.ModuleSpace>`
            The observation space.
        """
        return self._observation_space

    @property
    def is_source(self):
        """
        Whether the module is a source.

        Returns
        -------
        bool

        """
        return False

    @property
    def is_sink(self):
        """
        Whether the module is a sink.

        Returns
        -------
        bool

        """
        return False

    def dump(self, stream=None):
        """
        Save a module to a YAML buffer.

        Supports both strings of YAML or storing YAML in a path-like object.

        Parameters
        ----------
        stream : file-like object or None, default None
            Stream to save the YAML document. If None, returns the document instead.

        Returns
        -------
        str or None :
            Returns the YAMl document as a string if ``stream=None``. Otherwise, returns None.

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
        Load a module from yaml representation.

        Equivalent to ``yaml.safe_load(stream)``.

        Parameters
        ----------
        stream : str or file-like object
            Stream from which to read yaml representation of a module.

        Returns
        -------
        BaseMicrogridModule or child class of BaseMicrogridModule
            Deserialized module, populated with the state it possessed upon serialization.

        """
        return yaml.safe_load(stream)

    @classmethod
    def from_yaml(cls, loader, node):
        """
        Convert a yaml representation of a module to a module.

        Part of the ``load`` and equivalently the ``yaml.safe_load`` procedures.
        Should not be called directly.

        :meta private:

        Parameters
        ----------
        loader : yaml.SafeLoader
            The yaml loader.
        node : yaml.node.MappingNode
            yaml node representation of the module.

        Returns
        -------
        BaseMicrogridModule or child class of BaseMicrogridModule
            Deserialized module, populated with the state it possessed upon serialization.

        """
        add_numpy_pandas_constructors()
        mapping = loader.construct_mapping(node, deep=True)
        instance = cls.deserialize_instance(mapping["cls_params"])
        instance.logger = instance.logger.from_raw(mapping.get("log"))
        instance.name = tuple(mapping["name"])
        return instance.deserialize(mapping["state"])

    @classmethod
    def to_yaml(cls, dumper, data):
        """
        Convert a module to a yaml representation node.

        Part of the ``dump`` and equivalently the ``yaml.safe_dump`` procedures.
        Should not be called directly.

        :meta private:

        Parameters
        ----------
        dumper : yaml.SafeDumper
            The yaml dumper.
        data : BaseMicrogridModule or child class of BaseMicrogridModule
            Module to be serialized

        Returns
        -------
        yaml.node.MappingNode

        """
        add_numpy_pandas_representers()
        return dumper.represent_mapping(cls.yaml_tag, data.serialize(dumper.stream), flow_style=cls.yaml_flow_style)

    def serialize(self, dumper_stream):
        """
        Serialize module. The result is passed to a YAML dumper.

        :meta private:

        Parameters
        ----------
        dumper_stream : file-like object or None.
            The stream that the object will be dumped to.

        Returns
        -------
        dict
            The serialized module.

        """
        data = {
            "name": self.name,
            "cls_params": self._serialize_cls_params(),
            "state": self._serialize_state_attributes(),
            **self._logger.serialize("log")
        }

        return dump_data(data, dumper_stream, self.yaml_tag)

    def serializable_state_attributes(self):
        """

        Return the attributes of the module that represent the module's current state for serialization.

        :meta private:

        Returns
        -------
        list[str]
            List of attribute names.

        """
        return ["_current_step", *self.state_dict.keys()]

    def _serialize_state_attributes(self):
        return {attr_name: getattr(self, attr_name) for attr_name in self.serializable_state_attributes()}

    def _serialize_cls_params(self):
        serialized_args = {}
        cls_params = inspect.signature(self.__init__).parameters

        for p_name in cls_params.keys():
            try:
                serialized_args[p_name] = (getattr(self, p_name))
            except AttributeError:
                raise AttributeError(f"Module {self.__class__.__name__} must have attribute/property '{p_name}' corresponding to "
                                     f"class parameter of the same name.")

        return serialized_args

    @classmethod
    def deserialize_instance(cls, param_dict):
        """
        Generate an instance of this module with the arguments in param_dict.

        Part of the ``load`` and ``yaml.safe_load`` methods. Should not be called directly.

        :meta private:

        Parameters
        ----------
        param_dict : dict
            Class arguments.

        Returns
        -------
        BaseMicrogridModule or child class of BaseMicrogridModule
            The module instance.

        """
        param_dict = param_dict.copy()
        cls_params = inspect.signature(cls).parameters

        cls_kwargs = {}
        missing_params = []
        for p_name in cls_params.keys():
            try:
                cls_kwargs[p_name] = param_dict.pop(p_name)
            except KeyError:
                missing_params.append(p_name)

        if len(missing_params):
            raise KeyError(f"Missing parameter values {missing_params} for {cls}")
        return cls(**cls_kwargs)

    def deserialize(self, serialized_dict):
        """
        Populate the attributes in ``self.serializable_state_attributes``.

        Part of the ``load`` and ``yaml.safe_load`` methods. Should not be called directly.

        :meta private:

        Parameters
        ----------
        serialized_dict : dict
            Serialized state attributes

        Returns
        -------
        BaseMicrogridModule or child of BaseMicrogridModule
            The module instance.

        """
        serialized_dict = serialized_dict.copy()
        for attr_name in self.serializable_state_attributes():
            if not hasattr(self, attr_name):
                raise ValueError(f"Key {attr_name} is not an attribute of module {self} and cannot be set.")
            try:
                setattr(self, attr_name, serialized_dict.pop(attr_name))
            except KeyError:
                raise KeyError(f"Missing key {attr_name} in deserialized dict.")

        if len(serialized_dict):
            warn(f"Unused keys in serialized_dict: {list(serialized_dict.keys())}")
        return self

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented

        diff = [(k1, v1, v2) for (k1, v1), (k2, v2) in zip(self.__dict__.items(), other.__dict__.items()) if
                ((hasattr(v1, "any") and not np.allclose(v1, v2)) or (not hasattr(v1, "any") and v1 != v2))]

        return len(diff) == 0

    def __repr__(self):
        param_repr = {p: getattr(self, p) for p in inspect.signature(self.__init__).parameters}
        param_repr = [f'{p}={type(v) if hasattr(v, "__len__") and not isinstance(v, str) else v}' for p, v in param_repr.items()]
        param_repr = ', '.join(param_repr)
        return f'{self.__class__.__name__}(' \
               f'{param_repr})'
