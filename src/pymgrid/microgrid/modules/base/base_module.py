from abc import ABC, abstractmethod
import numpy as np
from pymgrid.microgrid.utils.logger import ModularLogger
from pymgrid.microgrid.utils.normalize import Normalize, IdentityNormalize
from gym.spaces import Box


class BaseMicrogridModule:
    """
    Base class for all microgrid _modules.
    All values passed to step(self) that result in non-negative
    """
    module_type = None

    def __init__(self,
                 raise_errors,
                 provided_energy_name='provided_energy',
                 absorbed_energy_name='absorbed_energy',
                 normalize_pos=...
                 ):
        """

        :param raise_errors:
        """

        self.raise_errors = raise_errors
        self._current_step = 0
        self._obs_normalizer = self._get_normalizer(normalize_pos, obs=True)
        self._act_normalizer = self._get_normalizer(normalize_pos, act=True)
        self._action_spaces = self._get_action_spaces()
        self._observation_spaces = self._get_observation_spaces()
        self._provided_energy_name, self._absorbed_energy_name = provided_energy_name, absorbed_energy_name
        self._logger = ModularLogger()
        self.name = (None, None)

    def _get_normalizer(self, normalize_pos, obs=False, act=False):
        assert obs+act == 1, 'Must be initiating normalizer for obs or act but not both or neither.'
        _str = 'obs' if obs else 'act'
        try:
            if obs:
                val_min, val_max = self.min_obs, self.max_obs
            else:
                val_min, val_max = self.min_act, self.max_act

        except AttributeError:
            print(f'min_{_str} and max_{_str} attributes not found for module {self.__class__.__name__}. '
                  f'Returning identity normalizer')

            return IdentityNormalize()

        try:
            normalize_position = normalize_pos['obs' if obs else 'act']
            val_min, val_max = val_min[normalize_position], val_max[normalize_position]
        except (KeyError, TypeError) as e:
            print(e)

        if val_min is None or np.isnan(val_min).any() or val_max is None or np.isnan(val_max).any():
            print(f'One of min_{_str} or max_{_str} attributes is None or NaN for module {self.__class__.__name__}. Returni'
                          f'ng identity normalizer')
            return IdentityNormalize()

        elif np.isinf(np.array(val_min)).any() or np.isinf(np.array(val_max)).any():
            print(f'One of min_{_str} or max_{_str} attributes is Infinity for module {self.__class__.__name__}. Returni'
                  f'ng identity normalizer')
            return IdentityNormalize()
        #
        # if normalize_pos is not None:
        #     val_min, val_max = val_min[normalize_pos], val_max[normalize_pos]
        try:
            assert val_max >= val_min, f'max_{_str} must be greater than min_{_str}'
        except ValueError:
            assert (val_max >= val_min).all(), f'max_{_str} must be greater than min_{_str}'

        return Normalize(val_min, val_max)

    def _get_action_spaces(self):
        shape = (1,) if isinstance(self.min_act, (float, int)) or len(self.min_act.shape) == 0 else self.min_act.shape
        return dict(normalized=Box(low=0, high=1, shape=shape),
                    unnormalized=Box(low=self.min_act if isinstance(self.min_act, np.ndarray) else np.array([self.min_act]),
                                     high=self.max_act if isinstance(self.max_act, np.ndarray) else np.array([self.max_act]),
                                     shape=shape))

    def _get_observation_spaces(self):
        shape = (1,) if isinstance(self.min_obs, (float, int)) else self.min_obs.shape
        return dict(normalized=Box(low=0, high=1, shape=shape),
                    unnormalized=Box(low=self.min_obs if isinstance(self.min_obs, np.ndarray) else np.array([self.min_obs]),
                                     high=self.max_obs if isinstance(self.max_obs, np.ndarray) else np.array([self.max_obs])
                                     ))

    def reset(self):
        self._current_step = 0
        self._logger = ModularLogger()

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
        try:
            action = action.item()
        except AttributeError:
            pass
        finally:
            if not isinstance(action, (float, int)):
                raise TypeError('action must be a float or singleton Numpy array.')

        state_dict = self.state_dict
        unnormalized_action = self._act_normalizer.from_normalized(action) if normalized else action
        step_out = self._unnormalized_step(unnormalized_action)
        obs, reward, done, info = self._conform_output(*step_out)
        obs = self._obs_normalizer.to_normalized(obs)
        self._log(state_dict, reward=reward, **info)
        self._current_step += 1

        return obs, reward, done, info

    def _conform_output(self, obs, reward, done, info):
        """
        Is this necessary? Make sure that for your modules at least it isnt
        :param obs:
        :param reward:
        :param done:
        :param info:
        :return:
        """
        def _squeeze_value(value):
            try:
                return value.squeeze()
            except AttributeError:
                return value

        _obs = np.array(obs).squeeze()
        try:
            _reward = reward.item()
        except AttributeError:
            _reward = reward

        _info = {k: _squeeze_value(v) for k, v in info.items()}
        try:
            assert _obs == obs or _obs.size == 0 and obs.size == 0 or (np.isnan(_obs).all() and np.isnan(obs).all())
        except ValueError:
            assert (_obs == obs).all() or (np.isnan(_obs).all() and np.isnan(obs).all())
        assert _reward == reward
        assert _info == info
        return _obs, _reward, done, _info

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
        # Act as source. For example, discharge battery, import from grid, use PV.
        # Will be called when action is positive.

        assert energy_demand >= 0
        assert self.is_source, f'step() was called with positive energy (source) for module {self} but module is not a source and' \
                               f'can only be called with negative energy.'

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
        # Act as sink e.g. charge battery, export to grid, use for load. Will be called when action is negative
        # (but energy_excess should be positive)
        assert energy_excess >= 0

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

        if self._provided_energy_name is not None:
            _info[self._provided_energy_name] = provided_energy if provided_energy is not None else 0.0
        else:
            assert provided_energy is None, 'Cannot log provided_energy with NoneType provided_energy_name.'

        if self._absorbed_energy_name is not None:
            _info[self._absorbed_energy_name] = absorbed_energy if absorbed_energy is not None else 0.0
        else:
            assert absorbed_energy is None, 'Cannot log absorbed_energy with NoneType absorbed_energy_name.'

        _info.update(state_dict_pre_step)
        self._logger.log(**_info)

    @abstractmethod
    def update(self, external_energy_change, as_source=False, as_sink=False):
        pass

    def sample_action(self, strict_bound=False):
        """
        Need to change this in the case of non-singleton action space
        :param strict_bound: bool. If True, choose action that is guaranteed to satisfy self.max_consumption and
            self.max_production bounds. Otherwise select action from min_act and min_act, which may not satisfy
            instantaneous bounds.
        :return:
        """

        min_bound, max_bound = 0, 1

        if strict_bound:
            if self.is_sink:
                min_bound = self._act_normalizer.to_normalized(-1 * self.max_consumption)
                if np.isnan(min_bound):
                    min_bound = 0

            if self.is_source:
                max_bound = self._act_normalizer.to_normalized(self.max_production)
                if np.isnan(max_bound):
                    max_bound = 0
        return np.random.rand()*(max_bound-min_bound) + min_bound

    def to_normalized(self, value, act=False, obs=False):
        assert act + obs == 1, 'One of act or obs must be True but not both.'
        if act:
            return self._act_normalizer.to_normalized(value)
        else:
            return self._obs_normalizer.to_normalized(value)

    def from_normalized(self, value, act=False, obs=False):
        assert act + obs == 1, 'One of act or obs must be True but not both.'
        if act:
            return self._act_normalizer.from_normalized(value)
        if obs:
            return self._obs_normalizer.from_normalized(value)

    def log_dict(self):
        return self._logger.to_dict()

    def log_frame(self):
        return self._logger.to_frame()

    @property
    def logger(self):
        return self._logger

    @property
    def logger_last(self):
        return {k: v[-1] for k, v in self._logger}

    @abstractmethod
    @property
    def state_dict(self):
        return NotImplemented

    @property
    def state(self):
        return np.array([*self.state_dict.values()])

    @property
    def current_step(self):
        return self._current_step

    @property
    def act_normalizer(self):
        return self._act_normalizer

    @property
    def obs_normalizer(self):
        return self._obs_normalizer

    @property
    def min_obs(self):
        raise NotImplementedError('Must define min_obs (along with the other three bounds) '
                                  'before calling super().__init__().')

    @property
    def max_obs(self):
        raise NotImplementedError('Must define max_obs (along with the other three bounds) '
                                  'before calling super().__init__().')

    @property
    def min_act(self):
        raise NotImplementedError('Must define min_act (along with the other three bounds) '
                                  'before calling super().__init__().')

    @property
    def max_act(self):
        raise NotImplementedError('Must define max_act (along with the other three bounds) '
                                  'before calling super().__init__().')

    @property
    def min_production(self):
        return 0

    @property
    def max_production(self):
        return NotImplemented

    @property
    def max_consumption(self):
        return NotImplemented

    @property
    def action_spaces(self):
        return self._action_spaces

    @property
    def observation_spaces(self):
        return self._observation_spaces

    @property
    def is_source(self):
        return False

    @property
    def is_sink(self):
        return False


