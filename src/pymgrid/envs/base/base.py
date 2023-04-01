import pandas as pd

from gym import Env
from gym.spaces import Box, Dict, Tuple, flatten_space, flatten
from abc import abstractmethod

from pymgrid import NonModularMicrogrid, Microgrid
from pymgrid.envs.base.skip_init import skip_init


class BaseMicrogridEnv(Microgrid, Env):
    """
    Base class for all microgrid environments.

    Implements the `OpenAI Gym API <https://www.gymlibrary.dev//>`_ for a microgrid;
    inherits from both :class:`.Microgrid` and :class:`gym.Env`.

    Parameters
    ----------
    modules : list, Microgrid, NonModularMicrogrid, or int.
        The constructor can be called in three ways:

        1. Passing a list of microgrid modules. This is identical to the :class:`.Microgrid` constructor.

        2. Passing a :class:`.Microgrid` or :class:`.NonModularMicrogrid` instance.
           This will effectively wrap the microgrid instance with the Gym API.

        3. Passing an integer in [0, 25).
           This will be result in loading the corresponding `pymgrid25` benchmark microgrids.

    add_unbalanced_module : bool, default True.
        Whether to add an unbalanced energy module to your microgrid. Such a module computes and attributes
        costs to any excess supply or demand.
        Set to True unless ``modules`` contains an :class:`.UnbalancedEnergyModule`.

    loss_load_cost : float, default 10.0
        Cost per unit of unmet demand. Ignored if ``add_unbalanced_module=False``.

    overgeneration_cost : float, default 2.0
        Cost per unit of excess generation.  Ignored if ``add_unbalanced_module=False``.

    flat_spaces : bool, default True
        Whether the environment's spaces should be flat.

        If True, all continuous spaces are :class:`gym:gym.spaces.Box`.

        Otherwise, they are nested :class:`gym:gym.spaces.Dict` of :class:`gym:gym.spaces.Tuple`
        of :class:`gym:gym.spaces.Box`, corresponding to the structure of the ``control`` arg of :meth:`.Microgrid.run`.

    trajectory_func : callable or None, default None
        Callable that sets an initial and final step for an episode. ``trajectory_func`` must take two inputs:
        :attr:`.initial_step` and :attr:`.final_step`, and return two integers: the initial and final step for
        that particular episode, respectively. This function will be called every time :meth:`.reset` is called.

        If None, :attr:`.initial_step` and :attr:`.final_step` are used to define every episode.

    """

    action_space = None
    'Space object corresponding to valid actions.'

    observation_space = None
    'Space object corresponding to valid observations.'

    def __new__(cls, modules, *args, **kwargs):
        if isinstance(modules, (NonModularMicrogrid, Microgrid)):
            import warnings
            warnings.warn('Initializing an environment with a microgrid will be deprecated in a future version.'
                          'Use from_microgrid() instead.', category=FutureWarning)

            instance = cls.from_microgrid(modules, **kwargs)

        elif isinstance(modules, int):
            import warnings
            warnings.warn('Initializing an environment with a scenario integer will be deprecated in a future version.'
                          'Use from_scenario() instead.', category=FutureWarning)
            instance = cls.from_scenario(modules, **kwargs)

        else:
            return super().__new__(cls)

        cls.__init__ = skip_init(cls, cls.__init__)
        return instance

    def __init__(self,
                 modules,
                 add_unbalanced_module=True,
                 loss_load_cost=10,
                 overgeneration_cost=2,
                 reward_shaping_func=None,
                 trajectory_func=None,
                 flat_spaces=True,
                 observation_keys=()
                 ):

        super().__init__(modules,
                         add_unbalanced_module=add_unbalanced_module,
                         loss_load_cost=loss_load_cost,
                         overgeneration_cost=overgeneration_cost,
                         reward_shaping_func=reward_shaping_func,
                         trajectory_func=trajectory_func)

        self._flat_spaces = flat_spaces
        self.observation_keys = self._validate_observation_keys(observation_keys)

        self.action_space = self._get_action_space()
        self.observation_space, self._nested_observation_space = self._get_observation_space()

    def _validate_observation_keys(self, keys):
        if not keys:
            return keys

        if isinstance(keys, str):
            keys = [keys]

        possible_keys = self.potential_observation_keys()
        bad_keys = [key for key in keys if key not in possible_keys]

        if bad_keys:
            raise NameError(f'Keys {bad_keys} not found in state.')

        return keys

    @abstractmethod
    def _get_action_space(self, remove_redundant_actions=False):
        pass

    def _get_observation_space(self):
        obs_space = {}

        state_series = self.state_series()

        for name, module_list in self.modules.iterdict():
            tup = []
            for module_num, module in enumerate(module_list):
                normalized_space = module.observation_space['normalized']

                if not self.observation_keys:
                    tup.append(normalized_space)
                else:
                    try:
                        relevant_state_idx = state_series.loc[pd.IndexSlice[name, module_num]].index
                    except KeyError:
                        continue

                    locs = [
                        relevant_state_idx.get_loc(key) for key in self.observation_keys if key in relevant_state_idx
                    ]
                    if locs:
                        box_slice = Box(
                            normalized_space.low[locs],
                            normalized_space.high[locs],
                            shape=(len(locs), ),
                            dtype=normalized_space.dtype
                        )

                        tup.append(box_slice)
            if tup:
                obs_space[name] = Tuple(tup)

        obs_space = Dict(obs_space)

        return (flatten_space(obs_space) if self._flat_spaces else obs_space), obs_space

    def potential_observation_keys(self):
        return self.state_series().index.get_level_values(-1).unique()

    def reset(self):
        obs = super().reset()
        return self._get_obs(obs)

    def step(self, action, normalized=True):
        """
        Run one timestep of the environment's dynamics.

        When the end of the episode is reached, you are responsible for calling `reset()`
        to reset the environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
        action : dict[str, list[float]]
            An action provided by the agent.

        normalized : bool, default True
            Whether the passed action is normalized or not.

        Returns
        -------
        observation : dict[str, list[float]] or np.ndarray, shape self.observation_space.shape
            Observations of each module after using the passed ``action``.
            ``observation`` is a nested dict if :attr:`~.flat_spaces` is True and a one-dimensional numpy array
            otherwise.

        reward : float
            Reward/cost of running the microgrid. A positive value implies revenue while a negative
            value is a cost.

        done : bool
            Whether the microgrid terminates.

        info : dict
            Additional information from this step.

        """

        obs, reward, done, info = self.run(action, normalized=normalized)

        obs = self._get_obs(obs)

        return obs, reward, done, info

    def _get_obs(self, obs):
        if self.observation_keys:
            obs = self.state_series(normalized=True).loc[pd.IndexSlice[:, :, self.observation_keys]]

            if self._flat_spaces:
                obs = obs.values
            else:
                obs = obs.to_frame().unstack(level=1).T.droplevel(level=1, axis=1).to_dict(orient='list')

        elif self._flat_spaces:
            obs = flatten(self._nested_observation_space, obs)

        return obs

    def render(self, mode="human"):
        """:meta private:"""
        raise NotImplementedError

    @property
    def unwrapped(self):
        """:meta private:"""
        return super().unwrapped

    @property
    def flat_spaces(self):
        """
        Whether the environment's spaces are flat.

        If True, all continuous spaces are :class:`gym:gym.spaces.Box`.

        Otherwise, they are nested :class:`gym:gym.spaces.Dict` of :class:`gym:gym.spaces.Tuple`
        of :class:`gym:gym.spaces.Box`, corresponding to the structure of the ``control`` arg of :meth:`Microgrid.run`.

        Returns
        -------
        flat_spaces : bool
            Whether the environment's spaces are flat.

        """
        return self._flat_spaces

    @classmethod
    def from_microgrid(cls, microgrid, **kwargs):
        """
        Construct an RL environment from a microgrid.

        Effectively wraps the microgrid with the environment API.

        .. warning::
            Any logs contained in the microgrid will not be ported over to the environment.

        Parameters
        ----------
        microgrid : :class:`pymgrid.Microgrid`
            Microgrid to wrap.

        Returns
        -------
        env
            The environment, suitable for reinforcement learning.

        """
        try:
            modules = microgrid.modules
        except AttributeError:
            assert isinstance(microgrid, NonModularMicrogrid)
            return cls.from_nonmodular(microgrid, **kwargs)

        kwargs = kwargs.copy()

        kwargs['add_unbalanced_module'] = kwargs.pop('add_unbalanced_module', False)
        kwargs['reward_shaping_func'] = kwargs.pop('reward_shaping_func', microgrid.reward_shaping_func)
        kwargs['trajectory_func'] = kwargs.pop('trajectory_func', microgrid.trajectory_func)

        return cls(modules.to_tuples(), **kwargs)

    @classmethod
    def from_nonmodular(cls, nonmodular, **kwargs):
        microgrid = super().from_nonmodular(nonmodular)
        return cls.from_microgrid(microgrid, **kwargs)

    @classmethod
    def from_scenario(cls, microgrid_number=0, **kwargs):
        env = super().from_scenario(microgrid_number=microgrid_number)

        if kwargs:
            return cls.from_microgrid(env, **kwargs)

        return env

    @classmethod
    def load(cls, stream):
        return cls(super().load(stream))
