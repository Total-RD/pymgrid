from gym import Env
from gym.spaces import Dict, Tuple, flatten_space, flatten
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

    """

    action_space = None
    'Space object corresponding to valid actions.'

    observation_space = None
    'Space object corresponding to valid observations.'

    def __new__(cls, modules, *args, **kwargs):
        if isinstance(modules, (NonModularMicrogrid, Microgrid)):
            instance = cls.from_microgrid(modules)

        elif isinstance(modules, int):
            instance = cls.from_scenario(modules)
        else:
            return super().__new__(cls)

        cls.__init__ = skip_init(cls, cls.__init__)
        return instance

    def __init__(self,
                 modules,
                 add_unbalanced_module=True,
                 loss_load_cost=10,
                 overgeneration_cost=2,
                 flat_spaces=True
                 ):

        super().__init__(modules,
                         add_unbalanced_module=add_unbalanced_module,
                         loss_load_cost=loss_load_cost,
                         overgeneration_cost=overgeneration_cost)

        self._flat_spaces = flat_spaces
        self.action_space = self._get_action_space()
        self.observation_space, self._nested_observation_space = self._get_observation_space()

    @abstractmethod
    def _get_action_space(self):
        pass

    def _get_observation_space(self):
        obs_space = Dict({name:
                         Tuple([module.observation_space['normalized'] for module in modules_list]) for
                     name, modules_list in self.modules.iterdict()})

        return (flatten_space(obs_space) if self._flat_spaces else obs_space), obs_space

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
        if self._flat_spaces:
            obs = flatten(self._nested_observation_space, obs)
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        return flatten(self._nested_observation_space, obs) if self._flat_spaces else obs

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
    def from_microgrid(cls, microgrid):
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
            return cls.from_nonmodular(microgrid)
        else:
            return cls(modules.to_tuples(), add_unbalanced_module=False)

    @classmethod
    def from_nonmodular(cls, nonmodular):
        microgrid = super().from_nonmodular(nonmodular)
        return cls.from_microgrid(microgrid)

    @classmethod
    def load(cls, stream):
        return cls(super().load(stream))
