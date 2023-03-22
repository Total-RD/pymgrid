import yaml

from gym.spaces import Discrete
from warnings import warn

from pymgrid.algos.priority_list import PriorityListAlgo
from pymgrid.envs.base import BaseMicrogridEnv


class DiscreteMicrogridEnv(BaseMicrogridEnv, PriorityListAlgo):
    """
        A discrete env that implements priority lists as actions on a microgrid.

        The environment deploys fixed controllable modules to the extent necessary to zero out the net load (load minus
        renewable generation).
    """

    actions_list: list
    """
    List of priority lists.
    
    Each element in this list corresponds to an action in the environment's action space, and defines an order 
    in which to deploy fixed controllable modules. Specifically, each action corresponds to a unique priority list, 
    itself containing :class:`PriorityListElements<.PriorityListElement>` that represents a particular module's position
    in the deployment order.
    
    Returns
    -------
    actions_list : list of list of :class:`.PriorityListElement`
        List of all priority lists.
        
    """

    yaml_tag = u"!DiscreteMicrogridEnv"
    yaml_loader = yaml.SafeLoader
    yaml_dumper = yaml.SafeDumper

    def __init__(self,
                 modules,
                 add_unbalanced_module=True,
                 loss_load_cost=10,
                 overgeneration_cost=2,
                 reward_shaping_func=None,
                 trajectory_func=None,
                 flat_spaces=True,
                 observation_keys=None,
                 remove_redundant_gensets=True
                 ):
        super().__init__(modules,
                         add_unbalanced_module=add_unbalanced_module,
                         loss_load_cost=loss_load_cost,
                         overgeneration_cost=overgeneration_cost,
                         reward_shaping_func=reward_shaping_func,
                         trajectory_func=trajectory_func,
                         flat_spaces=flat_spaces,
                         observation_keys=observation_keys)

        self.action_space, self.actions_list = self._get_action_space(remove_redundant_gensets)

    def _get_action_space(self, remove_redundant_gensets=False):
        """
        An action here is a priority list - in what order to deploy controllable source modules.
        Compute the total expected load
        It exhausts the entire production of the 0th source module before moving on to the second and the third, etc.

        If there are n fixed source modules, then there are 2^{n-1} actions.
        :return:
        """

        priority_lists = self.get_priority_lists(remove_redundant_gensets)

        n_actions = len(priority_lists)

        if n_actions > 1000:
            warn(f'Microgrid with {len(priority_lists[0])} fixed source modules defines large action space with '
                 f'{n_actions} elements.')

        space = Discrete(n_actions)

        return space, priority_lists

    def _get_action(self, action_num):
        if action_num not in self.action_space:
            raise ValueError(f" Action {action_num} not in action space {self.action_space}")

        priority_list = list(self.actions_list[action_num])

        return self._populate_action(priority_list)

    def remove_action(self, action_number):
        """
        Remove an action from the action space.

        Useful if two actions happen to be redundant in a particular use case.

        Parameters
        ----------
        action_number : int
            Index of the action to remove.

        """

        if action_number not in self.action_space:
            raise ValueError('Cannot remove action that is not in the action space!')

        self.actions_list.pop(action_number)
        self.action_space = Discrete(self.action_space.n - 1)

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.

        When the end of the episode is reached, you are responsible for calling `reset()`
        to reset the environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
        action : int
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
        self._microgrid_logger.log(action=action)
        microgrid_action = self._get_action(action)
        return super().step(microgrid_action, normalized=False)

    def sample_action(self, strict_bound=False, sample_flex_modules=False):
        return self.action_space.sample()

    def __repr__(self):
        return f"DiscreteMicrogridEnv({super().__repr__()}"

    def __str__(self):
        return self.__repr__()
