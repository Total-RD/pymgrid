import yaml
import numpy as np
from warnings import warn

from pymgrid.modules.base import BaseMicrogridModule


class GensetModule(BaseMicrogridModule):
    """
    A genset/generator module.

    This module is a controllable source module; when used as a module in a microgrid, you must pass it an energy production
    request.

    Parameters
    ----------
    running_min_production : float
        Minimum production of the genset when it is running.

    running_max_production : float
        Maximum production of the genset when it is running.

    genset_cost : float or callable
       * If float, the marginal cost of running the genset: ``total_cost = genset_cost * production``.

       * If callable, a function that takes the genset production as an argument and returns the genset cost.

    co2_per_unit : float, default 0.0
        Carbon dioxide production per unit energy production.

    cost_per_unit_co2 : float, default 0.0
        Carbon dioxide cost per unit carbon dioxide production.

    start_up_time : int, default 0
        Number of steps it takes to turn on the genset.

    wind_down_time : int, default 0
        Number of steps it takes to turn off the genset.

    allow_abortion : bool, default True
        Whether the genset is able to remain shut down while in the process of starting up and vice versa.

    init_start_up : bool, default True
        Whether the genset is running upon reset.

    raise_errors : bool, default False
        Whether to raise errors if bounds are exceeded in an action.
        If False, actions are clipped to the limit possible.

    provided_energy_name : str, default "genset_production"
        Name of the energy provided by this module, to be used in logging.

    """
    module_type = 'genset', 'controllable'
    yaml_tag = f"!Genset"
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    _energy_pos = 1

    def __init__(self,
                 running_min_production,
                 running_max_production,
                 genset_cost,
                 co2_per_unit=0.0,
                 cost_per_unit_co2=0.0,
                 start_up_time=0,
                 wind_down_time=0,
                 allow_abortion=True,
                 init_start_up=True,
                 raise_errors=False,
                 provided_energy_name='genset_production'):

        if running_min_production > running_max_production:
            raise ValueError('parameter min_production must not be greater than parameter max_production.')

        if not allow_abortion:
            warn('Gensets that do not allow abortions are not fully tested, setting allow_abortion=False '
                 'may lead to unexpected behavior.')

        self.running_min_production, self.running_max_production = running_min_production, running_max_production
        self.co2_per_unit, self.cost_per_unit_co2 = co2_per_unit, cost_per_unit_co2

        self.genset_cost = genset_cost
        self.start_up_time = start_up_time
        self.wind_down_time = wind_down_time
        self.allow_abortion = allow_abortion
        self.init_start_up = init_start_up

        self._current_status, self._goal_status = int(init_start_up), int(init_start_up)
        self._steps_until_up, self._steps_until_down = self._reset_up_down_times()
        self.name = ('genset', None)

        super().__init__(raise_errors, provided_energy_name=provided_energy_name, absorbed_energy_name=None)

    def step(self, action, normalized=True):
        """
        Take one step in the module, attempting to draw a certain amount of energy from the genset.

        Parameters
        ----------
        action : float or np.ndarray, shape (2,)
            Two-dimensional vector containing two values. The first value is used to passed to
            :meth:`.GensetModule.update_status` while the second is the amount of energy to draw from the genset.

            If ``normalized``, the amount of energy is assumed to be normalized and is un-normalized into the range
            [:attr:`.GensetModule.min_act`, :attr:`.GensetModule.max_act`].

            If the **unnormalized** action is positive, the module acts as a source and provides energy to the
            microgrid. Otherwise, the module acts as a sink and absorbs energy.

            If the unnormalized action implies acting as a sink and ``is_sink`` is False -- or the converse -- an
            ``AssertionError`` is raised.

            .. warning::
               The first element in ``action`` is not denormalized before being passed to
               :meth:`.GensetModule.update_status`, regardless of the value of ``normalized``.

        normalized : bool, default True
            Whether ``action`` is normalized. If True, action is assumed to be normalized and is un-normalized into the
            range [:attr:`.GensetModule.min_act`, :attr:`.GensetModule.max_act`].

        Raises
        ------
        AssertionError
            If action implies acting as a sink, or ``action[0]`` in outside of ``[0, 1]``.

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
            Will include either `provided_energy` or `absorbed_energy` as a key, denoting the amount of energy
            this module provided to or absorbed from the microgrid.

        """
        goal_status = action[0]
        assert 0 <= goal_status <= 1
        self.update_status(goal_status)
        return super().step(action, normalized=normalized)

    def get_co2(self, production):
        """
        Carbon dioxide emissions of energy production.

        Parameters
        ----------
        production : float
            Energy production.
        Returns
        -------
        co2 : float
            Carbon dioxide production.

        """
        return self.co2_per_unit*production

    def get_co2_cost(self, production):
        """
        Carbon dioxide production cost.

        Parameters
        ----------
        production : float
            Energy production.
        Returns
        -------
        co2_cost : float
            Carbon dioxide cost.

        """
        return self.cost_per_unit_co2 * self.get_co2(production)

    def _get_fuel_cost(self, production):
        if callable(self.genset_cost):
            return self.genset_cost(production)
        return self.genset_cost*production

    def get_cost(self, production):
        """
        Total cost of energy production.

        Includes both fuel and carbon dioxide costs.

        Parameters
        ----------
        production : float
            Energy production.

        Returns
        -------
        cost : float
            Total cost.

        """
        return self._get_fuel_cost(production) + self.get_co2_cost(production)

    def update(self, external_energy_change, as_source=False, as_sink=False):
        assert as_source, 'This module may only act as a source.'

        reward = -1.0 * self.get_cost(external_energy_change)
        info = {'provided_energy': external_energy_change,
                'co2_production': self.get_co2(external_energy_change)}

        return reward, False, info

    def _reset_up_down_times(self):
        if self._goal_status != self._current_status:
            raise RuntimeError('Attempting to reset up and down times with status change in progress.')

        if self._current_status:
            self._steps_until_up = 0
            self._steps_until_down = self.wind_down_time
        else:
            self._steps_until_down = 0
            self._steps_until_up = self.start_up_time

        return self._steps_until_up, self._steps_until_down

    def _update_up_down_times(self):
        if self._goal_status == 0: # Turning it off
            self._steps_until_down -= 1
        else:
            self._steps_until_up -= 1

    def update_status(self, goal_status):
        """
        Update the status of the microgrid.

        The status and goal status are updated, taking into account any in-progress status change as well as
        ``goal_status``. This method updates the internal properties ``self.current_status``, ``self.goal_status``,
        ``self.steps_until_up``, and ``self.steps_until_down`` as follows:

        1. If ``steps_until_up == 0`` or ``steps_until_down == 0``, the status is changed to on and off, respectively.
           The following steps are then executed.
        
        2. If ``goal_status == self.current_status == self.goal_status``, the genset is in equilibrium and its status
           does not change.

           In this case, one of ``self.steps_until_up``/``self.steps_until_down`` should be zero -- the former
           if ``self.current_status`` and the latter if not -- and the other should be
           ``self.start_up_time``/``self.wind_down_time``, respectively.

        3. If ``goal_status == self.current_status != self.goal_status``, we are trying to abort a status change.

           * If ``self.allow_abortion``, the abortion can succeed. ``self.goal_status`` changes to ``goal_status``
             and ``steps_until_up``/``steps_until_down`` are reset (one to zero, one to
             the corresponding ``self.start_up_time``/``self.wind_down_time``).

           * Otherwise, we proceed with an in-progress status change, and the corresponding
             ``steps_until_up``/``steps_until_down`` is incremented. This is identical to the case below.

        4. If ``goal_status == self.goal_status != self.current_status``, a previously requested status change is
           being continued, and the corresponding ``steps_until_up``/``steps_until_down`` is incremented.

        .. note::
            Steps 2, 3, and 4 are mutually exclusive, while step 1 is not and will be executed before the relevant
            step 2, 3 or 4.

        Parameters
        ----------
        goal_status : float in [0, 1].
            Goal status as defined by an external action.

            Will be rounded to 0 or 1 to define the goal status.
        """
        assert self._steps_until_down >= 0 and self._steps_until_up >= 0

        if not 0 <= goal_status <= 1:
            raise ValueError(f"Invalid goal_status value {goal_status}, must be in [0, 1].")

        goal_status = round(goal_status)
        next_prediction = self.next_status(goal_status)

        if goal_status == self._current_status == self._goal_status:
            # Everything is hunky-dory
            assert self._steps_until_down == 0 or self._steps_until_up == 0
            return

        instant_up = self.start_up_time == 0 and goal_status == 1
        instant_down = self.wind_down_time == 0 and goal_status == 0
        if goal_status != self._goal_status and (self.allow_abortion or instant_up or instant_down):
            self._goal_status = goal_status

        finished_change = self._finish_in_progress_change()

        if not finished_change:
            self._non_instantaneous_update(goal_status)

        if not self._current_status == next_prediction:
            raise ValueError('self.next_status working incorrectly.')

    def _finish_in_progress_change(self):
        if self._steps_until_up == 0 and self._goal_status == 1:
            self._current_status = 1
            self._reset_up_down_times()
            return True
        elif self._steps_until_down == 0 and self._goal_status == 0:
            self._current_status = 0
            self._reset_up_down_times()
            return True
        return False

    def _instant_up(self):
        self._goal_status = 1

        if not self._current_status:
            self._current_status = 1
            self._reset_up_down_times()

    def _instant_down(self):
        self._goal_status = 0

        if self._current_status:
            self._current_status = 0
            self._reset_up_down_times()

    def _non_instantaneous_update(self, goal_status):
        if (goal_status == self._current_status != self._goal_status) and self.allow_abortion:
            # First case: aborting an in-progress status change
            self._goal_status = goal_status
            self._reset_up_down_times()
        elif self._current_status == self._goal_status != goal_status:
            # Second case: new status change request
            self._reset_up_down_times()
            self._goal_status = goal_status

        if self._goal_status != self._current_status:
            """
            Current status is not equal to status goal; thus a status change is in progress and the relevant
            incrementer should be positive.
            """
            if self._goal_status:
                assert self._steps_until_up > 0
            else:
                assert self._steps_until_down > 0
            self._update_up_down_times()

    def sample_action(self, strict_bound=False, **kwargs):
        return np.array([np.random.rand(), super().sample_action(strict_bound=strict_bound)])

    def _raise_error(self, ask_value, available_value, as_source=False, as_sink=False, lower_bound=False):
        try:
            super()._raise_error(ask_value, available_value, as_source=as_source, as_sink=as_sink, lower_bound=lower_bound)
        except ValueError as e:
            if not self._current_status:
                raise ValueError(f'{e}\n This may be because this genset module is not currently running.') from e
            else:
                raise ValueError(f'{e}\n This is despite the fact this genset module is currently running.') from e

    def next_status(self, goal_status):
        """
        Predict the next status of the genset given a goal status.

        Does not modify the genset in any way.

        Parameters
        ----------
        goal_status : {0, 1}
            Goal status.

        Returns
        -------
        next_status : {0, 1}
            The next status given the current status and the goal status.

        """
        if goal_status:
            if self._current_status:
                return 1
            elif self._steps_until_up == 0:
                return 1
            else:
                return 0
        else:
            if not self._current_status:
                return 0
            elif self._steps_until_down == 0:
                return 0
            else:
                return 1

    def next_max_production(self, goal_status):
        """
        Maximum production given a goal status.

        Parameters
        ----------
        goal_status : {0, 1}
            A goal status.

        Returns
        -------
        next_max_production : float
            Maximum production given a goal status.

        """
        return self.next_status(goal_status) * self.running_max_production

    def next_min_production(self, goal_status):
        """
        Minimum production given a goal status.

        Parameters
        ----------
        goal_status : {0, 1}
            A goal status.

        Returns
        -------
        next_min_production : float
            Minimum production given a goal status.

        """
        return self.next_status(goal_status) * self.running_min_production

    def serializable_state_attributes(self):
        return ["_current_step"] + [f"_{key}" for key in self.state_dict.keys()]

    @property
    def state_dict(self):
        return {'current_status': self._current_status,
                'goal_status': self._goal_status,
                'steps_until_up': self._steps_until_up,
                'steps_until_down': self._steps_until_down}

    @property
    def current_status(self):
        """
        Status of the genset.

        On or off.

        Returns
        -------
        status : {0, 1}
            Integer value denoting the genset's current status.

        """
        return self._current_status

    @property
    def goal_status(self):
        """
        Goal of the genset.

        Whether the genset is trying to turn -- or keep -- itself on or off.

        Returns
        -------
        status : {0, 1}
            Integer value denoting the genset's goal status.

        """
        return self._goal_status

    @property
    def max_production(self):
        """
        Maximum amount of production at the current time step.

        .. warning::
            This value is aware of the genset's current status, but does not know if you're planning on turning
            it off at this step.

            This consideration is only relevant if ``start_up_time==0`` or ``wind_down_time==0``.

        Returns
        -------
        max_production : float
            Current maximum production.

        """
        return self._current_status * self.running_max_production

    @property
    def min_production(self):
        """
        Minimum amount of production at the current time step.

        .. warning::
            This value is aware of the genset's current status, but does not know if you're planning on turning
            it off at this step.

            This consideration is only relevant if ``start_up_time==0`` or ``wind_down_time==0``.

        Returns
        -------
        min_production : float
            Current minimum production.

        """
        return self._current_status * self.running_min_production

    @property
    def min_obs(self):
        return np.array([0, 0, 0, 0])

    @property
    def max_obs(self):
        return np.array([1, 1, self.start_up_time, self.wind_down_time])

    @property
    def min_act(self):
        return np.array([0, 0])

    @property
    def max_act(self):
        return np.array([1, self.running_max_production])

    @property
    def marginal_cost(self):
        return self.get_cost(1.0)

    @property
    def is_source(self):
        return True
