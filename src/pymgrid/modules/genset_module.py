import yaml
import numpy as np
from warnings import warn

from pymgrid.microgrid.modules.base import BaseMicrogridModule


class GensetModule(BaseMicrogridModule):
    module_type = 'genset', 'fixed'
    yaml_tag = f"!Genset"
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    def __init__(self,
                 running_min_production,
                 running_max_production,
                 genset_cost,
                 co2_per_unit=0,
                 cost_per_unit_co2=0,
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

        self._current_status, self._goal_status = init_start_up, init_start_up
        self._steps_until_up, self._steps_until_down = self._reset_up_down_times()
        self.name = ('genset', None)

        super().__init__(raise_errors,
                         provided_energy_name=provided_energy_name,
                         absorbed_energy_name=None,
                         normalize_pos=dict(obs=..., act=-1))

    def step(self, action, normalized=True):
        goal_status = action[0]
        assert 0 <= goal_status <= 1
        self.update_status(goal_status)
        return super().step(action[1:], normalized=normalized)

    def get_co2(self, production):
        return self.co2_per_unit*production

    def get_co2_cost(self, production):
        return self.cost_per_unit_co2 * self.get_co2(production)

    def _get_fuel_cost(self, production):
        if callable(self.genset_cost):
            return self.genset_cost(production)
        return self.genset_cost*production

    def get_cost(self, production):
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

        If steps_until_up/steps_until_down is zero:
            Change status. Then continue with the below.

        Things stay the same when _goal_status equals self._current_status equals self._goal_status.
            (I.e. goal status is same as current status and same as goal status. In this case, one of steps_until_up/steps_until_down
            should be zero (the current status one) and the other should be self.start_up_time/self.wind_down_time, respectively).
        If _goal_status equals self._current_status but does not equal self._goal_status:
            We are trying to abort a status change.
            If self.allow_abortion:
                Change self._goal_status, and reset steps_until_up/steps_until_down (one to zero, one to max).
            Otherwise:
            Increment steps_until_up/steps_until_down (whichever is being undergone).

        If _goal_status equals self._goal_status but not self._current_status.
            We are continuing a previously requested status change. Increment steps_until_up/steps_until_down (whichever is being undergone).

        :param goal_status:
        :return:
        """
        assert self._steps_until_down >= 0 and self._steps_until_up >= 0

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
            raise ValueError('This is to check is self.next_status works. If raised, it doesn\'t.')

    def _finish_in_progress_change(self):
        if self._steps_until_up == 0 and self._goal_status == 1:
            self._current_status = True
            self._reset_up_down_times()
            return True
        elif self._steps_until_down == 0 and self._goal_status == 0:
            self._current_status = False
            self._reset_up_down_times()
            return True
        return False

    def _instant_up(self):
        self._goal_status = 1

        if not self._current_status:
            self._current_status = True
            self._reset_up_down_times()

    def _instant_down(self):
        self._goal_status = 0

        if self._current_status:
            self._current_status = False
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
            incrementer should be positive"""
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
        return self.next_status(goal_status) * self.running_max_production

    def next_min_production(self, goal_status):
        return self.next_status(goal_status) * self.running_min_production

    def serializable_state_attributes(self):
        return ["_current_step"] + [f"_{key}" for key in self.state_dict.keys()]

    @property
    def state_dict(self):
        return dict(zip(('current_status', 'goal_status', 'steps_until_up', 'steps_until_down'), self.current_obs))

    @property
    def current_obs(self):
        return np.array([self._current_status, self._goal_status, self._steps_until_up, self._steps_until_down])

    @property
    def current_status(self):
        return self._current_status

    @property
    def goal_status(self):
        return self._goal_status

    @property
    def max_production(self):
        # Note: these values know if the genset is currently running, but they don't know if you're planning on turning
        #    it off at this step. Only applies if start_up_time or wind_down_time is 0.
        return self._current_status * self.running_max_production

    @property
    def min_production(self):
        # Note: these values know if the genset is currently running, but they don't know if you're planning on turning
        #    it off at this step. Only applies if start_up_time or wind_down_time is 0.
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
    def is_source(self):
        return True
