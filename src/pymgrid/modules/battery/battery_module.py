import numpy as np
import yaml

from warnings import warn

from pymgrid.modules.base import BaseMicrogridModule
from pymgrid.modules.battery.transition_models import BatteryTransitionModel


class BatteryModule(BaseMicrogridModule):
    """
    A battery module.

    Battery modules are fixed: when calling ``Microgrid.run``, you must pass a control for batteries.

    Parameters
    ----------
    min_capacity : float
        Minimum energy that must be contained in the battery.

    max_capacity : float
        Maximum energy that can be contained in the battery.
        If ``soc=1``, capacity is at this maximum.

    max_charge : float
        Maximum amount the battery can be charged in one step.

        .. warning::
            This amount is the maximum the battery can be charged internally, dependent on the
            ``battery_transition_model``. The amount the battery can be charged externally (e.g. the amount of
            energy the battery can absorb) is defined as the negative of :attr:`.min_act`.

    max_discharge : float
        Maximum amount the battery can be discharged in one step.

        .. warning::
            This amount is the maximum the battery can be discharged internally, dependent on the
            ``battery_transition_model``. The amount the battery can be discharged externally (e.g. the amount of
            energy the battery can provide) is defined as :attr:`.max_act`.

    efficiency : float
        Efficiency of the battery.
        See :meth:`BatteryModule.model_transition` for details.

    battery_cost_cycle : float, default 0.0
        Marginal cost of charging and discharging.

    battery_transition_model : callable or None, default None
        Function to model the battery's transition.
        If None, :class:`.BatteryTransitionModel` is used.

        .. note::
            If you define a battery_transition_model, it must be YAML-serializable if you plan to serialize
            your battery module or any microgrid containing your battery.

            For example, you can define it as a class with a ``__call__`` method and ``yaml.YAMLObject`` as its metaclass.
            See the `PyYAML documentation <https://pyyaml.org/wiki/PyYAMLDocumentation>`_ for details and
            :class:`.BatteryTransitionModel` for an example.

    init_charge : float or None, default None
        Initial charge of the battery.
        One of ``init_charge`` or ``init_soc`` must be passed, else an exception is raised.
        If both are passed, ``init_soc`` is ignored and ``init_charge`` is used.

    init_soc : float or None, default None
        Initial state of charge of the battery.
        One of ``init_charge`` or ``init_soc`` must be passed, else an exception is raised.
        If both are passed, ``init_soc`` is ignored and ``init_charge`` is used.

    raise_errors : bool, default False
        Whether to raise errors if bounds are exceeded in an action.
        If False, actions are clipped to the limit possible.

    """
    module_type = ('battery', 'controllable')
    yaml_tag = f"!BatteryModule"
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    def __init__(self,
                 min_capacity,
                 max_capacity,
                 max_charge,
                 max_discharge,
                 efficiency,
                 battery_cost_cycle=0.0,
                 battery_transition_model=None,
                 init_charge=None,
                 init_soc=None,
                 initial_step=0,
                 raise_errors=False):
        assert 0 < efficiency <= 1
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.max_charge = max_charge
        self.max_discharge = max_discharge
        self.efficiency = efficiency
        self.battery_cost_cycle = battery_cost_cycle
        self.battery_transition_model = battery_transition_model
        self.min_soc, self.max_soc = min_capacity/max_capacity, 1
        self.init_charge, self.init_soc = init_charge, init_soc
        self._current_charge, self._soc = self._init_battery(init_charge, init_soc)
        self._min_act, self._max_act = self._set_min_max_act()
        self.name = ('battery', None)

        super().__init__(raise_errors,
                         initial_step=initial_step,
                         provided_energy_name='discharge_amount',
                         absorbed_energy_name='charge_amount')

    def _init_battery(self, init_charge, init_soc):
        if init_charge is not None:
            if init_soc is not None:
                warn('Passed both init_capacity and init_soc. Using init_charge and ignoring init_soc')
            init_soc = init_charge / self.max_capacity
        elif init_soc is not None:
            init_charge = init_soc * self.max_capacity
        else:
            raise ValueError("Must set one of init_charge and init_soc.")

        return init_charge, init_soc

    def update(self, external_energy_change, as_source=False, as_sink=False):
        assert as_source + as_sink == 1, 'Must act as either source or sink but not both or neither.'

        if as_source:
            info_key = 'provided_energy'
            internal_energy_change = self.model_transition(-1.0 * external_energy_change)
            assert internal_energy_change <= 0 and (-1 * internal_energy_change <= self.max_discharge or
                                                    np.isclose(-1 * internal_energy_change, self.max_discharge))
        else:
            info_key = 'absorbed_energy'
            internal_energy_change = self.model_transition(external_energy_change)
            assert internal_energy_change >= 0 and (internal_energy_change <= self.max_charge or
                                                    np.isclose(internal_energy_change, self.max_charge))

        self._update_state(internal_energy_change)
        reward = -1.0 * self.get_cost(internal_energy_change)
        info = {info_key: external_energy_change}
        return reward, False, info

    def _update_state(self, energy_change):
        self._current_charge += energy_change
        if self._current_charge < self.min_capacity:
            assert np.isclose(self._current_charge, self.min_capacity)
            self._current_charge = self.min_capacity
        self._soc = self._current_charge/self.max_capacity

    def get_cost(self, energy_change):
        """
        Get the cost of charging or discharging.

        Parameters
        ----------
        energy_change : float
            Internal energy change.

        Returns
        -------
        cost : float
            Cost of charging or discharging.

        """
        return np.abs(energy_change)*self.battery_cost_cycle

    def model_transition(self, energy):
        """
        Convert an external energy request to a change in internal energy.

        This function uses the class argument ``battery_transition_model`` if one was passed.

        ``battery_transition_model`` must use the following api:

        .. code-block:: bash

            internal_energy_change = battery_transition_model(
                external_energy_change,
                min_capacity,
                max_capacity,
                max_charge,
                max_discharge,
                efficiency,
                battery_cost_cycle,
                current_step,
                state_dict
            )

        The return value ``internal_energy_change``  must be a float.
        See :meth:`transition_kwargs` and :meth:`battery_transition_model` for details on these parameters;
        all parameters are passed as keyword arguments.

        Parameters
        ----------
        energy : float
            External energy change.

        Returns
        -------
        internal_energy : float
            Amount of energy that the battery must use or will retain given the external amount of energy.

        """
        return self.battery_transition_model(external_energy_change=energy, **self.transition_kwargs())

    def transition_kwargs(self):
        """
        Values passed to transition models.

        Keys
        ----
        min_capacity : float
            Minimum energy that must be contained in the battery.

        max_capacity : float
            Maximum energy that can be contained in the battery.
            If ``soc=1``, capacity is at this maximum.

        max_charge : float
            Maximum amount the battery can be charged in one step.

        max_discharge : float
            Maximum amount the battery can be discharged in one step.

        efficiency : float
            Efficiency of the battery.

        battery_cost_cycle : float
            Marginal cost of charging and discharging.

        current_step : int
            Current step.

        state_dict : dict
            State dictionary, with state of charge and current capacity information.

        Returns
        -------
        kwargs : dict
            Transition keyword arguments.

        """
        return dict(min_capacity=self.min_capacity,
                    max_capacity=self.max_capacity,
                    max_charge=self.max_charge,
                    max_discharge=self.max_discharge,
                    efficiency=self.efficiency,
                    battery_cost_cycle=self.battery_cost_cycle,
                    current_step=getattr(self, '_current_step', 0),
                    state_dict=self.state_dict()
                    )

    def _set_min_max_act(self):
        min_act = self.model_transition(-1 * self.max_charge)
        max_act = self.model_transition(self.max_discharge)

        return min_act, max_act

    def _state_dict(self):
        return dict(zip(('soc', 'current_charge'), [self._soc, self._current_charge]))

    @property
    def max_production(self):
        # Max discharge
        return self.model_transition(min(self.max_discharge, self._current_charge-self.min_capacity))

    @property
    def max_consumption(self):
        # Max charge
        return -1 * self.model_transition(-1 * min(self.max_charge, self.max_capacity - self._current_charge))

    @property
    def current_charge(self):
        """
        Battery charge.

        Level of charge of the battery.

        Returns
        -------
        current_charge : float
            Charge.

        """
        return self._current_charge

    @property
    def soc(self):
        """
        Battery state of charge.

        Level of charge of the battery relative to its capacity.

        Returns
        -------
        soc : float
            State of charge. In the range [0, 1].

        """
        return self._soc

    @property
    def min_obs(self):
        # Min charge amount, min soc
        return np.array([self.min_soc, self.min_capacity])

    @property
    def max_obs(self):
        return np.array([self.max_soc, self.max_capacity])

    @property
    def min_act(self):
        return self._min_act

    @property
    def max_act(self):
        return self._max_act

    @property
    def max_external_charge(self):
        """
        Maximum amount of energy the battery can absorb when charging.

        This is distinct from :attr:`.max_charge`, which is the maximum difference in battery capacity when charging.
        If the battery is perfectly efficient, these are equivalent.

        Returns
        -------
        max_external_charge : float
            Maximum amount of energy the battery can aborb when charging.

        """
        return -1 * self.min_act

    @property
    def max_external_discharge(self):
        """
        Maximum amount of energy the battery can provide when discharging.

        This is distinct from :attr:`.max_discharge`, which is the maximum difference in battery capacity when
        discharging. If the battery is perfectly efficient, these are equivalent.

        Returns
        -------
        max_external_discharge : float
            Maximum amount of energy the battery can provide when discharging.

        """
        return self.max_act

    @property
    def production_marginal_cost(self):
        return self.battery_cost_cycle

    @property
    def absorption_marginal_cost(self):
        return self.battery_cost_cycle

    @property
    def is_source(self):
        return True

    @property
    def is_sink(self):
        return True

    @soc.setter
    def soc(self, value):
        self._current_charge, self._soc = self._init_battery(None, value)

    @current_charge.setter
    def current_charge(self, value):
        self._current_charge, self._soc = self._init_battery(value, None)

    @property
    def battery_transition_model(self):
        return self._battery_transition_model

    @battery_transition_model.setter
    def battery_transition_model(self, value):
        if value is None:
            self._battery_transition_model = BatteryTransitionModel()
        else:
            self._battery_transition_model = value
