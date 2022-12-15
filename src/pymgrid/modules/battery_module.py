from pymgrid.modules.base import BaseMicrogridModule
import numpy as np
import yaml
from warnings import warn


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

    max_discharge : float
        Maximum amount the battery can be discharged in one step.

    efficiency : float
        Efficiency of the battery.
        See :meth:`BatteryModule.model_transition` for details.

    battery_cost_cycle : float, default 0.0
        Marginal cost of charging and discharging.

    battery_transition_model : callable or None, default None
        Function to model the battery's transition.
        If None, :meth:`BatteryModule.default_transition_model` is used.

        .. note::
            If you define a battery_transition_model, it must be YAML-serializable if you plan to serialize
            your battery module or any microgrid containing your battery.

            For example, you can define it as a class with a ``__call__`` method and ``yaml.YAMLObject`` as its metaclass.
            See the `PyYAML documentation <https://pyyaml.org/wiki/PyYAMLDocumentation>`_ for details.

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
                 raise_errors=False):
        assert 0 < efficiency <= 1
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.max_charge = max_charge
        self.max_discharge = max_discharge
        self.efficiency = efficiency
        self.battery_transition_model = battery_transition_model
        self.battery_cost_cycle = battery_cost_cycle

        self.min_soc, self.max_soc = min_capacity/max_capacity, 1
        self.init_charge, self.init_soc = init_charge, init_soc
        self._current_charge, self._soc = self._init_battery(init_charge, init_soc)
        self.name = ('battery', None)
        super().__init__(raise_errors,
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
            assert internal_energy_change <= 0
        else:
            info_key = 'absorbed_energy'
            internal_energy_change = self.model_transition(external_energy_change)
            assert internal_energy_change >= 0

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
                max_production,
                max_consumption,
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
        if self.battery_transition_model is None:
            return self.default_transition_model(external_energy_change=energy, **self.transition_kwargs())
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

        max_production : float
            Maximum amount of production, which is the lower of the maximum discharge and the discharge that would
            send the battery to ``min_capacity``.

        max_consumption : float
            Maximum amount of consumption, which is the lower of the maximum charge and the charge that would send
            the battery to ``max_capacity``.

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
                    max_production=self.max_production,
                    max_consumption=self.max_consumption,
                    state_dict=self.state_dict
                    )

    @staticmethod
    def default_transition_model(external_energy_change, efficiency, **transition_kwargs):
        """
        A simple battery transition model.

        In this model, the amount of energy retained is given by ``efficiency``.

        For example, if a microgrid requests 100 kWh of energy and ``efficiency=0.5``, the battery must use
        200 kWh of energy. Alternatively, if a microgrid sends a battery 100 kWh of energy and ``efficiency=0.5``,
        the battery's charge will increase by 50 kWh.

        Parameters
        ----------
        external_energy_change : float
            Amount of energy that is being requested externally.
            If ``energy > 0``, it is energy that is absorbed by the battery -- a charge.
            If ``energy < 0``, it is energy provided by the battery: a discharge.

        efficiency : float
            Battery efficiency.

        transition_kwargs : dict
            State transition values given by :meth:`BatteryModule.transition_kwargs`.

        Returns
        -------
        internal_energy : float
            Amount of energy that the battery must use or will retain given the external amount of energy.

        """

        if external_energy_change < 0:
            return external_energy_change / efficiency
        else:
            return external_energy_change * efficiency

    @property
    def state_dict(self):
        return dict(zip(('soc', 'current_charge'), [self._soc, self._current_charge]))

    @property
    def max_production(self):
        # Max discharge
        return min(self.max_discharge, self._current_charge-self.min_capacity) * self.efficiency

    @property
    def max_consumption(self):
        # Max charge
        return min(self.max_charge, self.max_capacity - self._current_charge) / self.efficiency

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
        return -self.max_discharge / self.efficiency

    @property
    def max_act(self):
        return self.max_charge * self.efficiency

    @property
    def marginal_cost(self):
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
