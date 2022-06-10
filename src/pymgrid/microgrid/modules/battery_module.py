from src.pymgrid.microgrid.modules.base_module import BaseMicrogridModule
import numpy as np


class BatteryModule(BaseMicrogridModule):
    module_type = ('battery', 'fixed')

    def __init__(self,
                 min_capacity,
                 max_capacity,
                 max_charge,
                 max_discharge,
                 efficiency,
                 battery_cost_cycle=0.0,
                 init_charge=None,
                 init_soc=None,
                 raise_errors=False):
        assert 0 < efficiency <= 1
        self.max_capacity = max_capacity        # Maximum energy that can be contained in the battery. Equiv. to soc=1
        self.min_capacity = min_capacity        # Minimum energy that must be contained in the battery
        self.max_charge = max_charge            # Maximum charge in one step
        self.max_discharge = max_discharge      # Maximum discharge in one step.
        self.efficiency = efficiency
        self.battery_cost_cycle = battery_cost_cycle

        self.min_soc, self.max_soc = min_capacity/max_capacity, 1
        self._current_charge, self._soc = self._init_battery(init_charge, init_soc)
        self.name = ('battery', None)
        super().__init__(raise_errors,
                         provided_energy_name='discharge_amount',
                         absorbed_energy_name='charge_amount')

    def _init_battery(self, init_capacity, init_soc):
        if init_capacity is not None and init_soc is not None:
            print('Passed both init_capacity and init_soc. Using init_capacity and ignoring init_soc')
        elif init_soc is not None:
            init_capacity = init_soc*self.max_capacity
        else:
            init_capacity = (self.max_capacity+self.min_capacity)/2
            init_soc = init_capacity/self.max_capacity

        return init_capacity, init_soc

    def update(self, external_energy_change, as_source=False, as_sink=False):
        assert as_source + as_sink == 1, 'Must act as either source or sink but not both or neither.'

        if as_source:
            info_key = 'provided_energy'
            internal_energy_change = -1.0 * external_energy_change / self.efficiency
            assert internal_energy_change <= 0
        else:
            info_key = 'absorbed_energy'
            internal_energy_change = external_energy_change*self.efficiency
            assert internal_energy_change >= 0

        self._update_state(internal_energy_change)
        reward = -1.0 * self.get_cost(internal_energy_change)
        info = {info_key: external_energy_change}
        return self.current_obs, reward, False, info

    def _update_state(self, energy_change):
        self._current_charge += energy_change
        if self._current_charge < self.min_capacity:
            assert np.isclose(self._current_charge, self.min_capacity)
            self._current_charge = self.min_capacity
        self._soc = self._current_charge/self.max_capacity

    def get_cost(self, energy_change):
        return np.abs(energy_change)*self.battery_cost_cycle

    def state_dict(self):
        return dict(zip(('soc', 'current_charge'), self.current_obs))

    @property
    def max_production(self):
        # Max discharge
        return min(self.max_discharge, self._current_charge-self.min_capacity) * self.efficiency

    @property
    def max_consumption(self):
        # Max charge
        return min(self.max_charge, self.max_capacity - self._current_charge) / self.efficiency

    @property
    def current_obs(self):
        return np.array([self.soc, self.current_charge])

    @property
    def current_charge(self):
        return self._current_charge

    @property
    def soc(self):
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
        return -self.max_discharge

    @property
    def max_act(self):
        return self.max_charge

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