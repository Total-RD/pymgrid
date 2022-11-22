import numpy as np
import yaml

from pymgrid.microgrid.modules.base import BaseMicrogridModule


class UnbalancedEnergyModule(BaseMicrogridModule):
    module_type = ('balancing', 'flex')
    yaml_tag = u"!UnbalancedEnergyModule"
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    def __init__(self,
                 raise_errors,
                 loss_load_cost=10,
                 overgeneration_cost=2.0
                 ):

        super().__init__(raise_errors,
                         provided_energy_name='loss_load',
                         absorbed_energy_name='overgeneration')
        self.loss_load_cost, self.overgeneration_cost = loss_load_cost, overgeneration_cost
        self.name = ('unbalanced_energy', None)

    def update(self, external_energy_change, as_source=False, as_sink=False):
        assert as_source + as_sink == 1, 'Must act as either source or sink but not both or neither.'

        info_key = 'provided_energy' if as_source else 'absorbed_energy'
        reward = -1.0 * self.get_cost(external_energy_change, as_source, as_sink)
        assert reward <= 0
        info = {info_key: external_energy_change}

        return reward, False, info

    def get_cost(self, energy_amount, as_source, as_sink):
        if as_source: # loss-load
            return self.loss_load_cost*energy_amount
        elif as_sink:
            return self.overgeneration_cost*energy_amount
        else:
            raise RuntimeError

    @property
    def state_dict(self):
        return dict()

    @property
    def state(self):
        return np.array([])

    @property
    def min_obs(self):
        # Min charge amount, min soc
        return np.array([])

    @property
    def max_obs(self):
        return np.array([])

    @property
    def min_act(self):
        return -np.inf

    @property
    def max_act(self):
        return np.inf

    @property
    def max_production(self):
        return np.inf

    @property
    def max_consumption(self):
        return np.inf

    @property
    def is_source(self):
        return True

    @property
    def is_sink(self):
        return True


