from . import BatteryTransitionModel


class BiasedTransitionModel(BatteryTransitionModel):
    yaml_tag = u"!BiasedTransitionModel"

    def __init__(self, true_efficiency):
        self.true_efficiency = true_efficiency

    def transition(self, external_energy_change, efficiency, **kwargs):
        return super().transition(external_energy_change=external_energy_change,
                                  efficiency=self.true_efficiency)
