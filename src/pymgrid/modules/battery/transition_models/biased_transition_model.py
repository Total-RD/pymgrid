from . import BatteryTransitionModel


class BiasedTransitionModel(BatteryTransitionModel):
    yaml_tag = u"!BiasedTransitionModel"

    def __init__(self, true_efficiency):
        self.true_efficiency = true_efficiency

    def transition(self, external_energy_change, efficiency):
        return super().transition(external_energy_change, self.true_efficiency)
