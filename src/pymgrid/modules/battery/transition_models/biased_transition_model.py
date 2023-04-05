from pymgrid.modules.battery.transition_models import BatteryTransitionModel


class BiasedTransitionModel(BatteryTransitionModel):
    yaml_tag = u"!BiasedTransitionModel"

    def __init__(self, true_efficiency=None, relative_efficiency=None):
        if true_efficiency is None and relative_efficiency is None:
            raise ValueError("Must pass one of 'true_efficiency' and 'relative_efficiency'.")

        self.true_efficiency = true_efficiency
        self.relative_efficiency = relative_efficiency
        self.efficiency = None

    def _set_efficiency(self, efficiency):
        if self.efficiency is not None:
            return

        if self.true_efficiency is None:
            self.efficiency = self.relative_efficiency * efficiency
        else:
            self.efficiency = self.true_efficiency

    def transition(self, external_energy_change, efficiency, **kwargs):
        self._set_efficiency(efficiency)
        return super().transition(external_energy_change=external_energy_change,
                                  efficiency=self.efficiency)
