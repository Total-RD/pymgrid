from pymgrid.modules.battery.transition_models import BatteryTransitionModel


class DecayTransitionModel(BatteryTransitionModel):
    yaml_tag = u"!DecayTransitionModel"

    def __init__(self, decay_rate=0.999**(1/24)):
        """

        Parameters
        ----------
        decay_rate : float, default 0.99**(1/24)
            Amount to decay in one time step; should be in (0, 1]. If 1, no decay and equivalent to parent model.
            Default is equivalent to 1/10 of a percent decay in one day.

        """
        self.decay_rate = decay_rate
        self.initial_step = None

    def _current_efficiency(self, efficiency, current_step):
        return efficiency * (self.decay_rate ** (current_step-self.initial_step))

    def _update_step(self, current_step):
        if self.initial_step is None or current_step <= self.initial_step:
            self.initial_step = current_step

    def transition(self, external_energy_change, efficiency, current_step, **kwargs):
        self._update_step(current_step)
        current_efficiency = self._current_efficiency(efficiency, current_step)

        return super().transition(external_energy_change, efficiency=current_efficiency)
