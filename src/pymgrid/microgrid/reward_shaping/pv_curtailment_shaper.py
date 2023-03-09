from .base import BaseRewardShaper


class PVCurtailmentShaper(BaseRewardShaper):
    yaml_tag = u"!PVCurtailmentShaper"

    def __call__(self, step_info, cost_info):
        pv_curtailment = self.sum_module_val(step_info, 'pv', 'curtailment')
        return -1.0 * pv_curtailment
