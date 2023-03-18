from pymgrid.microgrid.reward_shaping.base import BaseRewardShaper


class PVCurtailmentShaper(BaseRewardShaper):
    """
    Use in a config with 
    
    microgrid:
        attributes:
            reward_shaping_func: !PVCurtailmentShaper {}
    """

    yaml_tag = u"!PVCurtailmentShaper"

    def __call__(self, step_info, cost_info):
        pv_curtailment = self.sum_module_val(step_info, 'pv', 'curtailment')
        return -1.0 * pv_curtailment
