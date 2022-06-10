from gym import Env
from gym.spaces import Box, Dict, Tuple
from src.pymgrid.microgrid.modular_microgrid.modular_microgrid import ModularMicrogrid
from src.pymgrid.microgrid.envs.base.base import BaseMicrogridEnv


class ContinuousMicrogridEnv(BaseMicrogridEnv):
    def __init__(self,
                 modules,
                 add_unbalanced_module=True,
                 loss_load_cost=10,
                 overgeneration_cost=2
                 ):
        super().__init__(modules,
                         add_unbalanced_module=add_unbalanced_module,
                         loss_load_cost=loss_load_cost,
                         overgeneration_cost=overgeneration_cost)


    def _get_action_space(self):
        return Dict({name:
                         Tuple([module.action_spaces['normalized'] for module in modules_list]) for
                     name, modules_list in self.fixed_modules.items()})


if __name__ == '__main__':
    from src.pymgrid.MicrogridGenerator import MicrogridGenerator
    from src.pymgrid.microgrid.convert.convert import to_modular, to_nonmodular

    mgen = MicrogridGenerator(nb_microgrid=3)
    mgen.generate_microgrid()
    microgrid = mgen.microgrids[2]

    modular_microgrid = to_modular(microgrid)
    env = ContinuousMicrogridEnv.from_microgrid(modular_microgrid)
    obs, reward, done, info = env.run(env.action_space.sample())
    print(modular_microgrid)