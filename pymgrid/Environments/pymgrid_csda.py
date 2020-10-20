
from Environments.Environment import Environment
import numpy as np


class MicroGridEnv(Environment):
    """
    Markov Decision Process associated to the microgrid.

        Parameters
        ----------
            microgrid: microgrid, mandatory
                The controlled microgrid.
            random_seed: int, optional
                Seed to be used to generate the needed random numbers to size microgrids.

    """

    def __init__(self, microgrid, seed=0):
        super().__init__(microgrid, seed)


    # Transition function
    def transition(self):
        #         net_load = round(self.mg.load - self.mg.pv)
        #         soc = round(self.mg.battery.soc,1)
        #         s_ = (net_load, soc)  # next state
        s_ = np.array(list(self.mg.get_updated_values().values()))
        #np.array(self.mg.get_updated_values().values)#.astype(np.float)#self.mg.get_updated_values()
        #s_ = [ s_[key] for key in s_.keys()]
        return s_

    def get_action(self, action):
        return self.get_action_discret(action)
    


