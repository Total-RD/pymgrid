import numpy as np

from pymgrid.microgrid.trajectory.base import BaseTrajectory


class StochasticTrajectory(BaseTrajectory):
    yaml_tag = u"!StochasticTrajectory"

    def __call__(self, initial_step, final_step):

        initial = np.random.randint(initial_step, final_step-2)
        final = np.random.randint(initial, final_step)

        return initial, final


class FixedLengthStochasticTrajectory(BaseTrajectory):
    yaml_tag = u"!FixedLengthStochasticTrajectory"

    def __init__(self, trajectory_length):
        self.trajectory_length = trajectory_length

    def __call__(self, initial_step, final_step):
        if final_step - initial_step < self.trajectory_length:
            raise ValueError(f'Cannot create a trajectory of length {self.trajectory_length}'
                             f'between initial_step ({initial_step}) and final_step ({final_step})')

        initial = np.random.randint(initial_step, final_step-self.trajectory_length)

        return initial, initial + self.trajectory_length
