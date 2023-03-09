from pymgrid.microgrid.trajectory.base import BaseTrajectory


class DeterministicTrajectory(BaseTrajectory):
    yaml_tag = u"!DeterministicTrajectory"

    def __init__(self, initial_step, final_step):
        self.initial_step = initial_step
        self.final_step = final_step

    def __call__(self, initial_step, final_step):
        return self.initial_step, self.final_step
