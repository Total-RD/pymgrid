import time

import numpy as np
import pandas as pd
from pymgrid.algos.Control import ControlOutput
from pymgrid.utils.DataGenerator import SampleGenerator
from pymgrid.algos import ModelPredictiveControl


class SampleAverageApproximation(SampleGenerator):
    """
    A class to run a Sample Average Approximation version of Stochastic MPC.

    Parameters

        microgrid: pymgrid.Microgrid.Microgrid
            the underlying microgrid
        control_duration: int
            number of iterations to learn over

    Attributes:

        microgrid: pymgrid.Microgrid.Microgrid
            the underlying microgrid
        control_duration: int
            number of iterations to learn over
        mpc: algos.Control.ModelPredictiveControl
            An instance of MPC class to run MPC over for each sample
        NPV: utils.DataGenerator.NoisyPVData
            An instance of NoisyPVData to produce pv forecast and samples
        NL: utils.DataGenerator.NoisyLoadData
            An instance of NoisyLoadData to produce initial load forecast
        NG: utils.DataGenerator.NoisyGridData or None
            An instance of NoisyGridData to produce initial grid forecast. None if there is no grid
        forecasts: pd.DataFrame, shape (8760,3)
            load, pv, grid forecasts. See create_forecasts for details.
        samples: list of pd.DataFrame of shape (8760,3), or None
            list of samples created from sampling from distributions defined in forecasts.
                See sample_from_forecasts for details. None if sample_from_forecasts hasn't been called
    """
    def __init__(self, microgrid, control_duration=8760, **forecast_args):
        if control_duration > 8760:
            raise ValueError('control_duration must be less than 8760')

        # if not isinstance(microgrid, Microgrid.Microgrid):
        #     raise TypeError('microgrid must be of type \'pymgrid.Microgrid.Microgrid\', is {}'.format(type(microgrid)))

        super().__init__(microgrid, **forecast_args)
        self.control_duration = control_duration
        self.mpc = ModelPredictiveControl(self.microgrid)

    def run(self, n_samples=10, forecast_steps=None, optimal_percentile=0.5, use_previous_samples=True, verbose=False, **kwargs):
        """
        Runs MPC over a number of samples for to average out for SAA
        :param n_samples: int, default 25
            number of samples to run
        :param forecast_steps: int or None, default None
            number of steps to use in forecast. If None, uses 8760-self.horizon
        :param use_previous_samples: bool, default True
            whether to use previous previous stored in self.samples if they are available
        :param verbose: bool, default False
            verbosity
        :return:
            outputs, list of ControlOutput
                list of ControlOutputs for each sample. See ControlOutput or run_mpc_on_sample for details.
        """
        if self.samples is None or not use_previous_samples:
            self.samples = self.sample_from_forecasts(n_samples=n_samples, **kwargs)

        outputs = []

        t0 = time.time()

        output = self.run_mpc_on_group(self.samples, forecast_steps=forecast_steps,
                                        optimal_percentile=optimal_percentile, verbose=verbose)

        if verbose:
            print('Running time: {}'.format(round(time.time()-t0)))

        return output

    def determine_optimal_actions(self, outputs=None, percentile=0.5, verbose=False):
        """
        Given a list of samples from run(), determines which one has cost at the percentile in percentile.

        :param outputs: list of ControlOutput
            list of ControlOutputs from run()
        :param percentile: float, default 0.5
            which percentile to return as optimal.
        :return:
            optimal_output, ControlOutput
                output at optimal percentile
        """
        if percentile < 0. or percentile > 1.:
            raise ValueError('percentile must be in [0,1]')

        partition_val = int(np.floor(len(outputs)*percentile))
        partition = np.partition(outputs, partition_val)

        if verbose:
            sorted_outputs = np.sort(outputs)
            selected_output = partition[partition_val]
            print()
            for j, output in enumerate(sorted_outputs):
                print('Output {}, cost: {}, battery charge {}, discharge {}:'.format(
                    j, round(output.cost,2) , round(output.first_dict['battery_charge'],2), round(output.first_dict['battery_discharge'],2)))
                if output is selected_output:
                    print('Selected output {} with percentile {}'.format(j, percentile))

        return partition[partition_val]

    def run_mpc_on_group(self, samples, forecast_steps=None, optimal_percentile=0.5, verbose=False):
        columns_needed = ('pv', 'load', 'grid')

        output = ControlOutput(alg_name='saa', empty=True, microgrid=self.microgrid)

        T = min([len(sample) for sample in samples])
        if forecast_steps is None:
            forecast_steps = T-self.microgrid.horizon
        elif forecast_steps>T-self.microgrid.horizon:
            raise ValueError('forecast steps must be less than length of samples minus horizon')

        for j in range(forecast_steps):
            if verbose:
                print('iter {}'.format(j))

            horizon_outputs = []

            for sample in samples:
                if not isinstance(sample, pd.DataFrame):
                    raise TypeError('samples must be pd.DataFrame')
                if not all([needed in sample.columns.values for needed in columns_needed]):
                    raise KeyError('samples must contain columns {}, currently contains {}'.format(
                        columns_needed, sample.columns.values))

                sample.iloc[j] = self.underlying_data.iloc[j]  # overwrite with actual data

                horizon_output = self.mpc.mpc_single_step(sample, output, j)

                horizon_outputs.append(horizon_output)

            # return horizon_outputs

            optimal_output = self.determine_optimal_actions(outputs=horizon_outputs, percentile=optimal_percentile)
            output.append(optimal_output, actual_load=self.underlying_data.loc[j,'load'],
                          actual_pv=self.underlying_data.loc[j,'pv'],
                          actual_grid=self.underlying_data.loc[j,'grid'])

        return output

    def run_deterministic_on_forecast(self, forecast_steps=None, verbose=False):

        sample = self.forecasts.copy()
        output = ControlOutput(alg_name='mpc', empty=True, microgrid=self.microgrid)

        T = len(sample)

        if forecast_steps is None:
            forecast_steps = T - self.microgrid.horizon
        elif forecast_steps > T - self.microgrid.horizon:
            raise ValueError('forecast steps must be less than length of samples minus horizon')

        for j in range(forecast_steps):
            if verbose:
                print('iter {}'.format(j))

                sample.iloc[j] = self.underlying_data.iloc[j]  # overwrite with actual data

                horizon_output = self.mpc.mpc_single_step(sample, output, j)

                output.append(horizon_output)

            return output