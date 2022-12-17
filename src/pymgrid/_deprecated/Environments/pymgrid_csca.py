from abc import ABC
import gym, logging, numpy as np, pandas as pd
from gym import Env
from pymgrid._deprecated.non_modular_microgrid import NonModularMicrogrid
from pymgrid.MicrogridGenerator import MicrogridGenerator
from copy import deepcopy
from pymgrid.algos.saa.saa import SampleAverageApproximation
from pymgrid.algos import ModelPredictiveControl

logger = logging.getLogger(__name__)
LOG = False
DEBUG = True

# If you get a JSON serializable error, turn this on:
JSON_ERROR = False


def sample_reset(has_grid, saa, microgrid, sampling_args=None):
    """
    Generates a new sample using an instance of SampleAverageApproximation and
    :param has_grid: bool, whether the microgrid has a grid.
    :param saa:, SampleAverageApproximation
    :param microgrid: Microgrid
    :param sampling_args: arguments to be passed to saa.sample_from_forecasts().
    :return:
    """
    if sampling_args is None:
        sampling_args = dict()

    sample = saa.sample_from_forecasts(n_samples=1, **sampling_args)
    sample = sample[0]

    microgrid._load_ts = pd.DataFrame(sample['load'])
    microgrid._pv_ts = pd.DataFrame(sample['pv'])
    microgrid._df_record_state['load'] = [sample['load'].iloc[0].squeeze()]
    microgrid._df_record_state['pv'] = [sample['pv'].iloc[0].squeeze()]
    if has_grid:
        microgrid._grid_status_ts = pd.DataFrame(sample['grid'])
        microgrid._df_record_state['grid_status'] = [sample['grid'].iloc[0].squeeze()]


def generate_sampler(microgrid, forecast_args):
    """
    Generates an instance of SampleAverageApproximate to use in future sampling.
    :param microgrid:
    :param forecast_args:
    :return:
    """
    if forecast_args is None:
        forecast_args = dict()

    return SampleAverageApproximation(microgrid, **forecast_args)


class MicrogridEnv(Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, microgrid, trajectory_len=None, max_episode_len=None):
        """
        :param max_episode_len:
        :param microgrid: Microgrid, the underlying microgrid.
        :param trajectory_len: int, length of a trajectory (to be started at a random index).
            Default None, runs an entire year
        """
        super().__init__()

        if isinstance(microgrid, int) and 0<=microgrid<=25:
            print('Initializing microgrid {} of 25 using 25 microgrids from MicrogridGenerator'.format(microgrid))
            m_gen = MicrogridGenerator(nb_microgrid=25)
            m_gen.generate_microgrid(verbose=False)
            self.microgrid = deepcopy(m_gen.microgrids[microgrid])

        elif isinstance(microgrid, NonModularMicrogrid):
            self.microgrid = deepcopy(microgrid)

        else:
            raise ValueError('microgrid must be of type Microgrid, is {}'.format(type(microgrid)))

        assert self.microgrid._data_length == 8760, 'Microgrid data length should be 8760, is {}'.format(self.microgrid._data_length)

        self.has_grid = self.microgrid.architecture['grid'] == 1
        self.has_genset = self.microgrid.architecture['genset'] == 1

        observation_dim = len(self.microgrid._df_record_state)
        self.observation_space = gym.spaces.Box(low=0, high=np.float('inf'), shape=(observation_dim,), dtype=np.float64)
        self.action_space = None

        self.current_action = None
        self.current_obs = None


        if max_episode_len is None:
            self.microgrid.horizon = 0
        else:
            self.microgrid.horizon = self.microgrid._data_length - max_episode_len

        self.trajectory_len = trajectory_len
        self._short_trajectory_set()


    def _short_trajectory_set(self):
        trajectory_len = self.trajectory_len
        if trajectory_len is not None:
            assert isinstance(trajectory_len, int)
            from numpy.random import randint
            high_range = self.microgrid._data_length - self.microgrid.horizon - trajectory_len

            start_index = randint(low=0, high=high_range)
            self.microgrid._tracking_timestep = start_index
            self.microgrid._data_length = start_index + trajectory_len + self.microgrid.horizon

    def reset(self):
        self.microgrid.reset()
        self._short_trajectory_set()

        initial_state = self.microgrid.get_updated_values()
        observations = np.array(list(initial_state.values()))

        self.current_obs = observations

        return observations

    def step(self, action, **kwargs):
        """
        :param **kwargs:
        :param
            action:
        :return:
            observation, np.ndarray
                If self.has_grid, shape (10,), with values
                        [load, hour, pv, battery_soc, capa_to_charge, capa_to_discharge,
                            grid_status, grid_co2, grid_price_import, grid_price_export]
                Else, shape (6,), with values
                        [load, hour, pv, battery_soc, capa_to_charge, capa_to_discharge]
            reward, float
                The reward (negative cost) of the step
            done, bool
                Whether the episode is complete
            info, dict
                Info

        """
        control_dict = self.get_control_dict(action)
        observation = self.run_control(control_dict)
        reward = -1.0 * self.microgrid.get_cost()
        done = self.microgrid.done
        info = dict()

        self.current_obs = observation
        self.current_action = action

        return observation, reward, done, info

    def get_control_dict(self, action):
        """
        A function that takes an action (discrete or continuous) and returns a control_dict.
            See ContinuousMicrogridEnv for an example.
        """
        return NotImplemented

    def run_control(self, control_dict):
        """
        Given a control_dict, calls microgrid.run(_) and returns the observations.
        :param control_dict:
        :return: observations, np.ndarray
        """
        updated_vals = self.microgrid.run(control_dict)
        observations = np.array(list(updated_vals.values()))

        assert len(observations) == self.observation_space.shape[0]
        return observations


class ContinuousMicrogridEnv(MicrogridEnv):
    """
    Class to run a Microgrid in the format of a gym env. Continuous states, continuous actions.
    """
    def __init__(self, microgrid, standardization=True, trajectory_len=None, max_episode_len=None, **kwargs):
        """
        :param microgrid: Microgrid, the underlying microgrid.
        :param standardization: bool, default True. whether to scale the actions to a factor determined by a run of MPC.
        """
        super().__init__(microgrid, trajectory_len=trajectory_len, max_episode_len=max_episode_len)

        self.logger = kwargs['logger'] if 'logger' in kwargs else None

        action_dim = 5+self.has_genset
        upper_bound, lower_bound = self._get_action_ub_lb()

        self.action_space = gym.spaces.Box(low=lower_bound, high=upper_bound, shape=(action_dim,), dtype=np.float64)

        self.standardization = standardization
        if not JSON_ERROR and self.standardization:
            self.standardizations = self.pre_compute_standardizations()

            # Rescale the action space
            low_new = self.standardize(self.action_space.low, use_proxy='action')
            self.action_space.low = low_new
            high_new = self.standardize(self.action_space.high, use_proxy='action')
            high_new[1] = 0.1 # This is a hard-coded grid_export bound
            self.action_space.high = high_new
        else:
            self.standardizations = None

    def _get_action_ub_lb(self):
        # Upper Bound
        p_max_import = self.microgrid.parameters['grid_power_import'].values[0]
        p_max_export = self.microgrid.parameters['grid_power_export'].values[0]
        p_max_charge = self.microgrid.parameters['battery_power_charge'].values[0]
        p_max_discharge = self.microgrid.parameters['battery_power_discharge'].values[0]
        pv_max = self.microgrid.parameters.PV_rated_power.squeeze()

        upper_bound = [p_max_import, p_max_export, p_max_charge, p_max_discharge, pv_max]

        if self.has_genset:
            p_genset_max = self.microgrid.parameters['genset_rated_power'].values[0] * self.microgrid.parameters['genset_pmax'].values[0]
            upper_bound.insert(0,p_genset_max)

        upper_bound = np.array(upper_bound)

        # Lower Bound

        lower_bound = [0]*5

        if self.has_genset:
            action_dim = 6
            p_genset_min = self.microgrid.parameters['genset_rated_power'].values[0] * \
                           self.microgrid.parameters['genset_pmin'].values[0]
            lower_bound.insert(0, p_genset_min)

        lower_bound = np.array(lower_bound)

        return upper_bound, lower_bound



    def get_values(self, *value_names):
        # TODO Speed this up. Refactor standardizations. nan_to_num takes significant time.
        """
        Helper function. Given a list of value names (e.g., the names of components of the state/actions),
            returns their values in the same order.
        Note: if the env standardizes values, this function returns the unstandarized values.
        :param value_names:
        :return:
        """
        genset_actions = ['genset', 'grid_import', 'grid_export', 'battery_charge', 'battery_discharge', 'pv_consummed']
        no_genset_actions = ['grid_import', 'grid_export', 'battery_charge', 'battery_discharge', 'pv_consummed']

        grid_observations = ['load', 'hour', 'pv', 'battery_soc', 'capa_to_charge', 'capa_to_discharge',
                            'grid_status', 'grid_co2', 'grid_price_import', 'grid_price_export']
        no_grid_observations = ['load', 'hour', 'pv', 'battery_soc', 'capa_to_charge', 'capa_to_discharge']

        if self.current_action is None:
            print('Warning: current_action is None, should only happen on first iteration')
            if self.has_genset:
                self.current_action = np.array([0]*len(genset_actions))
            else:
                self.current_action = np.array([0]*len(no_genset_actions))

            action = self.current_action
            obs = self.current_obs

        elif self.standardization:
            obs_mean, obs_std, action_mean, action_std = self.standardizations
            action = self.standardize(self.current_action, action_mean, action_std, direction='backward')
            obs = self.standardize(self.current_obs, obs_mean, obs_std, direction='backward')
        else:
            action = self.current_action
            obs = self.current_obs

        if self.has_genset:
            actions_dict = dict(zip(genset_actions, action))
        else:
            actions_dict = dict(zip(no_genset_actions, action))
        if self.has_grid:
            obs_dict = dict(zip(grid_observations, obs))
        else:
            obs_dict = dict(zip(no_grid_observations, obs))

        values = []


        for name in value_names:
            if name in actions_dict.keys():
                values.append(actions_dict[name])
            elif name in obs_dict.keys():
                values.append(obs_dict[name])
            else:
                raise ValueError('Value \'{}\' not recognized with current architecture'.format(name))

        return values

    def reset(self):
        observation = super().reset()
        if self.standardization:
            obs_mean, obs_std, action_mean, action_std = self.standardizations
            observation = self.standardize(observation, obs_mean, obs_std, direction='forward')
            self.current_obs = observation

        return observation



    def step(self, action, **kwargs):
        """
        :param **kwargs:
        :param
            action: np.ndarray
                If self.has_genset: shape (6,), with values
                    [genset, grid_import, grid_export, battery_charge, battery_discharge, pv_consummed]
                Else: shape (5,), with values
                    [grid_import, grid_export, battery_charge, battery_discharge, pv_consummed]
        :return:
            observation, np.ndarray
                If self.has_grid, shape (10,), with values
                        [load, hour, pv, battery_soc, capa_to_charge, capa_to_discharge,
                            grid_status, grid_co2, grid_price_import, grid_price_export]
                Else, shape (6,), with values
                        [load, hour, pv, battery_soc, capa_to_charge, capa_to_discharge]
            reward, float
                The reward (negative cost) of the step
            done, bool
                Whether the episode is complete
            info, dict
                Info

        """
        # Actions must be passed in order as defined in pymgrid25 paper
        assert isinstance(action, np.ndarray)

        unscaled_action = action.copy()

        if self.standardization:
            if not isinstance(action, np.ndarray):
                raise TypeError('action must be of type np.ndarray')

            obs_mean, obs_std, action_mean, action_std = self.standardizations

            action = self.standardize(action, action_mean, action_std, direction='backward')

        observation, reward, done, info = super().step(action)

        if self.standardization:
            observation = self.standardize(observation, obs_mean, obs_std, direction='forward')

        self.current_obs = observation
        self.current_action = unscaled_action

        # Do you want to deal w everything in normalized space or unnormalized space

        return observation, reward, done, info


    def standardize(self, data, mean_proxy=None, std_proxy=None, direction='forward', use_proxy=None):
        """
        :param data: np.ndarray, shape (observation_dim,) or (action_dim), observation or action to rescale
        :param mean_proxy: np.ndarray, shape (observation_dim,) or (action_dim), mean to use in rescaling
        :param std_proxy: np.ndarray, shape (observation_dim,) or (action_dim), standard deviation to use in rescaling
        :param direction: str, default 'forward'. One of 'forward' or 'backward', whether to scale to or from standard normal
        :return: np.ndarray, rescaled values.
        """

        if (mean_proxy is None and std_proxy is None and use_proxy is None) or (mean_proxy is not None and use_proxy is not None):
            raise ValueError('Must pass mean_proxy and std_proxy, or use_proxy must be a str in (\'action\', \'obs\'), but not both')
        if mean_proxy is None and std_proxy is None:
            if use_proxy == 'action':
                mean_proxy, std_proxy = [x for x in self.standardizations[2:]]
            elif use_proxy == 'obs':
                mean_proxy, std_proxy = [x for x in self.standardizations[:2]]
            else:
                raise NameError('Unable to recognize use_proxy {}, must be one of \'action\' or \'obs\''.format(use_proxy))


        names = ('data', 'mean_proxy', 'std_proxy')
        vals = (data, mean_proxy, std_proxy)
        dirs = ('forward', 'backward')

        for name, v in zip(names, vals):
            if not isinstance(v, np.ndarray):
                raise TypeError('{} must be of type numpy.ndarray, is {}'.format(name, type(v)))
        if not (data.shape == mean_proxy.shape and mean_proxy.shape == std_proxy.shape):
            raise ValueError('Incompatible shapes of data, mean_proxy, std_proxy. Must be equal, are: {}'.format(
                dict(zip(names, [v.shape for v in vals]))))

        if direction not in dirs:
            raise ValueError('direction must be one of {}'.format(dirs))

        if direction == 'forward':
            return (data-mean_proxy)/std_proxy
        else:
            return data*std_proxy+mean_proxy

    def pre_compute_standardizations(self,alg_to_use='mpc'):
        """
        Runs a control algorithm to pre compute the standardizations for actions/observations to rescale to standard normal (ish).
        :param alg_to_use: str, default 'mpc'. What algorithm to run to compute the standardizations
        :return: tuple len(4,): obs_mean, obs_std, action_mean, action_std
        """

        action_mean = [0]*self.action_space.shape[0]
        action_std = [0]*self.action_space.shape[0]
        obs_mean = [0]*self.observation_space.shape[0]
        obs_std = [0]*self.observation_space.shape[0]

        if alg_to_use == 'mpc':
            old_horizon = self.microgrid.horizon
            self.microgrid.horizon = 24
            MPC = ModelPredictiveControl(self.microgrid)
            mpc_output = MPC.run(max_steps=1000)
            self.microgrid.horizon = old_horizon
            if self.has_genset:

                action_keys = 'genset', 'grid_import, grid_export, battery_charge, battery_discharge, pv_consummed'

                for j, name in enumerate(action_keys):
                    action_mean[j] = np.mean(mpc_output['action'][name])
                    action_std[j] = np.std(mpc_output['action'][name])

                obs_keys = list(self.microgrid._df_record_state.keys())

                for j, name in enumerate(obs_keys):
                    obs_mean[j] = np.mean(mpc_output['status'][name])
                    obs_std[j] = np.std(mpc_output['status'][name])

            else:
                action_keys = 'grid_import', 'grid_export', 'battery_charge', 'battery_discharge', 'pv_consummed'

                for j, name in enumerate(action_keys):
                    action_mean[j] = np.mean(mpc_output['action'][name])
                    action_std[j] = np.std(mpc_output['action'][name])

                obs_keys = list(self.microgrid._df_record_state.keys())

                for j, name in enumerate(obs_keys):
                    obs_mean[j] = np.mean(mpc_output['status'][name])
                    obs_std[j] = np.std(mpc_output['status'][name])
        else:
            raise RuntimeError('algorithm name {} not currently supported'.format(alg_to_use))

        for j in range(len(obs_std)):
            if obs_std[j] < 1.0:
                obs_std[j] = 1.0
        for j in range(len(action_std)):
            if action_std[j] < 1.0:
                action_std[j] = 1.0

        names = ('obs_mean', 'obs_std', 'action_mean', 'action_std')
        outputs = obs_mean, obs_std, action_mean, action_std
        outputs = tuple(np.array(output) for output in outputs)

        for name, output in zip(names,outputs):
            if (output == 0).sum() != 0:
                for j,val in enumerate(output):
                    if val == 0:
                        print('Warning: Zero value in pos {} in {}, may not have been filled properly'.format(j,name))

        return outputs

    def get_control_dict(self, action):
        """
        Given an np.ndarray of actions, parses into a control_dict.
        :param action: np.ndarray, shape (action_dim,)
        :return: dict, control_dict
        """

        if not isinstance(action, np.ndarray):
            raise TypeError('action must be an ndarray, is {}'.format(type(action)))

        if self.has_genset:

            control_dict = {'battery_charge': action[3],
                            'battery_discharge': action[4],
                            'genset': action[0],
                            'grid_import': action[1],
                            'grid_export': action[2],
                            'pv_consummed': action[5]}
        else:
            control_dict = {'battery_charge': action[2],
                            'battery_discharge': action[3],
                            'grid_import': action[0],
                            'grid_export': action[1],
                            'pv_consummed': action[4]}

        return control_dict


class ContinuousMicrogridSampleEnv(ContinuousMicrogridEnv):
    """
    Same as ContinuousMicrogridEnv but uses samples generated from SampleAverageApproximation as states.
    """
    metadata = {'render.modes': ['human']}
    def __init__(self, microgrid, standardization=True,
                 forecast_args=None, baseline_sampling_args=None, max_episode_len=None):

        super().__init__(microgrid, standardization=standardization, max_episode_len=max_episode_len)
        self.forecast_args = forecast_args
        self.baseline_sampling_args = baseline_sampling_args
        self.saa = generate_sampler(self.microgrid, forecast_args)

    def reset(self, sampling_args=None):
        """
        Generates a new sample to use as load/pv/grid data. Then calls parent reset function.
        :param sampling_args:
        :return:
        """
        sample_reset(self.has_grid, self.saa, self.microgrid, sampling_args=sampling_args)
        observations = super().reset()
        return observations


class SafeExpMicrogridEnv(ContinuousMicrogridEnv):
    """
    ContinuousMicrogridEnv but with constraint functionality for safety layer.
    """
    def __init__(self, microgrid,
                 standardization=True,
                 balance_tolerance=1.,
                 scale_constraints=True,
                 only_inequality_constr=True,
                 trajectory_len=None,
                 max_episode_len=None):

        super().__init__(microgrid,
                         standardization=standardization,
                         trajectory_len=trajectory_len,
                         max_episode_len=max_episode_len)

        self.balance_tolerance=balance_tolerance
        self.scale_constraints = scale_constraints
        self.only_inequality_constr = only_inequality_constr

        self.n_constraints = 9 if self.has_genset else 7
        if only_inequality_constr:
            self.n_constraints -= 1

    def get_num_constraints(self):
        """
        Two for energy balance (one equality)
        One each p_charge, p_discharge, p_import, p_export
        Two p_genset if genset
        TODO: do you need pv_curtail and loss_load
        :return:
        """
        return self.n_constraints

    def get_constraint_values(self):
        """
        All constraints are set up here such that we return c_i for constraints of the form c_i<0
        :return:
        """
        inequality_constraints = self._get_inequality_constraints()
        energy_balance = self._get_energy_balance()

        if self.only_inequality_constr:
            constraints = inequality_constraints
            return constraints

        return np.append(inequality_constraints, energy_balance)

    def _get_energy_balance(self):
        if self.has_genset:
            p_import, p_export, p_charge, p_discharge, p_genset, load, pv, pv_consumed = \
                self.get_values('grid_import','grid_export', 'battery_charge', 'battery_discharge','genset','load','pv','pv_consummed')
        else:
            p_import, p_export, p_charge, p_discharge, load, pv, pv_consumed = \
                self.get_values('grid_import', 'grid_export', 'battery_charge', 'battery_discharge', 'load',
                                'pv', 'pv_consummed')

            p_genset = 0

        pv_curtailed = pv-pv_consumed

        energy_balance = np.array(p_import-p_export-p_charge+p_discharge+p_genset-pv_curtailed-load+pv)

        if self.scale_constraints:
            charge_scale_factor = float(self.microgrid.parameters.battery_capacity.squeeze())
            energy_balance /= charge_scale_factor

        return energy_balance

    def _get_inequality_constraints(self):
        constraints = []

        p_charge, p_discharge, p_max_charge, p_max_discharge = self.get_values('battery_charge','battery_discharge',
                                                                                'capa_to_charge','capa_to_discharge')

        if self.scale_constraints:
            charge_scale_factor = float(self.microgrid.parameters.battery_capacity.squeeze())

            constraints.append((p_charge-p_max_charge)/charge_scale_factor)
            constraints.append((p_discharge-p_max_discharge)/charge_scale_factor)
        else:
            constraints.append(p_charge - p_max_charge)
            constraints.append(p_discharge - p_max_discharge)

        p_max_import = self.microgrid.parameters['grid_power_import'].values[0]
        p_max_export = self.microgrid.parameters['grid_power_export'].values[0]
        p_import, p_export, grid_status = self.get_values('grid_import','grid_export','grid_status')

        if self.scale_constraints:
            constraints.append((p_import - p_max_import * grid_status)/p_max_import)
            constraints.append((p_export - p_max_export * grid_status)/p_max_export)
        else:
            constraints.append(p_import-p_max_import*grid_status)
            constraints.append(p_export-p_max_export*grid_status)

        battery_max = self.microgrid.parameters['battery_soc_max'].values[0]
        battery_min = self.microgrid.parameters['battery_soc_min'].values[0]
        battery_soc, = self.get_values('battery_soc')

        if self.scale_constraints:
            constraints.append((battery_soc - battery_max)/battery_max)
            constraints.append((battery_min - battery_soc)/battery_min)
        else:
            constraints.append(battery_soc-battery_max)
            constraints.append(battery_min-battery_soc)

        if self.has_genset:

            p_genset_max = self.microgrid.parameters['genset_rated_power'].values[0] * self.microgrid.parameters['genset_pmax'].values[0]
            p_genset_min = self.microgrid.parameters['genset_rated_power'].values[0] * self.microgrid.parameters['genset_pmin'].values[0]
            p_genset, = self.get_values('genset')

            # TODO what if we want it to be off? For now, this:
            if p_genset<1:
                if self.scale_constraints:
                    constraints.append((p_genset-1)/p_genset_max)
                    constraints.append((-p_genset-self.balance_tolerance)/p_genset_max)
                else:
                    constraints.append(p_genset - 1)
                    constraints.append(-p_genset - self.balance_tolerance)
            else:
                if self.scale_constraints:
                    constraints.append((p_genset - p_genset_max)/p_genset_max)
                    constraints.append((p_genset_min - p_genset)/p_genset_min)
                else:
                    constraints.append(p_genset-p_genset_max)
                    constraints.append(p_genset_min-p_genset)

        constraints = np.array(constraints)

        return constraints


class SafeExpMicrogridSampleEnv(SafeExpMicrogridEnv):
    def __init__(self,
                 microgrid,
                 standardization=True,
                 balance_tolerance=1.,
                 scale_constraints=True,
                 only_inequality_constr=True,
                 forecast_args=None,
                 baseline_sampling_args=None,
                 trajectory_len=None,
                 max_episode_len=None):
        super().__init__(microgrid,
                         standardization=standardization,
                         balance_tolerance=balance_tolerance,
                         scale_constraints=scale_constraints,
                         only_inequality_constr=only_inequality_constr,
                         trajectory_len=trajectory_len,
                         max_episode_len=max_episode_len)

        self.forecast_args = forecast_args
        self.forecast_args = forecast_args
        self.baseline_sampling_args = baseline_sampling_args
        self.saa = generate_sampler(self.microgrid, forecast_args)

    def reset(self, sampling_args=None):
        sample_reset(self.has_grid, self.saa, self.microgrid, sampling_args=sampling_args)
        observations = super().reset()
        return observations

