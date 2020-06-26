import sys
import unittest
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.regression.quantile_regression as quantile_regression

# Have start of exp. distribution be max(y_actual, y_predicted) where y_predicted is via interpolated curve

''' Actually, using uniform distribution:
If t is hour, then pick val: (where curve(t) is black line and y(t) is actual value)

    Suppose curve(t)>y_t:
        Then pick val uniformly on [curve(t)-2(curve(t)-y(t)),curve(t)]
        unless curve(t)-2(curve(t)-y(t)<0, in which case [0,curve(t)]

    If curve(t)<y_t:
        Pick val uniformly on [curve(t)-(y(t)-curve(t)),y(t)]

    This val is the max. Then, interpolate a parabola, using the x-intercepts and this max
    That's yo curve.

    Finally, add Gaussian noise to noise, variance equal to some (small) percentage of curve value


'''


class NoisyPVData:
    def __init__(self, pv_data = None,file_name = None):
        if pv_data is not None:
            if not isinstance(pv_data, pd.DataFrame):
                raise TypeError('known_data must be of type pd.DataFrame, is ({})'.format(type(pv_data)))

            self.unmunged_data = pv_data.copy()
            self.data = pv_data.copy()

        elif file_name is not None:
            self.data, self.unmunged_data = self.import_file(file_name)

        else:
            raise RuntimeError('Unable to initialize data')

        self.num_hours = len(self.data)
        self.munged = False
        self.interpolated = False
        self.daily_maxes = None
        self.feature_functions = None
        self.feature_names = None
        self.interpolated_coef = None

    def import_file(self,file_name):
        return pd.read_csv(file_name), pd.read_csv(file_name)

    def data_munge(self, verbose=False):

        hours = [j % 24 for j in range(self.num_hours)]
        day = [int(np.floor(j / 24)) for j in range(self.num_hours)]

        self.data['hour'] = pd.Series(data=hours)
        self.data['day'] = pd.Series(data=day)
        self.data = self.data.pivot(index='hour', columns='day', values='GH illum (lx)')

        indices_of_max = self.data.idxmax(axis=0)
        maxes = self.data.max(axis=0)
        indices_of_max.name = 'time_of_max'
        maxes.name = 'max_GHI'
        self.daily_maxes = pd.concat([indices_of_max, maxes], axis=1)
        self.daily_maxes['cumulative_hr'] = self.daily_maxes['time_of_max'] + self.daily_maxes.index.values * 24

        self.munged = True

        if verbose:
            print('Munging completed.')
            print(self.data.describe())
            print(self.daily_maxes.describe())

    def add_feature_columns(self, num_feature_functions=1, period_scale=1., ):
        feature_names = []
        feature_funcs = {}

        if not self.munged:
            raise RuntimeError('Data must be munged before adding feature columns or curve interpolation. '
                               'Call data_munge first')
        name = 'ones'
        self.daily_maxes[name] = pd.Series(data = [1.0]*len(self.daily_maxes['cumulative_hr']))

        def f(x):
            if isinstance(x, int) or isinstance(x, float):
                return 1.0
            else:
                return pd.Series(data=[1.0] * len(x))

        feature_names.append(name)
        feature_funcs[name] = f

        for k in range(num_feature_functions):
            if k % 2 == 0:
                num = int(np.floor((k + 2) / 2))
                name = 'cos' + str(num) + 'x'
                self.daily_maxes[name] = np.cos(
                    2 * num * np.pi / 8760. * period_scale * (self.daily_maxes['cumulative_hr'] - 173 * 24))

                def f(x):
                    return np.cos(2 * num * np.pi / 8760. * period_scale * (x - 173 * 24))
            #                 x_v = np.arange(8760)
            #                 plt.plot(x_v,f(x_v))
            #                 plt.show()
                assert all((f(self.daily_maxes['cumulative_hr']) == self.daily_maxes[name])), \
                    'Function declaration failed'


            elif k % 2 == 1:
                num = int(np.floor((k + 1) / 2))
                name = 'sin' + str(num) + 'x'
                self.daily_maxes[name] = np.sin(
                    2 * num * np.pi / 8760. * period_scale * (self.daily_maxes['cumulative_hr'] - 173 * 24))

                def f(x):
                    return np.sin(2 * num * np.pi / 8760. * period_scale * (x - 173 * 24))

            else:
                raise RuntimeError('Should not be here')

            feature_funcs[name] = f
            feature_names.append(name)

        for name in feature_names:
            assert all((feature_funcs[name](self.daily_maxes['cumulative_hr']) == self.daily_maxes[name])), \
                'Something wrong with feature functions'

        self.feature_functions = feature_funcs
        self.feature_names = feature_names

    def max_min_curve_interpolate(self, num_feature_functions=1,
                                  percentile=0.8,
                                  plot_curve=False,
                                  use_preset_params = True,
                                    params = 'sf'):
        """
        Interpolates the upper bound curve using cos/sin features so that at least
            percentile points are below the curve

        """
        sf_presets = {'period_scale': 0.8,
                      'q_max': 0.9,
                      'q_min': 0.25}
        houston_presets = {'period_scale':0.8,
             'q_max': 0.9,
             'q_min':0.05}

        presets = {'sf' : sf_presets, 'houston' : houston_presets}

        if use_preset_params:
            if params in presets.keys():
                parameters = presets[params]
                period_scale = parameters['period_scale']
                q_max = parameters['q_max']
                q_min = parameters['q_min']
            else:
                raise NameError('If use_preset_params is True, params must be one of {\'sf\',\'houston\' '
                                'denoting preset parameters')
        else:
            if not isinstance(params,dict):
                raise TypeError('params must be a dict of parameters, not ({})'.format(params))
            period_scale = params['period_scale']
            q_max = params['q_max']
            q_min = params['q_min']

        if not (1.0 >= percentile >= 0.0):
            raise ValueError('percentile must be in [0,1], is ({})'.format(percentile))

        self.add_feature_columns(num_feature_functions=num_feature_functions, period_scale=period_scale)

        x_vars = self.daily_maxes[self.feature_names]

        quantile_reg_model = quantile_regression.QuantReg(self.daily_maxes['max_GHI'], x_vars)
        results = quantile_reg_model.fit(q=q_max)
        max_coef = results.params
        max_curve = max_coef['ones']*x_vars['ones']

        for p_name in self.feature_names:
            if p_name == 'ones':
                continue
            else:
                max_curve+=max_coef[p_name]*x_vars[p_name]

        results = quantile_reg_model.fit(q=q_min)
        min_coef = results.params
        min_curve = min_coef['ones'] * x_vars['ones']

        for p_name in self.feature_names:
            if p_name == 'ones':
                continue
            else:
                min_curve += min_coef[p_name] * x_vars[p_name]

        self.interpolated_coef = {'max': max_coef, 'min': min_coef}
        self.interpolated = True

        if plot_curve:
            plt.scatter(self.daily_maxes['cumulative_hr']/24, self.daily_maxes['max_GHI'], color='r', marker='.')
            plt.plot(self.daily_maxes['cumulative_hr']/24, max_curve, label='{}th percentile'.format(int(100 * q_max)))
            plt.plot(self.daily_maxes['cumulative_hr']/24,
                     min_curve, color='k', label='{}th percentile'.format(int(100 * q_min)))
            plt.xlabel('Day')
            plt.legend()
            plt.title('Maximum daily PV upper/lower bounds')
            plt.show()
            # plt.plot(NPV.daily_maxes['cumulative_hr'], \
            #          0.25 * min_curve + 0.75 * max_curve, color='c')

        original_len = len(self.daily_maxes)
        current_daily_maxes = self.daily_maxes.copy()
        current_len = len(current_daily_maxes)

    def most_light_curve_eval(self, max_min, cumulative_hours=None, day_hour_pairs=None):
        """
        Evaluates the interpolated upper bound curve at the time values in val
        """
        to_return = []
        if max_min =='max':
            interpolated_coef = self.interpolated_coef['max']
        elif max_min == 'min':
            interpolated_coef = self.interpolated_coef['min']
        else:
            raise ValueError('max_min must be one of \'max\' or \'min\', is {}',format(max_min))

        if cumulative_hours is not None:
            try:
                cumul_times = pd.Series(data=cumulative_hours)
                y = pd.Series(data=[0] * len(cumul_times), name='Upper Bound Values')
            except TypeError:
                cumul_times = cumulative_hours
                y = interpolated_coef['ones']

            for j, name in enumerate(self.feature_names):
                feature_function = self.feature_functions[name]
                y += interpolated_coef[name] * feature_function(cumul_times)

            if len(y) == 1:
                y =  y[0]
            to_return.append(y)

        if day_hour_pairs is not None:
            cumul_times = []
            for pair in day_hour_pairs:
                if len(pair) != 2:
                    raise ValueError('pairs must be array-like of length two, containing days and hours')
                if pair[1] < 0 or pair[1] >= 24:
                    raise ValueError('hour must be in [0,23], is ({})'.format(pair[1]))

                cumul_times.append(pair[0] * 24 + pair[1])

            x = pd.Series(data=cumul_times)
            y_pairs = pd.Series(data=[0] * len(cumul_times), name='Upper Bound Values')
            for j, name in enumerate(self.feature_names):
                feature_function = self.feature_functions[name]
                y_pairs += interpolated_coef[name] * feature_function(x)

            if len(y_pairs) == 1:
                y_pairs = y_pairs[0]

            to_return.append(y_pairs)

        if len(to_return) == 1:
            to_return = to_return[0]

        return to_return

    def simulate_data(self,
                      noise_types=('uniform', None),
                      noise_params=({'lower': 0, 'upper': 0}, {'std_ratio': 0.05}),
                      return_format = 'stacked',
                      plot_noisy=False,
                      days_to_plot=(0, 5),
                      verbose=False
                      ):

        potential_noises = {0: ('uniform', 'triangular'),
                            1: (None, 'gaussian')}

        noise_parameters = ({'lower': 0, 'upper': 0, 'mode':0.5}, {'std_ratio': 0.05})

        for j, noise in enumerate(noise_types):
            if noise not in potential_noises[j]:
                raise ValueError('Noise ({}) not recognized in position ({}), must be one of {}'.format(
                    noise, j, potential_noises[j]))

            if not self.interpolated:
                raise RuntimeError('Must have an interpolating curve before adding noise. '
                                   'Call max_min_curve_interpolate first.')
        if len(noise_params)!=2:
            raise TypeError('Unable to parse noise_params, must be array-like length 2')

        for j, v in enumerate(noise_params):
            if v is not None and not isinstance(v, dict):
                raise TypeError('Element ({}) in noise_params must be None or dict, is {}'.format(j, type(v)))
            if v is not None:
                for key in noise_parameters[j].keys():
                    if key in v.keys():
                        noise_parameters[j][key] = v[key]



        noisy_data = self.data.copy()  # Columns are days, index is hours

        # Need three points for interpolation: two zeros
        # Get points for each day:

        lower_distribution_bounds = []
        upper_distribution_bounds = []

        for day in noisy_data.columns:
            if noisy_data[day][0] != 0:
                raise RuntimeError('It appears that it is sunny at midnight of day ({}). No good.'.format(day))
            if noisy_data[day][23] != 0:
                raise RuntimeError('It appears that it is sunny at 11PM of day ({}). No good.'.format(day))

            night_hours = np.where(noisy_data[day] == 0)[0]
            next_night_hours = np.roll(night_hours, -1)
            index_of_dawn = np.where(night_hours + 1 != next_night_hours)[0][0]
            dawn_time = night_hours[index_of_dawn]  # Gives a zero: (dawn_time,0)
            dusk_time = night_hours[index_of_dawn + 1]  # Another zero: (dusk_time,0)

            # time_of_most_light = self.daily_maxes.loc[day, 'time_of_max']
            time_of_most_light = (dawn_time+dusk_time)/2.0
            interpolated_least_light = self.most_light_curve_eval(max_min ='min',
                                                                 day_hour_pairs=((day, time_of_most_light),))
            interpolated_most_light = self.most_light_curve_eval(max_min ='max',
                                                                day_hour_pairs=((day, time_of_most_light),))

            lower_b = interpolated_least_light
            upper_b = interpolated_most_light
            spread = upper_b - lower_b

            if noise_types[0] == 'uniform':
                low = lower_b + noise_parameters[0]['lower'] * spread
                high = upper_b + (noise_parameters[0]['upper']-1) * spread
                lower_distribution_bounds.append(low)
                upper_distribution_bounds.append(high)
                peak_val = np.random.uniform(low=low, high=high)

                if verbose:
                    print('Day {}'.format(day))
                    print('Using uniform distribution between {} and {}'.format(round(low, 1),
                                                                                       round(high, 1)))
                    print('Unscaled bounds: [{},{}]'.format(round(lower_b, 1), round(upper_b, 1)))
                    print('Selected daily peak value {}'.format(peak_val))

            elif noise_types[0] == 'triangular':
                low = lower_b + noise_parameters[0]['lower'] * spread
                high = upper_b + (noise_parameters[0]['upper']-1) * spread
                if 'mode' in noise_parameters[0].keys():
                    mode_param = noise_parameters[0]['mode']
                    if not 0 <= mode_param <= 1:
                        raise ValueError(
                            'mode parameter ({}) invalid, must be scale value in [0,1]'.format(mode_param))
                    mode = spread * mode_param + lower_b
                    assert high >= mode >= low, 'mode computation did not work'
                else:
                    mode = 0.5 * (lower_b + upper_b)

                lower_distribution_bounds.append(low)
                upper_distribution_bounds.append(high)

                peak_val = np.random.triangular(left=low, mode=mode, right=high)

                if verbose:
                    print('Day {}'.format(day))
                    print('using triangular distribution with low {}, mode {}, high {}'.format(
                        round(low, 1), round(mode, 1), round(high, 1)))
                    print('Unscaled bounds: [{},{}]'.format(round(lower_b, 1), round(upper_b, 1)))
                    print('Selected daily peak value {}'.format(peak_val))

            else:
                raise RuntimeError('Fell through in noise_types, unable to recognize ({})'.format(noise_types[0]))

            daytime_x = np.array([dawn_time, time_of_most_light, dusk_time])
            daytime_y = np.array([0, peak_val, 0])
            if any(np.diff(daytime_x) <= 0):
                raise RuntimeError('Something is wrong in interpolating daily curves, '
                                   'have dawn/peak/dusk times as ({}), not in order'.format(daytime_x))

            # Interpolate that
            f = interp1d(daytime_x, daytime_y, kind='quadratic', bounds_error=False, fill_value=0)
            noisy_data[day] = f(noisy_data.index)

        if noise_types[1] == 'gaussian':
            noisy_data += np.random.normal(scale=noise_parameters[1]['std_ratio'] * noisy_data)

        if plot_noisy:
            stacked_data = noisy_data.transpose().stack()
            stacked_data = stacked_data.reset_index()
            stacked_data = stacked_data.drop(columns=['hour', 'day'])
            indices = slice(24 * days_to_plot[0],24 * days_to_plot[1])
            daily_slice = slice(*days_to_plot)

            plt.plot(stacked_data[indices].index, stacked_data[indices].values, label='noisy')
            plt.plot(stacked_data[indices].index, self.unmunged_data[indices].values, label='original')
            plt.plot(stacked_data[indices].index,
                     self.most_light_curve_eval('max',cumulative_hours=stacked_data[indices].index),color='k',label='UB')
            plt.plot(stacked_data[indices].index,
                     self.most_light_curve_eval('min', cumulative_hours=stacked_data[indices].index), color='c',
                     label='LB')

            plt.scatter(self.daily_maxes['cumulative_hr'].iloc[daily_slice],lower_distribution_bounds[daily_slice],
                        marker='.', color='r')
            plt.scatter(self.daily_maxes['cumulative_hr'].iloc[daily_slice], upper_distribution_bounds[daily_slice],
                        marker='.', color='r')
            plt.legend()
            plt.show()

        if return_format == 'stacked':
            return stacked_data
        return noisy_data


class NoisyLoadData:
    def __init__(self, load_data = None, file_name = None):
        if load_data is not None:
            if not isinstance(load_data, pd.DataFrame):
                raise TypeError('known_data must be of type pd.DataFrame, is ({})'.format(type(load_data)))

            self.unmunged_data = load_data.copy()
            self.data = load_data.copy()

        elif file_name is not None:
            self.data, self.unmunged_data = self.import_file(file_name)

        else:
            raise RuntimeError('Unable to initialize data')

        self.num_hours = len(load_data)
        self.munged = False
        self.interpolated = False

    def import_file(self,file_name):
        return pd.read_csv(file_name), pd.read_csv(file_name)

    def data_munge(self, verbose=False):

        hours = [j % 24 for j in range(self.num_hours)]
        day = [int(np.floor(j / 24)) for j in range(self.num_hours)]

        self.data['hour'] = pd.Series(data=hours)
        self.data['day'] = pd.Series(data=day)
        self.data = self.data.pivot(index='day', columns='hour', values='Electricity:Facility [kW](Hourly)')
        self.data['day_of_week'] = self.data.index % 7

        self.load_mean = self.data.groupby(['day_of_week']).mean()
        self.load_std = self.data.groupby(['day_of_week']).std()

        self.munged = True

    def sample(self, distribution='gaussian', variance_scale=1.):

        possible_distributions = ('gaussian',)
        if distribution not in possible_distributions:
            raise ValueError(
                'distribution {} not recognized, must be one of ({})'.format(distribution, possible_distributions))

        if distribution == 'gaussian':
            copied_mean = self.data.copy()
            copied_mean = copied_mean.set_index([copied_mean.index, 'day_of_week'])
            copied_std = copied_mean.copy()

            for ind in copied_mean.index:
                copied_mean.loc[ind] = self.load_mean.loc[ind[1]]
                copied_std.loc[ind] = self.load_std.loc[ind[1]]

        data_sample = pd.DataFrame(data=np.random.normal(loc=copied_mean, scale=variance_scale * copied_std),
                                   index=self.data.index,
                                   columns=self.data.columns[:-1])
        return data_sample

    def plot_sample_v_original(self, sample, days_to_plot=(0, 10)):
        stacked_data = sample.stack()
        stacked_data = stacked_data.reset_index()
        cols_list = list(stacked_data.columns.values)

        stacked_data = stacked_data.drop(columns=['day', 'hour'])

        plt.plot(stacked_data[24 * days_to_plot[0]:24 * days_to_plot[1]].values, label='sample')
        plt.plot(self.unmunged_data[24 * days_to_plot[0]:24 * days_to_plot[1]].values, label='original')
        plt.title('Load Sample')
        plt.legend()
        plt.show()
