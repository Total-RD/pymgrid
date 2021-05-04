import unittest
from pymgrid.utils.DataGenerator import *
from pandas import Series
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
from matplotlib import pyplot as plt

def create_pv_test_set():
    test_set = np.zeros(48)
    test_set[7:20] = np.arange(7,20)
    test_set[7:20] = -20*(test_set[7:20]-13)**2  + 30*-1**(np.arange(7,20) % 2) + 5*(np.arange(6,19) %5)
    test_set[7:20] += -1*np.min(test_set[7:20])

    n = 24
    test_set[7+n:20+n] = np.arange(7,20)
    test_set[7+n:20+n] = -20*(test_set[7+n:20+n]-13)**2  + 20*-1**(np.arange(6,19) % 2) + 4*(np.arange(6,19) %3)
    test_set[7+n:20+n] += -1*np.min(test_set[7+n:20+n])

    return test_set/10


class TestNoisyPV(unittest.TestCase):
    def setUp(self):

        self.test_data = create_pv_test_set()
        self.test_series = Series(data = self.test_data)


    def test_init(self):
        NPV = NoisyPVData(pv_data = self.test_series)
        df = pd.DataFrame(self.test_data)
        assert_frame_equal(NPV.unmunged_data, df)
        assert_frame_equal(NPV.data, df)

    def test_data_munge(self):
        NPV = NoisyPVData(pv_data=self.test_series)
        NPV.data_munge()

        # Assertions:
        assert_array_equal(NPV.data.values[:,0],self.test_data[:24])
        assert_array_equal(NPV.data.values[:,1],self.test_data[24:])
        assert_array_equal(NPV.daily_maxes['time_of_max'].values, np.array([13,13]))
        assert_array_equal(NPV.daily_maxes['cumulative_hr'], np.array([13, 37]))
        self.assertTrue(NPV.munged)

    def test_add_feature_columns(self):
        NPV = NoisyPVData(pv_data=self.test_series)
        NPV.data_munge()

        num_feature_functions = 1
        period_scale = 0.8

        NPV._add_feature_columns(num_feature_functions=num_feature_functions, period_scale=period_scale)

        self.assertIn('ones', NPV.daily_maxes.columns.values)
        self.assertIn('cos1x', NPV.daily_maxes.columns.values)
        assert_array_equal(NPV.daily_maxes['ones'], np.array([1,1]))
        cos1x = np.cos(
                    2  * np.pi / 8760. * period_scale * (NPV.daily_maxes['cumulative_hr'] - 173 * 24))
        assert_array_equal(NPV.daily_maxes['cos1x'], cos1x)

        self.assertListEqual(NPV.feature_names, ['ones', 'cos1x'])

        for name in NPV.feature_names:
             assert_array_equal(NPV.feature_functions[name](NPV.daily_maxes['cumulative_hr']).values, NPV.daily_maxes[name].values)

    # TODO Finish wriitng these starting at max_min_curve_interpolate


class TestNoisyGrid(unittest.TestCase):

    def setUp(self) -> None:
        always_on = np.ones(48)
        self.always_on = pd.Series(always_on)
        self.with_outages = self.always_on.copy()
        self.with_outages.iloc[3:6] = 0
        self.with_outages.iloc[40:47] = 0
        self.with_outages_data = dict(naive_probabilities = np.array([10/48, 38/48]),
                                      occurences=np.array([10,37]),
                                      transition_prob_matrix = np.array([
                                                                [8 / 10, 2 / 10],
                                                                [2/37, 35/37]
                                                                ]))
        self.dist_types = ('naive', 'markov')

    def test_init(self):

        for dist_type in self.dist_types:
            for data in self.always_on, self.with_outages:
                NGD = NoisyGridData(data,dist_type=dist_type)
                assert_series_equal(data, NGD.data)
                assert_series_equal(data, NGD.unmunged_data)

    def test_bad_grid_data(self):
        grid_data = self.with_outages.copy()
        grid_data[5] = -3
        try:
            NoisyGridData(grid_data)
        except ValueError:
            pass
        except Exception:
            self.fail('unexpected exception raised')
        else:
            self.fail('ValueError not raised')
        grid_data[5] = 1.1
        try:
            NoisyGridData(grid_data)
        except ValueError:
            pass
        except Exception:
            self.fail('unexpected exception raised')
        else:
            self.fail('ValueError not raised')

    def test_learn_distribution_always_on_naive(self):
        NGD = NoisyGridData(self.always_on, dist_type='naive')
        self.assertFalse(NGD.has_distribution)
        NGD.learn_distribution()
        self.assertTrue(NGD.has_distribution)

        assert_array_equal(NGD.transition_prob_matrix, np.array([0, 1]))

    def test_learn_distribution_always_on_markov(self):
        NGD = NoisyGridData(self.always_on, dist_type='markov')
        self.assertFalse(NGD.has_distribution)
        NGD.learn_distribution()
        self.assertTrue(NGD.has_distribution)

        assert_array_equal(NGD.occurrences, np.array([0, 47]))
        assert_array_equal(NGD.transition_prob_matrix, np.array([[1,0],[0,1]]))

    def test_learn_distribution_with_outages_naive(self):
        NGD = NoisyGridData(self.with_outages, dist_type='naive')
        self.assertFalse(NGD.has_distribution)
        NGD.learn_distribution()
        self.assertTrue(NGD.has_distribution)

        assert_array_almost_equal(NGD.transition_prob_matrix, self.with_outages_data['naive_probabilities'])

    def test_learn_distribution_with_outages_markov(self):
        NGD = NoisyGridData(self.with_outages, dist_type='markov')
        self.assertFalse(NGD.has_distribution)
        NGD.learn_distribution()
        self.assertTrue(NGD.has_distribution)

        assert_array_almost_equal(NGD.occurrences, self.with_outages_data['occurences'])
        assert_array_almost_equal(NGD.transition_prob_matrix, self.with_outages_data['transition_prob_matrix'])

    def test_sample_always_on_naive(self):
        NGD = NoisyGridData(self.always_on, dist_type='naive')
        NGD.learn_distribution()
        sample = NGD.sample()
        assert_array_equal(sample, np.ones(48))

    def test_sample_always_on_markov(self):
        NGD = NoisyGridData(self.always_on, dist_type='markov')
        NGD.learn_distribution()
        sample = NGD.sample()
        assert_array_equal(sample, np.ones(48))

    def test_sample_with_outages_naive(self):
        # This is a ridiculous unit test
        np.random.seed(0)
        num_tests = 50

        NGD = NoisyGridData(self.with_outages, dist_type='naive')
        NGD.learn_distribution()

        probs_list = []
        for j in range(num_tests):
            sample = NGD.sample()
            new_NGD = NoisyGridData(sample, dist_type='naive')
            new_NGD.learn_distribution()
            probs_list.append(new_NGD.transition_prob_matrix)

        transition_prob_mean = np.mean(np.array(probs_list), axis=0)
        assert_array_almost_equal(self.with_outages_data['naive_probabilities'], transition_prob_mean, decimal=2)

    def test_sample_with_outages_markov(self):
        # This is also a ridiculous unit test
        np.random.seed(0)
        num_tests = 50

        NGD = NoisyGridData(self.with_outages, dist_type='markov')
        NGD.learn_distribution()

        probs_list = []
        for j in range(num_tests):
            sample = NGD.sample()
            new_NGD = NoisyGridData(sample, dist_type='markov')
            new_NGD.learn_distribution()
            probs_list.append(new_NGD.transition_prob_matrix)

        transition_prob_mean = np.mean(np.array(probs_list), axis=0)
        assert_array_almost_equal(self.with_outages_data['transition_prob_matrix'], transition_prob_mean, decimal=1)


class TestNoisyLoad(unittest.TestCase):
    def setUp(self) -> None:
        self.n_days = 12

        load_data = np.array([304, 205, 200, 200, 202, 306, 524, 611, 569, 466, 571, 579, 569, 470, 466, 465, 597, 625, 620, 525, 521, 524, 522, 531, 305, 200, 199, 200, 202, 306, 524, 611, 568, 466, 568, 579, 569, 467, 467, 466, 597, 626, 626, 525, 525, 524, 522, 533])
        load_data = np.concatenate([load_data + j % 5 for j in range(int(self.n_days/2))])

        self.load_data = pd.Series(data = load_data)

    def test_init(self):
        NLD = NoisyLoadData(load_data=self.load_data)
        assert_frame_equal(NLD.data, self.load_data.to_frame())
        assert_frame_equal(NLD.unmunged_data, self.load_data.to_frame())

    def test_data_munge(self):
        NLD = NoisyLoadData(load_data=self.load_data)

        self.assertFalse(NLD.munged)
        NLD.data_munge()
        self.assertTrue(NLD.munged)

        self.assertTupleEqual(NLD.load_mean.shape, (7,24))
        self.assertTupleEqual(NLD.load_std.shape, (7,24))

        self.assertEqual(NLD.data.shape[0], self.n_days)
        self.assertFalse(np.isnan(NLD.load_mean).any(axis=None))
        self.assertFalse(np.isnan(NLD.load_std).any(axis=None))

        for j in range(7):
            NLD_computed_avg = NLD.load_mean.iloc[j,:].values
            NLD_computed_std = NLD.load_std.iloc[j,:].values


            slice = self.load_data[24*j:24*(j+1)].values

            for k in range(1,(self.n_days-j) // 7 + 1):
                if (j+k*7)>=self.n_days:
                    continue
                slice = np.stack((slice,self.load_data[24*(j+k*7):24*(j+k*7+1)]))

            if len(slice.shape) == 1:
                slice = slice.reshape((1,24))
                assert_array_almost_equal(NLD_computed_avg, np.mean(slice, axis=0))
            else:
                assert_array_almost_equal(NLD_computed_avg, np.mean(slice, axis=0))
                assert_array_almost_equal(NLD_computed_std, np.std(slice, axis=0, ddof=1))

        # Todo finish this test



if __name__ == '__main__':

    unittest.main()

