"""Tests of the 38 Druckmann 2013 test classes"""

import unittest
import pickle
import quantities as pq
from neuronunit.tests.druckmann2013 import *
from neuronunit.neuromldb import NeuroMLDBStaticModel
from numpy import array
from quantities import *

class Druckmann2013BaseTestCase:
    class BaseTest(unittest.TestCase):
        # Use cached pickled model for testing (faster, no external dependency)
        import sys, os
        pickle_file = 'nmldb-model-cache.pkl'

        # Handle being executed from test.sh
        if not os.path.exists(pickle_file):
            pickle_file = os.path.join('neuronunit','unit_test',pickle_file)

        try:
            with open(pickle_file, 'rb') as f:
                if sys.version_info[0] >= 3:
                    model_cache = pickle.load(f, encoding='Latin-1')
                else:
                    model_cache = pickle.load(f)
        except:
            model_cache = {}

        predicted = {}

        def set_expected(self, expected_values):
            assert len(expected_values) == len(self.test_set)

            for i, v in enumerate(expected_values):
                self.test_set[i]['expected'] = v

        def setUp(self):
            assert self.model_id

            if not hasattr(self, "expected"):
                self.expected = [0.0 for i in range(38)]

            self.model = self.get_model()
            self.standard = self.model.nmldb_model.get_druckmann2013_standard_current()
            self.strong = self.model.nmldb_model.get_druckmann2013_strong_current()
            self.ir_currents = self.model.nmldb_model.get_druckmann2013_input_resistance_currents()

            self.test_set = [
                {'test': AP12AmplitudeDropTest(self.standard), 'units': pq.mV, 'expected': None},
                {'test': AP1SSAmplitudeChangeTest(self.standard), 'units': pq.mV, 'expected': None},
                {'test': AP1AmplitudeTest(self.standard), 'units': pq.mV, 'expected': None},
                {'test': AP1WidthHalfHeightTest(self.standard), 'units': pq.ms, 'expected': None},
                {'test': AP1WidthPeakToTroughTest(self.standard), 'units': pq.ms, 'expected': None},
                {'test': AP1RateOfChangePeakToTroughTest(self.standard), 'units': pq.mV/pq.ms, 'expected': None},
                {'test': AP1AHPDepthTest(self.standard), 'units': pq.mV, 'expected': None},
                {'test': AP2AmplitudeTest(self.standard), 'units': pq.mV, 'expected': None},
                {'test': AP2WidthHalfHeightTest(self.standard), 'units': pq.ms, 'expected': None},
                {'test': AP2WidthPeakToTroughTest(self.standard), 'units': pq.ms, 'expected': None},
                {'test': AP2RateOfChangePeakToTroughTest(self.standard), 'units': pq.mV/pq.ms, 'expected': None},
                {'test': AP2AHPDepthTest(self.standard), 'units': pq.mV, 'expected': None},
                {'test': AP12AmplitudeChangePercentTest(self.standard), 'units': None, 'expected': None},
                {'test': AP12HalfWidthChangePercentTest(self.standard), 'units': None, 'expected': None},
                {'test': AP12RateOfChangePeakToTroughPercentChangeTest(self.standard), 'units': None, 'expected': None},
                {'test': AP12AHPDepthPercentChangeTest(self.standard), 'units': None, 'expected': None},
                {'test': InputResistanceTest(injection_currents=self.ir_currents), 'units': pq.Quantity(1,'MOhm'), 'expected': None},
                {'test': AP1DelayMeanTest(self.standard), 'units': pq.ms, 'expected': None},
                {'test': AP1DelaySDTest(self.standard), 'units': pq.ms, 'expected': None},
                {'test': AP2DelayMeanTest(self.standard), 'units': pq.ms, 'expected': None},
                {'test': AP2DelaySDTest(self.standard), 'units': pq.ms, 'expected': None},
                {'test': Burst1ISIMeanTest(self.standard), 'units': pq.ms, 'expected': None},
                {'test': Burst1ISISDTest(self.standard), 'units': pq.ms, 'expected': None},
                {'test': InitialAccommodationMeanTest(self.standard), 'units': None, 'expected': None},
                {'test': SSAccommodationMeanTest(self.standard), 'units': None, 'expected': None},
                {'test': AccommodationRateToSSTest(self.standard), 'units': 1/pq.ms, 'expected': None},
                {'test': AccommodationAtSSMeanTest(self.standard), 'units': None, 'expected': None},
                {'test': AccommodationRateMeanAtSSTest(self.standard), 'units': pq.ms, 'expected': None},
                {'test': ISICVTest(self.standard), 'units': None, 'expected': None},
                {'test': ISIMedianTest(self.standard), 'units': pq.ms, 'expected': None},
                {'test': ISIBurstMeanChangeTest(self.standard), 'units': None, 'expected': None},
                {'test': SpikeRateStrongStimTest(self.strong), 'units': pq.Hz, 'expected': None},
                {'test': AP1DelayMeanStrongStimTest(self.strong), 'units': pq.ms, 'expected': None},
                {'test': AP1DelaySDStrongStimTest(self.strong), 'units': pq.ms, 'expected': None},
                {'test': AP2DelayMeanStrongStimTest(self.strong), 'units': pq.ms, 'expected': None},
                {'test': AP2DelaySDStrongStimTest(self.strong), 'units': pq.ms, 'expected': None},
                {'test': Burst1ISIMeanStrongStimTest(self.strong), 'units': pq.ms, 'expected': None},
                {'test': Burst1ISISDStrongStimTest(self.strong), 'units': pq.ms, 'expected': None},
            ]

            self.set_expected(self.expected)

        def get_model(self):

            if self.model_id not in self.__class__.model_cache:
                print('Model ' + self.model_id + ' not in cache. Downloading waveforms...')
                self.__class__.model_cache[self.model_id] = NeuroMLDBStaticModel(self.model_id)

            if self.model_id not in self.__class__.predicted:
                self.__class__.predicted[self.model_id] = [None for i in range(38)] # There are 38 tests

            return self.__class__.model_cache[self.model_id]

        @classmethod
        def pickle_model_cache(cls):
            '''
            Use this function to re-pickle models after tests have
            run (and waveforms have been downloaded from NeuroML-DB.org)
            :return: Nothing, models are saved in a pickle file
            '''

            for model in cls.model_cache.values():
                # Clear AnalogSignal versions (to reduce file size) and pickle the model (to speed up unit tests)
                model.vm = None
                model.nmldb_model.waveform_signals = {}
                model.nmldb_model.steady_state_waveform = None

            import pickle
            with open(cls.pickle_file, 'w') as fp:
                pickle.dump(cls.model_cache, fp)

        def run_test(self, index):
            test_class = self.test_set[index]['test']
            expected = self.test_set[index]['expected']
            units = self.test_set[index]['units']

            if units is None:
                units = pq.dimensionless

            try:
                predicted = test_class.generate_prediction(self.model)['mean']

            except:
                predicted = -3333333 * units
                import traceback
                print(traceback.format_exc())

            self.__class__.predicted[self.model_id][index] = {
                'test': test_class.__class__.__name__,
                'predicted': predicted
            }

            if predicted is not None and expected is not None:
                self.assertTrue(predicted - expected * units <= 0.001 * units)
            else:
                self.assertTrue(predicted == expected)

        def runTest(self):
            for i, t in enumerate(self.test_set):
                try:
                    self.run_test(i)
                except:
                    print('Test ' + str(i) + ' failed')
                    import traceback
                    print(traceback.format_exc())

        def test_0(self):
            self.run_test(0)

        def test_1(self):
            self.run_test(1)

        def test_2(self):
            self.run_test(2)

        def test_3(self):
            self.run_test(3)

        def test_4(self):
            self.run_test(4)

        def test_5(self):
            self.run_test(5)

        def test_6(self):
            self.run_test(6)

        def test_7(self):
            self.run_test(7)

        def test_8(self):
            self.run_test(8)

        def test_9(self):
            self.run_test(9)

        def test_10(self):
            self.run_test(10)

        def test_11(self):
            self.run_test(11)

        def test_12(self):
            self.run_test(12)

        def test_13(self):
            self.run_test(13)

        def test_14(self):
            self.run_test(14)

        def test_15(self):
            self.run_test(15)

        def test_16(self):
            self.run_test(16)

        def test_17(self):
            self.run_test(17)

        def test_18(self):
            self.run_test(18)

        def test_19(self):
            self.run_test(19)

        def test_20(self):
            self.run_test(20)

        def test_21(self):
            self.run_test(21)

        def test_22(self):
            self.run_test(22)

        def test_23(self):
            self.run_test(23)

        def test_24(self):
            self.run_test(24)

        def test_25(self):
            self.run_test(25)

        def test_26(self):
            self.run_test(26)

        def test_27(self):
            self.run_test(27)

        def test_28(self):
            self.run_test(28)

        def test_29(self):
            self.run_test(29)

        def test_30(self):
            self.run_test(30)

        def test_31(self):
            self.run_test(31)

        def test_32(self):
            self.run_test(32)

        def test_33(self):
            self.run_test(33)

        def test_34(self):
            self.run_test(34)

        def test_35(self):
            self.run_test(35)

        def test_36(self):
            self.run_test(36)

        def test_37(self):
            self.run_test(37)

        @classmethod
        def print_predicted(cls):

            for model_id in cls.predicted.keys():
                print('Predicted values for '+model_id+': [')
                for i, p in enumerate(cls.predicted[model_id]):
                    if p['predicted'] is not None:
                        print('             ' + str((p['predicted'] * dimensionless).magnitude).rjust(25) + ', # ' + p['test'])
                    else:
                        print('             '+'None'.rjust(25)+', # ' + p['test'])

                print('         ]')

class Model1TestCase(Druckmann2013BaseTestCase.BaseTest):
    def setUp(self):
        self.model_id = 'NMLCL000086'
        self.expected = [
                          [1.75890231], # AP12AmplitudeDropTest
                          [3.11139343], # AP1SSAmplitudeChangeTest
                           [78.188898], # AP1AmplitudeTest
                   0.18999999999999984, # AP1WidthHalfHeightTest
                                [5.02], # AP1WidthPeakToTroughTest
                        [-20.76529225], # AP1RateOfChangePeakToTroughTest
                          [26.0528691], # AP1AHPDepthTest
                         [76.42999568], # AP2AmplitudeTest
                   0.18000000000000005, # AP2WidthHalfHeightTest
                                [4.62], # AP2WidthPeakToTroughTest
                        [-22.90667367], # AP2RateOfChangePeakToTroughTest
                         [29.39883667], # AP2AHPDepthTest
                         [-2.24955506], # AP12AmplitudeChangePercentTest
                    -5.263157894736734, # AP12HalfWidthChangePercentTest
                         [10.31231053], # AP12RateOfChangePeakToTroughPercentChangeTest
                         [12.84299075], # AP12AHPDepthPercentChangeTest
                    1.1415498952202383, # InputResistanceTest
                      7.67999999999995, # AP1DelayMeanTest
                                   0.0, # AP1DelaySDTest
                    122.41000000000008, # AP2DelayMeanTest
                                   0.0, # AP2DelaySDTest
                    119.37000000000009, # Burst1ISIMeanTest
                 2.842170943040401e-14, # Burst1ISISDTest
                                 -25.0, # InitialAccommodationMeanTest
                                   0.0, # SSAccommodationMeanTest
                                   0.0, # AccommodationRateToSSTest
                   -7.5109801211565745, # AccommodationAtSSMeanTest
                    22.453539761369523, # AccommodationRateMeanAtSSTest
                     54.75354914623598, # ISICVTest
                    124.03999999999996, # ISIMedianTest
                     8.088555739562318, # ISIBurstMeanChangeTest
                                  21.5, # SpikeRateStrongStimTest
                    2.7200000000000273, # AP1DelayMeanStrongStimTest
                                   0.0, # AP1DelaySDStrongStimTest
                     22.25999999999999, # AP2DelayMeanStrongStimTest
                                   0.0, # AP2DelaySDStrongStimTest
                     33.43999999999994, # Burst1ISIMeanStrongStimTest
                                   0.0, # Burst1ISISDStrongStimTest
         ]



        super(Model1TestCase, self).setUp()

class Model2TestCase(Druckmann2013BaseTestCase.BaseTest):
    def setUp(self):
        self.model_id = 'NMLCL000001'
        self.expected = [
                            [2.390436], # AP12AmplitudeDropTest
                          [2.20057816], # AP1SSAmplitudeChangeTest
                           [74.860976], # AP1AmplitudeTest
                                  0.82, # AP1WidthHalfHeightTest
                                 [1.7], # AP1WidthPeakToTroughTest
                        [-49.82091765], # AP1RateOfChangePeakToTroughTest
                            [9.834584], # AP1AHPDepthTest
                            [72.47054], # AP2AmplitudeTest
                                  0.82, # AP2WidthHalfHeightTest
                                [1.83], # AP2WidthPeakToTroughTest
                        [-44.61279717], # AP2RateOfChangePeakToTroughTest
                          [9.17087882], # AP2AHPDepthTest
                         [-3.19316702], # AP12AmplitudeChangePercentTest
                                   0.0, # AP12HalfWidthChangePercentTest
                        [-10.45368235], # AP12RateOfChangePeakToTroughPercentChangeTest
                         [-6.74868583], # AP12AHPDepthPercentChangeTest
                     10.47565649894051, # InputResistanceTest
                     5.070000000000164, # AP1DelayMeanTest
                                   0.0, # AP1DelaySDTest
                    18.019999999999982, # AP2DelayMeanTest
                                   0.0, # AP2DelaySDTest
                    13.489999999999895, # Burst1ISIMeanTest
                                   0.0, # Burst1ISISDTest
                    -3.571428571428571, # InitialAccommodationMeanTest
                    -3.571428571428571, # SSAccommodationMeanTest
                  -0.11143302874972157, # AccommodationRateToSSTest
                   -11.300595234774967, # AccommodationAtSSMeanTest
                     13.86602232475643, # AccommodationRateMeanAtSSTest
                       95.817295850982, # ISICVTest
                    14.600000000000364, # ISIMedianTest
                     8.339768339769652, # ISIBurstMeanChangeTest
                                   2.5, # SpikeRateStrongStimTest
                    1.5499999999999545, # AP1DelayMeanStrongStimTest
                                   0.0, # AP1DelaySDStrongStimTest
                     6.920000000000073, # AP2DelayMeanStrongStimTest
                                   0.0, # AP2DelaySDStrongStimTest
                     5.465000000000032, # Burst1ISIMeanStrongStimTest
                                   0.0, # Burst1ISISDStrongStimTest
         ]

        super(Model2TestCase, self).setUp()

class Model3TestCase(Druckmann2013BaseTestCase.BaseTest):
    def setUp(self):
        self.model_id = 'NMLCL001128'
        self.expected = [
                          [2.56788933], # AP12AmplitudeDropTest
                          [2.53662642], # AP1SSAmplitudeChangeTest
                         [62.88228933], # AP1AmplitudeTest
                    1.8800000000000001, # AP1WidthHalfHeightTest
                               [13.95], # AP1WidthPeakToTroughTest
                         [-6.49617036], # AP1RateOfChangePeakToTroughTest
                         [27.73928724], # AP1AHPDepthTest
                             [60.3144], # AP2AmplitudeTest
                                  2.04, # AP2WidthHalfHeightTest
                                [15.8], # AP2WidthPeakToTroughTest
                         [-5.65219231], # AP2RateOfChangePeakToTroughTest
                         [28.99023843], # AP2AHPDepthTest
                         [-4.08364479], # AP12AmplitudeChangePercentTest
                     8.510638297872337, # AP12HalfWidthChangePercentTest
                        [-12.99193233], # AP12RateOfChangePeakToTroughPercentChangeTest
                          [4.50967316], # AP12AHPDepthPercentChangeTest
                    1065.7851209290623, # InputResistanceTest
                    25.579999999999927, # AP1DelayMeanTest
                                   0.0, # AP1DelaySDTest
                    116.06999999999994, # AP2DelayMeanTest
                                   0.0, # AP2DelaySDTest
                     92.70500000000004, # Burst1ISIMeanTest
                                   0.0, # Burst1ISISDTest
                                   0.0, # InitialAccommodationMeanTest
                                   0.0, # SSAccommodationMeanTest
                                   0.0, # AccommodationRateToSSTest
                   -4.6888507590764315, # AccommodationAtSSMeanTest
                    17.827898489946417, # AccommodationRateMeanAtSSTest
                      97.6465924440232, # ISICVTest
                     94.94000000000005, # ISIMedianTest
                       4.8955685711129, # ISIBurstMeanChangeTest
                                  28.0, # SpikeRateStrongStimTest
                    7.3400000000001455, # AP1DelayMeanStrongStimTest
                                   0.0, # AP1DelaySDStrongStimTest
                    33.049999999999955, # AP2DelayMeanStrongStimTest
                                   0.0, # AP2DelaySDStrongStimTest
                    30.139999999999986, # Burst1ISIMeanStrongStimTest
                                   0.0, # Burst1ISISDStrongStimTest
         ]

        super(Model3TestCase, self).setUp()

class Model4TestCase(Druckmann2013BaseTestCase.BaseTest):
    def setUp(self):
        self.model_id = 'NMLCL000114'
        self.expected = [
                          [3.72355428], # AP12AmplitudeDropTest
                         [10.77160122], # AP1SSAmplitudeChangeTest
                            [68.09856], # AP1AmplitudeTest
                    0.5899999999999999, # AP1WidthHalfHeightTest
                                [6.75], # AP1WidthPeakToTroughTest
                          [-12.630314], # AP1RateOfChangePeakToTroughTest
                         [17.15605951], # AP1AHPDepthTest
                         [64.37500572], # AP2AmplitudeTest
                    0.6299999999999999, # AP2WidthHalfHeightTest
                                [7.47], # AP2WidthPeakToTroughTest
                        [-10.87835074], # AP2RateOfChangePeakToTroughTest
                         [16.88627428], # AP2AHPDepthTest
                          [-5.4678899], # AP12AmplitudeChangePercentTest
                     6.779661016949161, # AP12HalfWidthChangePercentTest
                        [-13.87109826], # AP12RateOfChangePeakToTroughPercentChangeTest
                         [-1.57253609], # AP12AHPDepthPercentChangeTest
                     347.1747619853342, # InputResistanceTest
                    18.320000000000164, # AP1DelayMeanTest
                                   0.0, # AP1DelaySDTest
                     66.58999999999992, # AP2DelayMeanTest
                                   0.0, # AP2DelaySDTest
                     59.16999999999996, # Burst1ISIMeanTest
                                   0.0, # Burst1ISISDTest
                                 -50.0, # InitialAccommodationMeanTest
                                 -50.0, # SSAccommodationMeanTest
                   -0.1316898440792246, # AccommodationRateToSSTest
                     -86.6444748708576, # AccommodationAtSSMeanTest
                    152.34253585703436, # AccommodationRateMeanAtSSTest
                     2.851214176443463, # ISICVTest
                    227.24500000000012, # ISIMedianTest
                      45.1626268904092, # ISIBurstMeanChangeTest
                                   9.0, # SpikeRateStrongStimTest
                    6.8799999999999955, # AP1DelayMeanStrongStimTest
                                   0.0, # AP1DelaySDStrongStimTest
                     32.34999999999991, # AP2DelayMeanStrongStimTest
                                   0.0, # AP2DelaySDStrongStimTest
                    27.045000000000016, # Burst1ISIMeanStrongStimTest
                                   0.0, # Burst1ISISDStrongStimTest
         ]

        super(Model4TestCase, self).setUp()



class Model5TestCase(Druckmann2013BaseTestCase.BaseTest):
    def setUp(self):
        self.model_id = 'NMLCL000760'
        self.expected = [
                          [0.17678428], # AP12AmplitudeDropTest
                          [0.22638091], # AP1SSAmplitudeChangeTest
                         [66.45599429], # AP1AmplitudeTest
                    0.7699999999999998, # AP1WidthHalfHeightTest
                                [7.16], # AP1WidthPeakToTroughTest
                        [-12.09735754], # AP1RateOfChangePeakToTroughTest
                         [20.16108571], # AP1AHPDepthTest
                            [66.27921], # AP2AmplitudeTest
                    0.7799999999999999, # AP2WidthHalfHeightTest
                                [7.17], # AP2WidthPeakToTroughTest
                        [-12.05909763], # AP2RateOfChangePeakToTroughTest
                            [20.18452], # AP2AHPDepthTest
                         [-0.26601706], # AP12AmplitudeChangePercentTest
                    1.2987012987013147, # AP12HalfWidthChangePercentTest
                         [-0.31626669], # AP12RateOfChangePeakToTroughPercentChangeTest
                          [0.11623523], # AP12AHPDepthPercentChangeTest
                    480.87320683703115, # InputResistanceTest
                     45.40000000000009, # AP1DelayMeanTest
                                   0.0, # AP1DelaySDTest
                    123.28999999999996, # AP2DelayMeanTest
                                   0.0, # AP2DelaySDTest
                     78.27499999999998, # Burst1ISIMeanTest
                                   0.0, # Burst1ISISDTest
                                 -20.0, # InitialAccommodationMeanTest
                                 -20.0, # SSAccommodationMeanTest
                 -0.021201038850903697, # AccommodationRateToSSTest
                   -17.448009894240503, # AccommodationAtSSMeanTest
                    1319.6849972649404, # AccommodationRateMeanAtSSTest
                      23.0999877545198, # ISICVTest
                     85.54500000000019, # ISIMedianTest
                    0.9885736294777384, # ISIBurstMeanChangeTest
                                  24.5, # SpikeRateStrongStimTest
                                  16.5, # AP1DelayMeanStrongStimTest
                                   0.0, # AP1DelaySDStrongStimTest
                    54.819999999999936, # AP2DelayMeanStrongStimTest
                                   0.0, # AP2DelaySDStrongStimTest
                    38.629999999999995, # Burst1ISIMeanStrongStimTest
                                   0.0, # Burst1ISISDStrongStimTest
         ]

        super(Model5TestCase, self).setUp()

class Model6TestCase(Druckmann2013BaseTestCase.BaseTest):
    def setUp(self):
        self.model_id = 'NMLCL001437'
        self.expected = [
                          [4.65326772], # AP12AmplitudeDropTest
                          [4.92507728], # AP1SSAmplitudeChangeTest
                         [51.13195733], # AP1AmplitudeTest
                                  0.34, # AP1WidthHalfHeightTest
                               [37.67], # AP1WidthPeakToTroughTest
                         [-2.26092797], # AP1RateOfChangePeakToTroughTest
                         [34.03719926], # AP1AHPDepthTest
                         [46.47868962], # AP2AmplitudeTest
                                  0.41, # AP2WidthHalfHeightTest
                                [14.7], # AP2WidthPeakToTroughTest
                           [-4.684297], # AP2RateOfChangePeakToTroughTest
                         [22.38047635], # AP2AHPDepthTest
                         [-9.10050771], # AP12AmplitudeChangePercentTest
                     20.58823529411763, # AP12HalfWidthChangePercentTest
                        [107.18470773], # AP12RateOfChangePeakToTroughPercentChangeTest
                        [-34.24700963], # AP12AHPDepthPercentChangeTest
                     323.6803558738357, # InputResistanceTest
                    15.279999999999973, # AP1DelayMeanTest
                                   0.0, # AP1DelaySDTest
                    1092.8300000000002, # AP2DelayMeanTest
                2.2737367544323206e-13, # AP2DelaySDTest
                               771.115, # Burst1ISIMeanTest
                1.1368683772161603e-13, # Burst1ISISDTest
                                   0.0, # InitialAccommodationMeanTest
                                -100.0, # SSAccommodationMeanTest
                                  None, # AccommodationRateToSSTest
                                  None, # AccommodationAtSSMeanTest
                                  None, # AccommodationRateMeanAtSSTest
                    2.5164064157162183, # ISICVTest
                     771.1150000000001, # ISIMedianTest
                    -56.87624704190064, # ISIBurstMeanChangeTest
                                   4.0, # SpikeRateStrongStimTest
                     4.110000000000014, # AP1DelayMeanStrongStimTest
                                   0.0, # AP1DelaySDStrongStimTest
                     729.6000000000003, # AP2DelayMeanStrongStimTest
                1.1368683772161603e-13, # AP2DelaySDStrongStimTest
                    469.71500000000003, # Burst1ISIMeanStrongStimTest
                 5.684341886080802e-14, # Burst1ISISDStrongStimTest
         ]

        super(Model6TestCase, self).setUp()

class Model7TestCase(Druckmann2013BaseTestCase.BaseTest):
    def setUp(self):
        self.model_id = 'NMLCL001633'
        self.expected = [
                          [0.63616282], # AP12AmplitudeDropTest
                          [6.42100497], # AP1SSAmplitudeChangeTest
                         [50.72791882], # AP1AmplitudeTest
                    0.5400000000000001, # AP1WidthHalfHeightTest
                                [1.91], # AP1WidthPeakToTroughTest
                         [-40.8440733], # AP1RateOfChangePeakToTroughTest
                         [27.28426118], # AP1AHPDepthTest
                           [50.091756], # AP2AmplitudeTest
                    0.5399999999999999, # AP2WidthHalfHeightTest
                                [1.93], # AP2WidthPeakToTroughTest
                        [-40.22087491], # AP2RateOfChangePeakToTroughTest
                         [27.53453257], # AP2AHPDepthTest
                         [-1.25406844], # AP12AmplitudeChangePercentTest
               -4.1119371282413194e-14, # AP12HalfWidthChangePercentTest
                         [-1.52579883], # AP12RateOfChangePeakToTroughPercentChangeTest
                          [0.91727386], # AP12AHPDepthPercentChangeTest
                    100.16096711671804, # InputResistanceTest
                     52.73000000000002, # AP1DelayMeanTest
                                   0.0, # AP1DelaySDTest
                     87.08999999999992, # AP2DelayMeanTest
                                   0.0, # AP2DelaySDTest
                     36.77499999999998, # Burst1ISIMeanTest
                                   0.0, # Burst1ISISDTest
                                 -25.0, # InitialAccommodationMeanTest
                                 -25.0, # SSAccommodationMeanTest
                 -0.020285127755734607, # AccommodationRateToSSTest
                    -52.74702876175882, # AccommodationAtSSMeanTest
                     499.3946365205547, # AccommodationRateMeanAtSSTest
                    5.3777211840747485, # ISICVTest
                     68.74000000000001, # ISIMedianTest
                    14.057043073341585, # ISIBurstMeanChangeTest
                                  45.0, # SpikeRateStrongStimTest
                    18.639999999999986, # AP1DelayMeanStrongStimTest
                                   0.0, # AP1DelaySDStrongStimTest
                     32.58999999999992, # AP2DelayMeanStrongStimTest
                                   0.0, # AP2DelaySDStrongStimTest
                    14.824999999999989, # Burst1ISIMeanStrongStimTest
                                   0.0, # Burst1ISISDStrongStimTest
         ]

        super(Model7TestCase, self).setUp()


class Model8TestCase(Druckmann2013BaseTestCase.BaseTest):
    def setUp(self):
        self.model_id = 'NMLCL000002'
        self.expected = [
                                  None, # AP12AmplitudeDropTest
                                  None, # AP1SSAmplitudeChangeTest
                         [42.89015636], # AP1AmplitudeTest
                                  0.17, # AP1WidthHalfHeightTest
                                [0.76], # AP1WidthPeakToTroughTest
                        [-55.23017105], # AP1RateOfChangePeakToTroughTest
                         [-0.91522636], # AP1AHPDepthTest
                                  None, # AP2AmplitudeTest
                                  None, # AP2WidthHalfHeightTest
                                  None, # AP2WidthPeakToTroughTest
                                  None, # AP2RateOfChangePeakToTroughTest
                                  None, # AP2AHPDepthTest
                                  None, # AP12AmplitudeChangePercentTest
                                  None, # AP12HalfWidthChangePercentTest
                                  None, # AP12RateOfChangePeakToTroughPercentChangeTest
                                  None, # AP12AHPDepthPercentChangeTest
                    4266.3210449606395, # InputResistanceTest
                    1.5900000000000318, # AP1DelayMeanTest
                                   0.0, # AP1DelaySDTest
                                  None, # AP2DelayMeanTest
                                  None, # AP2DelaySDTest
                                  None, # Burst1ISIMeanTest
                                  None, # Burst1ISISDTest
                                -100.0, # InitialAccommodationMeanTest
                                -100.0, # SSAccommodationMeanTest
                                  None, # AccommodationRateToSSTest
                                  None, # AccommodationAtSSMeanTest
                                  None, # AccommodationRateMeanAtSSTest
                                  None, # ISICVTest
                                  None, # ISIMedianTest
                                  None, # ISIBurstMeanChangeTest
                                   0.5, # SpikeRateStrongStimTest
                                  0.62, # AP1DelayMeanStrongStimTest
                                   0.0, # AP1DelaySDStrongStimTest
                                  None, # AP2DelayMeanStrongStimTest
                                  None, # AP2DelaySDStrongStimTest
                                  None, # Burst1ISIMeanStrongStimTest
                                  None, # Burst1ISISDStrongStimTest
         ]


        super(Model8TestCase, self).setUp()


class Model9TestCase(Druckmann2013BaseTestCase.BaseTest):
    def setUp(self):
        self.model_id = 'NMLCL001423'
        self.expected = [
                          [3.09824345], # AP12AmplitudeDropTest
                           [4.3887503], # AP1SSAmplitudeChangeTest
                           [78.188898], # AP1AmplitudeTest
                   0.18999999999999984, # AP1WidthHalfHeightTest
                                  [0.], # AP1WidthPeakToTroughTest
                    -8167.498499773694, # AP1RateOfChangePeakToTroughTest
                            [3.486087], # AP1AHPDepthTest
                         [75.09065454], # AP2AmplitudeTest
                   0.18000000000000005, # AP2WidthHalfHeightTest
                                [0.01], # AP2WidthPeakToTroughTest
                      [-8068.28266642], # AP2RateOfChangePeakToTroughTest
                          [5.59217212], # AP2AHPDepthTest
                         [-3.96251071], # AP12AmplitudeChangePercentTest
                    -5.263157894736734, # AP12HalfWidthChangePercentTest
                         [-1.21476402], # AP12RateOfChangePeakToTroughPercentChangeTest
                         [60.41401495], # AP12AHPDepthPercentChangeTest
                    1.1415498952202383, # InputResistanceTest
                      7.67999999999995, # AP1DelayMeanTest
                                   0.0, # AP1DelaySDTest
                     9.600000000000023, # AP2DelayMeanTest
                                   0.0, # AP2DelaySDTest
                     2.055000000000007, # Burst1ISIMeanTest
                                   0.0, # Burst1ISISDTest
                                   0.0, # InitialAccommodationMeanTest
                   -16.666666666666664, # SSAccommodationMeanTest
                  -0.11554816047328524, # AccommodationRateToSSTest
                    -101.2734178230418, # AccommodationAtSSMeanTest
                    27.555219323339742, # AccommodationRateMeanAtSSTest
                     0.623654001611502, # ISICVTest
                    2.6700000000005275, # ISIMedianTest
                    14.062499999992598, # ISIBurstMeanChangeTest
                                  65.5, # SpikeRateStrongStimTest
                    2.7200000000000273, # AP1DelayMeanStrongStimTest
                                   0.0, # AP1DelaySDStrongStimTest
                     4.350000000000023, # AP2DelayMeanStrongStimTest
                                   0.0, # AP2DelaySDStrongStimTest
                     1.589999999999975, # Burst1ISIMeanStrongStimTest
                                   0.0, # Burst1ISISDStrongStimTest
         ]


        super(Model9TestCase, self).setUp()


class Model10TestCase(Druckmann2013BaseTestCase.BaseTest):
    def setUp(self):
        self.model_id = 'NMLCL001650'
        self.expected = [
                          [0.02459799], # AP12AmplitudeDropTest
                         [-0.00849319], # AP1SSAmplitudeChangeTest
                         [54.98052908], # AP1AmplitudeTest
                     7.599999999999996, # AP1WidthHalfHeightTest
                                [0.02], # AP1WidthPeakToTroughTest
                      [-4249.18817799], # AP1RateOfChangePeakToTroughTest
                         [30.00323448], # AP1AHPDepthTest
                         [54.95593108], # AP2AmplitudeTest
                    7.6099999999999985, # AP2WidthHalfHeightTest
                                [0.02], # AP2WidthPeakToTroughTest
                      [-4247.85874166], # AP2RateOfChangePeakToTroughTest
                         [30.00124375], # AP2AHPDepthTest
                         [-0.04473946], # AP12AmplitudeChangePercentTest
                   0.13157894736845338, # AP12HalfWidthChangePercentTest
                         [-0.03128683], # AP12RateOfChangePeakToTroughPercentChangeTest
                         [-0.00663506], # AP12AHPDepthPercentChangeTest
                     64.81284333347436, # InputResistanceTest
                    195.23000000000002, # AP1DelayMeanTest
                                   0.0, # AP1DelaySDTest
                    346.30999999999995, # AP2DelayMeanTest
                                   0.0, # AP2DelaySDTest
                    151.10000000000005, # Burst1ISIMeanTest
                 2.842170943040401e-14, # Burst1ISISDTest
                                   0.0, # InitialAccommodationMeanTest
                                   0.0, # SSAccommodationMeanTest
                                   0.0, # AccommodationRateToSSTest
                 -0.030150625552840676, # AccommodationAtSSMeanTest
                     66.31099959954255, # AccommodationRateMeanAtSSTest
                    10961.079423403653, # ISICVTest
                    151.12000000000035, # ISIMedianTest
                  0.026476039184664422, # ISIBurstMeanChangeTest
                                  10.5, # SpikeRateStrongStimTest
                     98.75000000000023, # AP1DelayMeanStrongStimTest
                                   0.0, # AP1DelaySDStrongStimTest
                    191.13000000000034, # AP2DelayMeanStrongStimTest
                                   0.0, # AP2DelaySDStrongStimTest
                     92.44499999999994, # Burst1ISIMeanStrongStimTest
                                   0.0, # Burst1ISISDStrongStimTest
         ]

        super(Model10TestCase, self).setUp()


class Model11TestCase(Druckmann2013BaseTestCase.BaseTest):
    def setUp(self):
        self.model_id = 'NMLCL001139'
        self.expected = [
                            [-1.00554], # AP12AmplitudeDropTest
                         [-3.12254013], # AP1SSAmplitudeChangeTest
                            [97.41034], # AP1AmplitudeTest
                                  0.38, # AP1WidthHalfHeightTest
                                [0.81], # AP1WidthPeakToTroughTest
                       [-134.59654321], # AP1RateOfChangePeakToTroughTest
                            [11.61286], # AP1AHPDepthTest
                            [98.41588], # AP2AmplitudeTest
                   0.39000000000000007, # AP2WidthHalfHeightTest
                                [0.84], # AP2WidthPeakToTroughTest
                       [-128.73185714], # AP2RateOfChangePeakToTroughTest
                             [9.71888], # AP2AHPDepthTest
                          [1.03227234], # AP12AmplitudeChangePercentTest
                     2.631578947368438, # AP12HalfWidthChangePercentTest
                          [-4.3572338], # AP12RateOfChangePeakToTroughPercentChangeTest
                        [-16.30933293], # AP12AHPDepthPercentChangeTest
                    0.6310505934804042, # InputResistanceTest
                    3.0399999999999636, # AP1DelayMeanTest
                                   0.0, # AP1DelaySDTest
                     9.860000000000014, # AP2DelayMeanTest
                                   0.0, # AP2DelaySDTest
                    12.639999999999986, # Burst1ISIMeanTest
                                   0.0, # Burst1ISISDTest
                    -41.66666666666667, # InitialAccommodationMeanTest
                    -33.33333333333333, # SSAccommodationMeanTest
                  -0.05561580601206861, # AccommodationRateToSSTest
                    -79.69609768644382, # AccommodationAtSSMeanTest
                    158.63333295894842, # AccommodationRateMeanAtSSTest
                     4.294241000865026, # ISICVTest
                    55.335000000000036, # ISIMedianTest
                    170.67448680351595, # ISIBurstMeanChangeTest
                                 174.0, # SpikeRateStrongStimTest
                    1.4400000000000546, # AP1DelayMeanStrongStimTest
                                   0.0, # AP1DelaySDStrongStimTest
                    2.7900000000000773, # AP2DelayMeanStrongStimTest
                                   0.0, # AP2DelaySDStrongStimTest
                     3.605000000000018, # Burst1ISIMeanStrongStimTest
                                   0.0, # Burst1ISISDStrongStimTest
         ]


        super(Model11TestCase, self).setUp()

if __name__ == '__main__':
    unittest.main()

# tc = Model11TestCase()
# tc.setUp()
# tc.runTest()

# tc.print_predicted()
# tc.test_set[0]['test'].generate_prediction(tc.model)
#
# from matplotlib import pyplot as plt
#
# fig = plt.figure()
#
# plt.subplot(2, 3, 1)
# plt.plot(tc.model.vm.times, tc.model.vm)
# plt.xlim(0, tc.model.vm.times[-1].magnitude)
#
# plt.subplot(2, 3, 2)
# plt.plot(tc.model.vm.times, tc.model.vm)
# plt.xlim(0.990, 1.200)
#
# aps = tc.test_set[0]['test'].get_APs(tc.model)
#
# if len(aps) > 0:
#     plt.subplot(2, 3, 3)
#     plt.plot(tc.model.vm.times, tc.model.vm)
#     plt.xlim(aps[0].get_beginning()[1].rescale(sec), aps[0].get_beginning()[1].rescale(sec)+10*ms)
#
# if len(aps) > 1:
#     plt.subplot(2, 3, 4)
#     plt.plot(tc.model.vm.times, tc.model.vm)
#     plt.xlim(aps[1].get_beginning()[1].rescale(sec), aps[1].get_beginning()[1].rescale(sec) + 10 * ms)
#
# tc.test_set[16]['test'].generate_prediction(tc.model)
# plt.subplot(2, 3, 5)
# plt.plot(tc.model.vm.times, tc.model.vm)
# plt.xlim(0, tc.model.vm.times[-1].magnitude)
#
# tc.test_set[31]['test'].generate_prediction(tc.model)
# plt.subplot(2, 3, 6)
# plt.plot(tc.model.vm.times, tc.model.vm)
# plt.xlim(0.990, 1.200)
#
# plt.show()


