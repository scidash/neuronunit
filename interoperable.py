"""Tests of the 38 Druckmann 2013 test classes"""

import unittest
import pickle
import quantities as pq
from neuronunit.tests.druckman2013 import *
from neuronunit.neuromldb import NeuroMLDBStaticModel
from numpy import array
from quantities import *
import pickle
import glob
import sys, os

#class Druckmann2013BaseTestCase:
class Interoperabe(object):
    def __init__(self):

        self.predicted = {}
        pickle_file = 'nmldb-model-cache.pkl'

        if not os.path.exists(pickle_file):
            pickle_file = os.path.join('neuronunit','unit_test',pickle_file)

        try:

            with open(pickle_file, 'rb') as f:
                if sys.version_info[0] >= 3:
                    model_cache = pickle.load(f, encoding='Latin-1')
                else:
                    model_cache = pickle.load(f)
        except:
            per_file_cache = glob.glob('for_dm_tests_*.p')
            model_cache = {}
            for model_file in per_file_cache:
                with open(model_file, 'rb') as f:
                    key = model_file.split('.')[0]
                    key = key.split('for_dm_tests_')[1]
                    model_cache[key] = pickle.load(f)

        self.model_cache = model_cache

    def set_expected(self, expected_values):
        assert len(expected_values) == len(self.test_set)

        for i, v in enumerate(expected_values):
            self.test_set[i]['expected'] = v

    def test_setup(self,model_id,model_dict,model=None):
        if not hasattr(self, "expected"):
            self.expected = [0.0 for i in range(38)]


        if type(model) is type(None):
            self.model = model_dict[model_id]
            self.model_id = model_id
            if self.model_id not in self.predicted:
                self.predicted[self.model_id] = [None for i in range(38)] # There are 38 tests
            self.standard = self.model.nmldb_model.get_druckmann2013_standard_current()
            self.strong = self.model.nmldb_model.get_druckmann2013_strong_current()
            self.ir_currents = self.model.nmldb_model.get_druckmann2013_input_resistance_currents()

        #model = self.__class__.model_cache[self.model_id]
        else:
            self.model = model

            self.standard = model.druckmann2013_standard_current
            self.strong = model.druckmann2013_strong_current
            self.ir_currents = model.druckmann2013_input_resistance_currents
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
        #import pdb; pdb.set_trace()

    def get_model(self):

        if self.model_id not in self.__class__.model_cache:
            print('Model ' + self.model_id + ' not in cache. Downloading waveforms...')

            self.__class__.model_cache[self.model_id] = NeuroMLDBStaticModel(self.model_id)
            model = self.__class__.model_cache[self.model_id]
            fname = str('for_dm_tests_')+str(self.model_id)+str('.p')
            with open(str(fname), 'wb') as fp: pickle.dump(model, fp)

        if self.model_id not in self.predicted:
            self.predicted[self.model_id] = [None for i in range(38)] # There are 38 tests

        model = self.__class__.model_cache[self.model_id]
        return model

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

        #import pickle
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
        return predicted


        return predicted
    def runTest(self):
        predictions = []
        for i, t in enumerate(self.test_set):
            predictions.append(self.run_test(i))
            print(predictions[-1])
            #import pdb; pdb.set_trace()
            #try:
            #
            #except:
            #    pass
        return predictions

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
        #import pdb; pdb.set_trace()

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
'''
if __name__ == '__main__':
    a = Interoperabe()
    #a.model_cache.keys()
    for k,v in a.model_cache.items():
        print(k,v)
        a.test_setup(k,a.model_cache)
        a.runTest()
'''
    #import pdb;
    #pdb.set_trace()

    #import pdb; pdb.set_trace()
    # a = Interoperabe()
    #a.test_setup(model_id)
    #a.setUp()
    #a.getModel()

    #model = a.getModel()
    #a.test_set

    #unittest.main()

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
