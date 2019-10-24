"""

This file contains a Druckman NU test static-neuromld model running object.
This is a hacking, re-writing and re-purposing of JB NU unit test of Druckman tests.
Which seemed to work really well with a static NU backend.

"""

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

def map_to_protocol():
    '''
    A method that takes nothing and returns
    a hard coded dictionary that keeps track of which protocol is used by each test.
    which is helpful on the data analysis end of this pipeline.
    '''
    standard = 1.5
    strong = 3.0
    easy_map = [
            {'AP12AmplitudeDropTest':standard},
            {'AP1SSAmplitudeChangeTest':standard},
            {'AP1AmplitudeTest':standard},
            {'AP1WidthHalfHeightTest':standard},
            {'AP1WidthPeakToTroughTest':standard},
            {'AP1RateOfChangePeakToTroughTest':standard},
            {'AP1AHPDepthTest':standard},
            {'AP2AmplitudeTest':standard},
            {'AP2WidthHalfHeightTest':standard},
            {'AP2WidthPeakToTroughTest':standard},
            {'AP2RateOfChangePeakToTroughTest':standard},
            {'AP2AHPDepthTest':standard},
            {'AP12AmplitudeChangePercentTest':standard},
            {'AP12HalfWidthChangePercentTest':standard},
            {'AP12RateOfChangePeakToTroughPercentChangeTest':standard},
            {'AP12AHPDepthPercentChangeTest':standard},
            {'InputResistanceTest':str('ir_currents')},
            {'AP1DelayMeanTest':standard},
            {'AP1DelaySDTest':standard},
            {'AP2DelayMeanTest':standard},
            {'AP2DelaySDTest':standard},
            {'Burst1ISIMeanTest':standard},
            {'Burst1ISISDTest':standard},
            {'InitialAccommodationMeanTest':standard},
            {'SSAccommodationMeanTest':standard},
            {'AccommodationRateToSSTest':standard},
            {'AccommodationAtSSMeanTest':standard},
            {'AccommodationRateMeanAtSSTest':standard},
            {'ISICVTest':standard},
            {'ISIMedianTest':standard},
            {'ISIBurstMeanChangeTest':standard},
            {'SpikeRateStrongStimTest':strong},
            {'AP1DelayMeanStrongStimTest':strong},
            {'AP1DelaySDStrongStimTest':strong},
            {'AP2DelayMeanStrongStimTest':strong},
            {'AP2DelaySDStrongStimTest':strong},
            {'Burst1ISIMeanStrongStimTest':strong},
            {'Burst1ISISDStrongStimTest':strong},
        ]
    test_prot_map = {}
    for easy in easy_map:
        test_prot_map.update(easy)
    test_prot_map = test_prot_map
    return test_prot_map

class DMTNMLO(object):
    '''
    An object for wrapping Druckman tests on instancable NeuroML-DB static models all in one neat package.
    '''
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
        self.test_prot_map = map_to_protocol()
        self.model_cache = model_cache

    def set_expected(self, expected_values):
        #
        assert (len(expected_values) == len(self.test_set)) or (len(expected_values) == len(self.test_set)+1)

        for i, v in enumerate(expected_values):
            self.test_set[i]['expected'] = v

    def test_setup(self,model_id,model_dict,model=None,ir_current_limited=False):
        '''
        Synopsis: Construct initialize and otherwise setup Druckman tests.
        if a model does not exist yet, but a desired NML-DB model id is known, use the model-id
        to quickly initialize a NML-DB model.

        If a model is actually passed instead, assume that model has known current_injection value
        attributes and use those.

        inputs: model_id, and a dictionary lookup table of models/model_ids

        '''


        if type(model) is type(None):
            self.model = model_dict[model_id]
            self.model_id = model_id
            if self.model_id not in self.predicted:
                self.predicted[self.model_id] = [None for i in range(38)] # There are 38 tests
            self.standard = self.model.nmldb_model.get_druckmann2013_standard_current()
            self.strong = self.model.nmldb_model.get_druckmann2013_strong_current()
            if not ir_current_limited==True:
                self.ir_currents = self.model.nmldb_model.get_druckmann2013_input_resistance_currents()

        #model = self.__class__.model_cache[self.model_id]
        else:
            self.model = model

            self.standard = model.druckmann2013_standard_current
            self.strong = model.druckmann2013_strong_current
            if not ir_current_limited==True:
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
            {'test': Burst1ISISDStrongStimTest(self.strong), 'units': pq.ms, 'expected': None}

            ]
        if ir_current_limited==True:
            pass
        else:
            self.test_set.append({'test': InputResistanceTest(injection_currents=self.ir_currents), 'units': pq.Quantity(1,'MOhm'), 'expected': None})
        if not hasattr(self, "expected"):
            self.expected = [0.0 for i in range(len(self.test_set))]
        self.set_expected(self.expected)




    def test_setup_subset(self,model_id,model_dict,model=None,ir_current_limited=False):
        '''
        Synopsis: Construct initialize and otherwise setup Druckman tests.
        if a model does not exist yet, but a desired NML-DB model id is known, use the model-id
        to quickly initialize a NML-DB model.

        If a model is actually passed instead, assume that model has known current_injection value
        attributes and use those.

        inputs: model_id, and a dictionary lookup table of models/model_ids

        '''


        if type(model) is type(None):
            self.model = model_dict[model_id]
            self.model_id = model_id
            if self.model_id not in self.predicted:
                self.predicted[self.model_id] = [None for i in range(38)] # There are 38 tests
            self.standard = self.model.nmldb_model.get_druckmann2013_standard_current()
            self.strong = self.model.nmldb_model.get_druckmann2013_strong_current()
            if not ir_current_limited==True:
                self.ir_currents = self.model.nmldb_model.get_druckmann2013_input_resistance_currents()

        #model = self.__class__.model_cache[self.model_id]
        else:
            self.model = model

            self.standard = model.druckmann2013_standard_current
            self.strong = model.druckmann2013_strong_current
            if not ir_current_limited==True:
                self.ir_currents = model.druckmann2013_input_resistance_currents
            self.test_set = [
            {'test': AP1AmplitudeTest(self.standard), 'units': pq.mV, 'expected': None},
            {'test': AP1WidthHalfHeightTest(self.standard), 'units': pq.ms, 'expected': None},
            ]
        if ir_current_limited==True:
            pass
        else:
            self.test_set.append({'test': InputResistanceTest(injection_currents=self.ir_currents), 'units': pq.Quantity(1,'MOhm'), 'expected': None})
        if not hasattr(self, "expected"):
            self.expected = [0.0 for i in range(len(self.test_set))]
        self.set_expected(self.expected)

        #import pdb; pdb.set_trace()

    '''
    Depreciated

    def get_model(self):

        if self.model_id not in self.__class__.model_cache:
            #print('Model ' + self.model_id + ' not in cache. Downloading waveforms...')

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
        # Use this function to re-pickle models after tests have
        # run (and waveforms have been downloaded from NeuroML-DB.org)
        # :return: Nothing, models are saved in a pickle file

        for model in cls.model_cache.values():
            # Clear AnalogSignal versions (to reduce file size) and pickle the model (to speed up unit tests)
            model.vm = None
            model.nmldb_model.waveform_signals = {}
            model.nmldb_model.steady_state_waveform = None

        #import pickle
        with open(cls.pickle_file, 'w') as fp:
            pickle.dump(cls.model_cache, fp)
    '''
    def run_test(self, index):
        test_class = self.test_set[index]['test']
        expected = self.test_set[index]['expected']
        units = self.test_set[index]['units']

        if units is None:
            units = pq.dimensionless
        try:
            predicted = test_class.generate_prediction(self.model)['mean']

        except:
            predicted = None

        return (test_class,predicted)

    def runTest(self):
        predictions = {}
        for i, t in enumerate(self.test_set):
           (tclass,prediction) = self.run_test(i)
           #try:
           prot =  self.test_prot_map[tclass.name]
           #except:
           #    print(self.test_prot_map)
           #    print(tclass.name)
           #    prot = str('figure out protocol for Drop in AP amplitude from 1st to 2nd AP')
           
           predictions[tclass.name] = prediction
           predictions[str(prot)] = prediction

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

    #@classmethod
    def print_predicted(cls):

        for model_id in cls.predicted.keys():
            print('Predicted values for '+model_id+': [')
            for i, p in enumerate(cls.predicted[model_id]):
                if p['predicted'] is not None:
                    print('             ' + str((p['predicted'] * dimensionless).magnitude).rjust(25) + ', # ' + p['test'])
                else:
                    print('             '+'None'.rjust(25)+', # ' + p['test'])

            print('         ]')
