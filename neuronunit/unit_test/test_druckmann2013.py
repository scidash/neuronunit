"""Tests of the 38 Druckmann 2013 test classes"""

import unittest
import pickle
import quantities as pq
from neuronunit.tests.druckmann2013 import *

class Druckmann2013TestCase(unittest.TestCase):

    try:
        # Use cached pickled model for testing (faster, no external dependency)

        import sys, os
        pickle_file = 'nmldb-model.pkl'

        # Handle being executed from test.sh
        if not os.path.exists(pickle_file):
            pickle_file = os.path.join('neuronunit','unit_test',pickle_file)

        with open(pickle_file, 'rb') as f:
            model = pickle.load(f)

        #print('Succesfully unpickled NML-DB model for use in Druckmann 2013 tests...')

    except:
        # Otherwise, download waveforms from NeuroML-DB.org (slower, depends on server connection)
        print('Failed to unpickle model, connecting to NeuroML-DB.org...')

        from neuronunit.neuromldb import NeuroMLDBStaticModel
        model = NeuroMLDBStaticModel("NMLCL001129")

    def setUp(self):
            self.model = self.__class__.model
            self.standard = self.model.nmldb_model.get_druckmann2013_standard_current()
            self.strong = self.model.nmldb_model.get_druckmann2013_strong_current()
            self.ir_currents = self.model.nmldb_model.get_druckmann2013_input_resistance_currents()

    def pickle_model(self):
        '''
        Use this function to re-pickle the model after tests have
        run (and waveforms have been downloaded from NeuroML-DB.org)
        :return: Nothing, model is saved in a pickle file
        '''

        # Clear AnalogSignal versions (to reduce file size) and pickle the model (to speed up unit tests)
        self.model.nmldb_model.waveform_signals = {}

        import pickle
        with open('nmldb-model.pkl', 'w') as fp:
            pickle.dump(self.model, fp)

    def test_0(self):
        self.assertTrue(AP12AmplitudeDropTest(self.standard).generate_prediction(self.model)['mean'] - 0.5299825 * pq.mV <= 0.001 * pq.mV)

    def test_1(self):
        self.assertTrue(AP1SSAmplitudeChangeTest(self.standard).generate_prediction(self.model)['mean'] - 0.55190155 * pq.mV <= 0.001 * pq.mV)

    def test_2(self):
        self.assertTrue(AP1AmplitudeTest(self.standard).generate_prediction(self.model)['mean'] - 94.9859825 * pq.mV <= 0.001 * pq.mV)

    def test_3(self):
        self.assertTrue(AP1WidthHalfHeightTest(self.standard).generate_prediction(self.model)['mean'] - 1.42 * pq.ms <= 0.001 * pq.ms)

    def test_4(self):
        self.assertTrue(AP1WidthPeakToTroughTest(self.standard).generate_prediction(self.model)['mean'] - 3.56 * pq.ms <= 0.001 * pq.ms)

    def test_5(self):
        self.assertTrue(AP1RateOfChangePeakToTroughTest(self.standard).generate_prediction(self.model)['mean'] - -33.68388413 * pq.mV/pq.ms <= 0.001 * pq.mV/pq.ms)

    def test_6(self):
        self.assertTrue(AP1AHPDepthTest(self.standard).generate_prediction(self.model)['mean'] - 24.928645 * pq.mV <= 0.001 * pq.mV)

    def test_7(self):
        self.assertTrue(AP2AmplitudeTest(self.standard).generate_prediction(self.model)['mean'] - 94.456 * pq.mV <= 0.001 * pq.mV)

    def test_8(self):
        self.assertTrue(AP2WidthHalfHeightTest(self.standard).generate_prediction(self.model)['mean'] - 1.42 * pq.ms <= 0.001 * pq.ms)

    def test_9(self):
        self.assertTrue(AP2WidthPeakToTroughTest(self.standard).generate_prediction(self.model)['mean'] - 3.58 * pq.ms <= 0.001 * pq.ms)

    def test_10(self):
        self.assertTrue(AP2RateOfChangePeakToTroughTest(self.standard).generate_prediction(self.model)['mean'] - -33.40811662 * pq.mV/pq.ms <= 0.001 * pq.mV/pq.ms)

    def test_11(self):
        self.assertTrue(AP2AHPDepthTest(self.standard).generate_prediction(self.model)['mean'] - 25.1450575 * pq.mV <= 0.001 * pq.mV)

    def test_12(self):
        self.assertTrue(AP12AmplitudeChangePercentTest(self.standard).generate_prediction(self.model)['mean'] - -0.55795864 <= 0.001)

    def test_13(self):
        self.assertTrue(AP12HalfWidthChangePercentTest(self.standard).generate_prediction(self.model)['mean'] <= 0.001)

    def test_14(self):
        self.assertTrue(AP12RateOfChangePeakToTroughPercentChangeTest(self.standard).generate_prediction(self.model)['mean'] - -0.81869273 <= 0.001)

    def test_15(self):
        self.assertTrue(AP12AHPDepthPercentChangeTest(self.standard).generate_prediction(self.model)['mean'] - 0.86812781 <= 0.001)

    def test_16(self):
        self.assertTrue(InputResistanceTest(injection_currents=self.ir_currents).generate_prediction(self.model)['mean'] - 16.61969092 * pq.Quantity(1,'MOhm') <= 0.001 * pq.Quantity(1,'MOhm'))

    def test_17(self):
        self.assertTrue(AP1DelayMeanTest(self.standard).generate_prediction(self.model)['mean'] - 8.58 * pq.ms <= 0.001 * pq.ms)

    def test_18(self):
        self.assertTrue(AP1DelaySDTest(self.standard).generate_prediction(self.model)['mean'] <= 0.001 * pq.ms)

    def test_19(self):
        self.assertTrue(AP2DelayMeanTest(self.standard).generate_prediction(self.model)['mean'] - 104.03 * pq.ms <= 0.001 * pq.ms)

    def test_20(self):
        self.assertTrue(AP2DelaySDTest(self.standard).generate_prediction(self.model)['mean'] <= 0.001 * pq.ms)

    def test_21(self):
        self.assertTrue(Burst1ISIMeanTest(self.standard).generate_prediction(self.model)['mean'] - 96.815 * pq.ms <= 0.001 * pq.ms)

    def test_22(self):
        self.assertTrue(Burst1ISISDTest(self.standard).generate_prediction(self.model)['mean'] <= 0.001 * pq.ms)

    def test_23(self):
        self.assertTrue(InitialAccommodationMeanTest(self.standard).generate_prediction(self.model)['mean'] - -20.0 <= 0.001)

    def test_24(self):
        self.assertTrue(SSAccommodationMeanTest(self.standard).generate_prediction(self.model)['mean'] - -20.0 <= 0.001)

    def test_25(self):
        self.assertTrue(AccommodationRateToSSTest(self.standard).generate_prediction(self.model)['mean'] - -0.19225223 / pq.ms <= 0.001 / pq.ms)

    def test_26(self):
        self.assertTrue(AccommodationAtSSMeanTest(self.standard).generate_prediction(self.model)['mean'] - -2.7839110417547794 <= 0.001)

    def test_27(self):
        self.assertTrue(AccommodationRateMeanAtSSTest(self.standard).generate_prediction(self.model)['mean'] - 14.67334883 * pq.ms <= 0.001 * pq.ms)

    def test_28(self):
        self.assertTrue(ISICVTest(self.standard).generate_prediction(self.model)['mean'] - 164.5639062227548 <= 0.001)

    def test_29(self):
        self.assertTrue(ISIMedianTest(self.standard).generate_prediction(self.model)['mean'] - 98.18 * pq.ms <= 0.001 * pq.ms)

    def test_30(self):
        self.assertTrue(ISIBurstMeanChangeTest(self.standard).generate_prediction(self.model)['mean'] - 2.8601361969619004 <= 0.001)

    def test_31(self):
        self.assertTrue(SpikeRateStrongStimTest(self.strong).generate_prediction(self.model)['mean'] - 7.5 * pq.Hz <= 0.001 * pq.Hz)

    def test_32(self):
        self.assertTrue(AP1DelayMeanStrongStimTest(self.strong).generate_prediction(self.model)['mean'] - 5 * pq.ms <= 0.001 * pq.ms)

    def test_33(self):
        self.assertTrue(AP1DelaySDStrongStimTest(self.strong).generate_prediction(self.model)['mean'] <= 0.001 * pq.ms)

    def test_34(self):
        self.assertTrue(AP2DelayMeanStrongStimTest(self.strong).generate_prediction(self.model)['mean'] - 69.34 * pq.ms <= 0.001 * pq.ms)

    def test_35(self):
        self.assertTrue(AP2DelaySDStrongStimTest(self.strong).generate_prediction(self.model)['mean'] <= 0.001 * pq.ms)

    def test_36(self):
        self.assertTrue(Burst1ISIMeanStrongStimTest(self.strong).generate_prediction(self.model)['mean'] - 67.115 * pq.ms <= 0.001 * pq.ms)

    def test_37(self):
        self.assertTrue(Burst1ISISDStrongStimTest(self.strong).generate_prediction(self.model)['mean'] <= 0.001 * pq.ms)


if __name__ == '__main__':
    unittest.main()
