"""Tests of NeuronUnit test classes"""

import unittest
import os,sys
old = str(os.getcwd())
this_nu = os.path.join(str(os.getcwd()),'../../')
sys.path.insert(0,this_nu)


class TestsTestCase(object):
    """Abstract base class for testing tests"""


    def setUp(self):
        self.dtcpop = None
        self.pop = None
        self.pf = None
        self.logbook = None
        self.params = {}
        self.prediction = None
        from neuronunit.models.reduced import ReducedModel
        #from neuronunit.model_tests import ReducedModelTestCase
        #path = ReducedModelTestCase().path
        path = os.getcwd()+str('/NeuroML2/LEMS_2007One.xml')
        print(path)
        #self.model = ReducedModel(path, backend='jNeuroML')
        self.model = ReducedModel(path, backend='NEURON')


    def get_observation(self, cls):
        print(cls.__name__)
        neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
        return cls.neuroelectro_summary_observation(neuron)


    def try_hard_coded0(self):
        params0 = {'C': '0.000107322241995',
        'a': '0.177922330376',
        'b': '-5e-09',
        'c': '-59.5280130394',
        'd': '0.153178745992',
        'k': '0.000879131572692',
        'v0': '-73.3255584633',
        'vpeak': '34.5214177196',
        'vr': '-71.0211905343',
        'vt': '-46.6016774842'}
        #rheobase = {'value': array(131.34765625) * pA}
        return params0



    def try_hard_coded1(self):
        params1 = {'C': '0.000106983591242',
        'a': '0.480856799107',
        'b': '-5e-09',
        'c': '-57.4022276619',
        'd': '0.0818117582621',
        'k': '0.00114004749537',
        'v0': '-58.4899756601',
        'vpeak': '36.6769758895',
        'vr': '-63.4080852004',
        'vt': '-44.1074682812'}
        #rheobase = {'value': array(106.4453125) * pA}131.34765625
        return params1




    def difference(self,observation,prediction): # v is a tesst
        import quantities as pq
        import numpy as np

        # The trick is.
        # prediction always has value. but observation 7 out of 8 times has mean.

        if 'value' in prediction.keys():
            unit_predictions = prediction['value']
            if 'mean' in observation.keys():
                unit_observations = observation['mean']
            elif 'value' in observation.keys():
                unit_observations = observation['value']

        if 'mean' in prediction.keys():
            unit_predictions = prediction['mean']
            if 'mean' in observation.keys():
                unit_observations = observation['mean']
            elif 'value' in observation.keys():
                unit_observations = observation['value']


        to_r_s = unit_observations.units
        unit_predictions = unit_predictions.rescale(to_r_s)
        #unit_observations = unit_observations.rescale(to_r_s)
        unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )

        ##
        # Repurposed from from sciunit/sciunit/scores.py
        # line 156
        ##
        assert type(observation) in [dict,float,int,pq.Quantity]
        assert type(prediction) in [dict,float,int,pq.Quantity]
        ratio = unit_predictions / unit_observations
        unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )
        return unit_delta, ratio

    def bar_char_out(self,score,test,par):
        import pandas as pd
        import numpy as np
        unit_delta, ratio = self.difference(score.observation,score.prediction)
        columns1 = []
        if 'mean' in score.observation.keys():
            unit_observations = score.observation['mean']
            to_r_s = unit_observations.units
            columns1.append(str(score.observation['mean'].units))
        if 'value' in score.observation.keys():
            unit_observations = score.observation['value']
            to_r_s = unit_observations.units
            columns1.append(str(score.observation['value'].units))
        if 'mean' in score.prediction.keys():
            unit_predictions = score.prediction['mean']
            unit_predictions = unit_predictions.rescale(to_r_s)
            columns1.append(str(unit_predictions.units))
        if 'value' in score.prediction.keys():
            unit_predictions = score.prediction['value']
            unit_predictions = unit_predictions.rescale(to_r_s)
            columns1.append(str(unit_predictions.units))


        annotations = ['observation'+str(columns1[0]),'prediction'+str(columns1[1]),'difference','ratio','score.sort_key']
        five = [ unit_observations,unit_predictions,unit_delta,ratio,score.sort_key]
        stacked = np.column_stack(np.array(five))

        df = pd.DataFrame(np.array(stacked))
        df = pd.DataFrame(stacked,columns = annotations)
        df = df.transpose()
        html = df.to_html()
        html_file = open("tests_agreement_table_{0}.html".format(str(test)),"w")
        html_file.write(html)
        html_file.close()
        import os
        #os.system('sudo /opt/conda/bin/pip install cufflinks')
        import cufflinks as cf
        import plotly.tools as tls
        import plotly.plotly as py

        #tls.embed('https://plot.ly/~cufflinks/8')
        #py.sign_in('RussellJarvis','FoyVbw7Ry3u4N2kCY4LE')
        #df.iplot(kind='bar', barmode='stack', yTitle=str(test)+str(' ')+str(score.sort_key), title='tests_agreement_table_{0}{1}'.format(test,score.sort_key), filename='grouped-bar-chart-{0}'.format(str(par['C'])))
        return df, html

    def run_test(self, cls):
        observation = self.get_observation(cls)
        test = cls(observation=observation)
        #attrs = pickle.load(open('opt_run_data.p','rb'))
        #from neuronunit.optimization import nsga_parallel
        #self.dtcpop = dtcpop
        #self.pop = pop
        #self.pf = pf
        #for d in dtcpop:
            #print(d,d.attrs)
        params0 = self.try_hard_coded0()
        params1 = self.try_hard_coded1()
        params = [params0,params1]

        #for par in params:
        self.model.set_attrs(params0)

        score = test.judge(self.model,stop_on_error = True, deep_error = True)
        #df, html = self.bar_char_out(score,str(test),params0)

        #score.summarize()
        return score

class TestsPassiveTestCase(TestsTestCase, unittest.TestCase):
    """Test passive validation tests"""
    #def test_0optimizer(self):


    def test_1inputresistance(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.passive import InputResistanceTest as T
        score = self.run_test(T)
        #self.assertTrue(-0.6 < score < -0.5)

    def test_2restingpotential(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.passive import RestingPotentialTest as T
        score = self.run_test(T)
        #self.assertTrue(1.2 < score < 1.3)

    def test_3capacitance(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.passive import CapacitanceTest as T
        score = self.run_test(T)
        #self.assertTrue(-0.15 < score < -0.05)

    def test_4timeconstant(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.passive import TimeConstantTest as T
        score = self.run_test(T)
        #self.assertTrue(-1.45 < score < -1.35)


class TestsWaveformTestCase(TestsTestCase, unittest.TestCase):
    """Test passive validation tests"""
    @unittest.skip("This times out")
    def test_0rheobase_parallel(self):
        import os
        os.system('ipcluster start -n 8 --profile=default & sleep 55;ß')

        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.fi import RheobaseTest as T
        score = self.run_test(T)
        self.prediction = score.prediction
        #super(TestsWaveformTestCase,self).prediction = score.prediction
        #self.prediction = score.prediction
        print(self.prediction, 'is prediction being updated properly?')

        self.assertTrue( score.prediction['value'] == 106.4453125 or score.prediction['value'] ==131.34765625)

    def test_01rheobase_serial(self):
        from neuronunit.optimization import data_transport_container

        from neuronunit.tests.fi import RheobaseTest as T
        score = self.run_test(T)
        super(TestsWaveformTestCase,self).prediction = score.prediction
        #self.prediction = score.prediction
        self.assertTrue( int(score.prediction['value']) == int(106) or int(score.prediction['value']) == int(131))


    def update_amplitude(self,test):
        rheobase = self.prediction['value']#first find a value for rheobase
        test.params['injected_square_current']['amplitude'] = rheobase * 1.01



    def test_5ap_width(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.waveform import InjectedCurrentAPWidthTest as T
        print(self.prediction, 'is prediction being updated properly?')
        self.update_amplitude(T)
        score = self.run_test(T)
        #self.assertTrue(-0.6 < score < -0.5)

    def test_6ap_amplitude(self):
        #from neuronunit.optimization import data_transport_container
        from neuronunit.tests.waveform import InjectedCurrentAPAmplitudeTest as T
        print(self.prediction, 'is prediction being updated properly?')

        self.update_amplitude(T)
        print(self.prediction, 'is prediction being updated properly?')

        score = self.run_test(T)
        #self.assertTrue(-1.7 < score < -1.6)

    def test_7ap_threshold(self):
        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.waveform import InjectedCurrentAPThresholdTest as T
        print(self.prediction, 'is prediction being updated properly?')

        self.update_amplitude(T)
        print(self.prediction, 'is prediction being updated properly?')

        score = self.run_test(T)
        #self.assertTrue(2.25 < score < 2.35)


class TestsFITestCase(TestsTestCase, unittest.TestCase):
    """Test F/I validation tests"""


    #@unittest.skip("This test takes a long time")
    def test_01rheobase_serial(self):
        from neuronunit.optimization import data_transport_container

        from neuronunit.tests.fi import RheobaseTest as T
        score = self.run_test(T)
        self.prediction = score.prediction
        self.assertTrue( int(score.prediction['value']) == int(106) or int(score.prediction['value']) == int(131))

        #self.assertTrue(0.2 < score < 0.3)

    #@unittest.skip("This test takes a long time")
    @unittest.skip("This test is not yet implemented")
    def test_02rheobase_parallel(self):
        import os
        os.system('ipcluster start -n 8 --profile=default & sleep 45 ')

        #from neuronunit.optimization import data_transport_container

        from neuronunit.tests.fi import RheobaseTestP as T
        score = self.run_test(T)
        #score.prediction
        self.prediction = score.prediction
        self.assertTrue( int(score.prediction['value']) == int(106) or int(score.prediction['value']) == int(131))




if __name__ == '__main__':
    unittest.main()
