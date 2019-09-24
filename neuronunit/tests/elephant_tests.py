"""

This file contains a Druckman NU test static-neuromld model running object.
This is a hacking, re-writing and re-purposing of JB NU unit test of Druckman tests.
Which seemed to work really well with a static NU backend.

"""

import unittest
import pickle
import quantities as pq
#from neuronunit.tests.druckman2013 import *
#from neuronunit.neuromldb import NeuroMLDBStaticModel
from numpy import array
from quantities import *
import pickle
import glob
import sys, os
from collections import Iterable, OrderedDict
from neuronunit.tests.base import AMPL, DELAY, DURATION
from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
from neuronunit.tests import InjectedCurrentAPThresholdTest, APThresholdTest
from neuronunit.tests import InjectedCurrentAPAmplitudeTest, APAmplitudeTest, APWidthTest, RestingPotentialTest, CapacitanceTest
from neuronunit.tests import TimeConstantTest, InputResistanceTest, InjectedCurrentAPWidthTest

def format_test(dtc):
    import copy
    # pre format the current injection dictionary based on pre computed
    # rheobase values of current injection.
    # This is much like the hooked method from the old get neab file.
    dtc.vtest = {}

    dtc.tests = copy.copy(dtc.tests)

    if type(dtc.tests) is type({}):
        tests = [key for key in dtc.tests.values()]
        dtc.tests = switch_logic(tests)
    else:
        dtc.tests = switch_logic(dtc.tests)



    for k,v in enumerate(dtc.tests):
        dtc.vtest[k] = {}
        #for t in tests:
        if hasattr(v,'passive'):#['protocol']:
            if v.passive == False and v.active == True:
                keyed = dtc.vtest[k]
                dtc.vtest[k] = active_values(keyed,dtc.rheobase)
                #print(dtc.vtest[k]['injected_square_current']['delay']+dtc.vtest[k]['injected_square_current']['duration'])
            elif v.passive == True and v.active == False:
                keyed = dtc.vtest[k]
                dtc.vtest[k] = passive_values(keyed)
        if v.name in str('RestingPotentialTest'):

            #keyed['injected_square_current']['amplitude'] = -10*pq.pA
            dtc.vtest[k]['injected_square_current']['amplitude'] = 0.0*pq.pA
            keyed = dtc.vtest[k]
            #print(keyed)
    return dtc


def switch_logic(tests):
    # move this logic into sciunit tests
    '''
    Hopefuly depreciated by future NU debugging.
    '''
    if not isinstance(tests,Iterable):
        if str('RheobaseTest') == tests.name or str('RheobaseTestP') == tests.name:
            active = True
            passive = False
    else:
        for t in tests:
            try:
                t.passive = None
                t.active = None
            except:
                import pdb
                pdb.set_trace()
            active = False
            passive = False

            if str('RheobaseTest') == t.name:
                active = True
                passive = False
            elif str('RheobaseTestP') == t.name:
                active = True
                passive = False
            elif str('InjectedCurrentAPWidthTest') == t.name:
                active = True
                passive = False
            elif str('InjectedCurrentAPAmplitudeTest') == t.name:
                active = True
                passive = False
            elif str('InjectedCurrentAPThresholdTest') == t.name:
                active = True
                passive = False
            elif str('RestingPotentialTest') == t.name:
                passive = True
                active = False
            elif str('InputResistanceTest') == t.name:
                passive = True
                active = False
            elif str('TimeConstantTest') == t.name:
                passive = True
                active = False
            elif str('CapacitanceTest') == t.name:
                passive = True
                active = False
            t.passive = passive
            t.active = active
    return tests

def active_values(keyed,rheobase,square = None):
    keyed['injected_square_current'] = {}
    if square == None:
        if type(rheobase) is type({str('k'):str('v')}):
            keyed['injected_square_current']['amplitude'] = float(rheobase['value'])*pq.pA
        else:
            keyed['injected_square_current']['amplitude'] = rheobase

        keyed['injected_square_current']['delay'] = DELAY
        keyed['injected_square_current']['duration'] = DURATION

    else:
        keyed['injected_square_current']['duration'] = square['Time_End'] - square['Time_Start']
        keyed['injected_square_current']['delay'] = square['Time_Start']
        keyed['injected_square_current']['amplitude'] = square['prediction']#value'])*pq.pA

    return keyed

def passive_values(keyed):
    PASSIVE_DURATION = 500.0*pq.ms
    PASSIVE_DELAY = 200.0*pq.ms
    keyed['injected_square_current'] = {}
    keyed['injected_square_current']['delay']= PASSIVE_DELAY
    keyed['injected_square_current']['duration'] = PASSIVE_DURATION
    keyed['injected_square_current']['amplitude'] = -10*pq.pA
    return keyed



def map_to_protocol():
    '''
    A method that takes nothing and returns
    a hard coded dictionary that keeps track of which protocol is used by each test.
    which is helpful on the data analysis end of this pipeline.
    '''
    standard = 1.0

    easy_map = [
            {'InjectedCurrentAPThresholdTest':standard},
            {'APThresholdTest':standard},
            {'InjectedCurrentAPAmplitudeTest':standard},
            {'APAmplitudeTest':standard},
            {'InjectedCurrentAPWidthTest':standard},
            {'APWidthTest':standard},
            {'RestingPotentialTest':standard},
            {'CapacitanceTest':standard},
            {'TimeConstantTest':standard},
            {'InputResistanceTest':standard}
        ]
    test_prot_map = {}
    for easy in easy_map:
        test_prot_map.update(easy)
    test_prot_map = test_prot_map
    return test_prot_map

def test_setup(self,model,protocol_container):#,model_id,model_dict,model=None,ir_current_limited=False):
    '''
    Synopsis: Construct initialize and otherwise setup Druckman tests.
    if a model does not exist yet, but a desired NML-DB model id is known, use the model-id
    to quickly initialize a NML-DB model.

    If a model is actually passed instead, assume that model has known current_injection value
    attributes and use those.

    inputs: model_id, and a dictionary lookup table of models/model_ids

    '''
    standard = model.rheobase
    params_dic = {t.name:t.params for t in protocol_container}
    obs_dic = {t.name:t.observation for t in protocol_container }
    name_to_test = {t.name:t for t in protocol_container }

    self.test_set = []

    if 'InjectedCurrentAPThresholdTest' in params_dic.keys():
        print(params_dic.keys())
        self.test_set.append(InjectedCurrentAPThresholdTest(obs_dic['InjectedCurrentAPThresholdTest'], \
            params = params_dic['InjectedCurrentAPThresholdTest']))
    if 'CapacitanceTest' in params_dic.keys():
        print(params_dic.keys())
        self.test_set.append(CapacitanceTest(obs_dic['CapacitanceTest'], \
            params = params_dic['CapacitanceTest']))
    if 'APThresholdTest' in params_dic.keys():
        print(params_dic.keys())
        self.test_set.append(APThresholdTest(obs_dic['APThresholdTest'], \
            params = params_dic['APThresholdTest']))
    if 'RheobasTest' in params_dic.keys():
        print(params_dic.keys())
        self.test_set.append(RheobasTestP(obs_dic['RheobasTest'], \
            params = params_dic['RheobasTest']))
    if 'InjectedCurrentAPAmplitudeTest' in params_dic.keys():
        print(params_dic.keys())
        self.test_set.append(InjectedCurrentAPAmplitudeTest(obs_dic['InjectedCurrentAPAmplitudeTest'], \
            params = params_dic['InjectedCurrentAPAmplitudeTest']))
    if 'InjectedCurrentAPWidthTest' in params_dic.keys():
        print(params_dic.keys())
        self.test_set.append(InjectedCurrentAPWidthTest(obs_dic['InjectedCurrentAPWidthTest'], \
            params = params_dic['InjectedCurrentAPWidthTest']))

    if 'RestingPotentialTest' in params_dic.keys():
        print(params_dic.keys())
        self.test_set.append(RestingPotentialTest(obs_dic['RestingPotentialTest'], \
            params = params_dic['RestingPotentialTest']))
    if 'TimeConstantTest' in params_dic.keys():
        print(params_dic.keys())
        self.test_set.append(TimeConstantTest(obs_dic['TimeConstantTest'], \
            params = params_dic['TimeConstantTest']))
    if 'InputResistanceTest' in params_dic.keys():
        print(params_dic.keys())
        print(obs_dic['InputResistanceTest'])
        print(params_dic['InputResistanceTest'])
        inht = InputResistanceTest(obs_dic['InputResistanceTest'], \
            params = params_dic['InputResistanceTest'])
        self.test_set.append(inht)

    return self.test_set


class ETest(object):
    '''
    An object for wrapping Druckman tests on instancable NeuroML-DB static models all in one neat package.
    '''
    def __init__(self,model,dtc):
        model = dtc.dtc_to_model()
        model.set_attrs(dtc.attrs)
        self.model = model
        dtc = format_test(dtc)
        self.test_set = test_setup(self,self.model,dtc.tests)


    def run_test(self, index):
        test_class = self.test_set[index]
        score = test_class.judge(self.model)#['mean']
        return (test_class,score)

    def runTest(self):
        scores = {}
        for i, t in enumerate(self.test_set):
           (tclass,score) = self.run_test(i)
           scores[tclass.name] = score
        return scores

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
    '''
    #@classmethod
