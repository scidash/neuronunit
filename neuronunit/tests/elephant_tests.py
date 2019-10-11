
"""
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
PASSIVE_DURATION = 500.0*pq.ms
PASSIVE_DELAY = 200.0*pq.ms

from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
from neuronunit.tests import InjectedCurrentAPThresholdTest, APThresholdTest
from neuronunit.tests import InjectedCurrentAPAmplitudeTest, APAmplitudeTest, APWidthTest, RestingPotentialTest, CapacitanceTest
from neuronunit.tests import TimeConstantTest, InputResistanceTest, InjectedCurrentAPWidthTest

def ft(tests,rheobase):


    protocols = {}
    for k,v in enumerate(tests):
        protocols[k] = {}
        #for t in tests:
        if hasattr(v,'passive'):#['protocol']:
            if v.passive == False and v.active == True:
                keyed = tests[k].params
                v.params = active_values(keyed,rheobase)
                #print(dtc.vtest[k]['injected_square_current']['delay']+dtc.vtest[k]['injected_square_current']['duration'])
            elif v.passive == True and v.active == False:
                keyed = tests[k].params
                v.params = passive_values(keyed)
        if v.name in str('RestingPotentialTest'):
            #try:
            v.params['injected_square_current']['amplitude'] = 0.0*pq.pA
            #except:
            #    pdb.set_trace()
            #keyed = dtc.vtest[k]
    return tests


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
    keyed['injected_square_current'] = {}
    keyed['injected_square_current']['delay']= PASSIVE_DELAY
    keyed['injected_square_current']['duration'] = PASSIVE_DURATION
    keyed['injected_square_current']['amplitude'] = -10*pq.pA
    return keyed


def test_setup(self,model,protocol_container):#,model_id,model_dict,model=None,ir_current_limited=False):
    '''
    Synopsis: Construct initialize and otherwise setup Druckman tests.
    if a model does not exist yet, but a desired NML-DB model id is known, use the model-id
    to quickly initialize a NML-DB model.

    If a model is actually passed instead, assume that model has known current_injection value
    attributes and use those.

    inputs: model_id, and a dictionary lookup table of models/model_ids

    '''
    #standard = model.rheobase
    #protocol_container = dtc.tests
    params_dic = {t.name:t.params for t in protocol_container}
    obs_dic = {t.name:t.observation for t in protocol_container }
    name_to_test = {t.name:t for t in protocol_container }
    self.test_set = []

    if 'InjectedCurrentAPThresholdTest' in params_dic.keys():
        #print(params_dic.keys())
        self.test_set.append(InjectedCurrentAPThresholdTest(obs_dic['InjectedCurrentAPThresholdTest'], \
                                                            name='',params = params_dic['InjectedCurrentAPThresholdTest']))
    if 'CapacitanceTest' in params_dic.keys():
        #print(params_dic.keys())
        self.test_set.append(CapacitanceTest(obs_dic['CapacitanceTest'], \
            name='',params = params_dic['CapacitanceTest']))
    if 'APThresholdTest' in params_dic.keys():
        #print(params_dic.keys())
        self.test_set.append(APThresholdTest(obs_dic['APThresholdTest'], \
            name='',params = params_dic['APThresholdTest']))
    if 'RheobaseTest' in params_dic.keys():
        rt = RheobaseTest(obs_dic['RheobaseTest'], \
                                          name='',params = params_dic['RheobaseTest'])
        if 'params' in rt.params.keys():
            rt.params = rt.params['params']
        self.test_set.append(rt)
    if 'InjectedCurrentAPAmplitudeTest' in params_dic.keys():
        #print(params_dic.keys())
        self.test_set.append(InjectedCurrentAPAmplitudeTest(obs_dic['InjectedCurrentAPAmplitudeTest'], \
            name='',params = params_dic['InjectedCurrentAPAmplitudeTest']))
    if 'InjectedCurrentAPWidthTest' in params_dic.keys():
        print(params_dic.keys())
        self.test_set.append(InjectedCurrentAPWidthTest(obs_dic['InjectedCurrentAPWidthTest'], \
            name='',params = params_dic['InjectedCurrentAPWidthTest']))

    if 'RestingPotentialTest' in params_dic.keys():
        #print(params_dic.keys())
        self.test_set.append(RestingPotentialTest(obs_dic['RestingPotentialTest'], \
            name='',params = params_dic['RestingPotentialTest']))
    if 'TimeConstantTest' in params_dic.keys():
        #print(params_dic.keys())
        self.test_set.append(TimeConstantTest(obs_dic['TimeConstantTest'], \
            name='',params = params_dic['TimeConstantTest']))
    if 'InputResistanceTest' in params_dic.keys():
        #print(params_dic.keys())
        #print(obs_dic['InputResistanceTest'])
        #print(params_dic['InputResistanceTest'])
        inht = InputResistanceTest(obs_dic['InputResistanceTest'], \
            name='',params = params_dic['InputResistanceTest'])
        #print(inht)
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
        self.model.rheobase = dtc.rheobase

        self.test_set = switch_logic(test_setup(self,self.model,dtc.tests))
        self.protocols = ft(self.test_set,dtc.rheobase)


    def run_test(self, index):
        test_class = self.test_set[index]
        #print(test_class.name)
        #test_class = switch_logic(test_class)
        if 'std' not in test_class.observation.keys():
            test_class.observation['std'] = list(test_class.observation.values())[0]
        
        try:
            score = test_class.judge(self.model)
        except:
            score = None
        #print('fails at {0}'.format(test_class.name))
        #import pdb
        #pdb.set_trace()
        if test_class.active:
            test_class.prediction = test_class.generate_prediction(self.model)
        if test_class.passive:
            test_class.setup_protocol(self.model)
            result = test_class.get_result(self.model)
            test_class.prediction = test_class.extract_features(self.model,result)
        #print('fails at {0}'.format(test_class.name))

        return (test_class,score)

    def runTest(self):
        scores = {}
        tclasses = {}
        for i, t in enumerate(self.test_set):
           (tclass,score) = self.run_test(i)
           if score is not None:
               if type(score.norm_score) is not type(None):
                   scores[tclass.name] = 1.0 - score.norm_score
               else:
                   scores[tclass.name] = score
           else:
               scores[tclass.name] = score

                   
           tclasses[tclass.name] = tclass
        return (scores,tclass)

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
