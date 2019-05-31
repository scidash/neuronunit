# Obtains the cell threshold, rheobase, resting v, and bias currents for
# steady state v of a cell defined in a hoc file in the given directory.
# Usage: python getCellProperties /path/To/dir/with/.hoc
#from neuronunit.tests.druckmann2013 import *
try:
    import cPickle
except:
    import _pickle as cPickle
import csv
import json
import os
import re
import shutil
import string
import urllib

import inspect

import numpy as np
from matplotlib import pyplot as plt

import dask.bag as dbag # a pip installable module, usually installs without complication
import dask
import urllib.request, json
import os
import requests
from neo.core import AnalogSignal
from quantities import mV, ms, nA
from neuronunit import models
import pickle
from neuronunit.optimisation import get_neab
from neuronunit.optimisation.optimisation_management import switch_logic#, active_values
import efel
from types import MethodType
from neuronunit.optimisation.optimisation_management import init_dm_tests

import pdb


def crawl_ids(url):
    #pdb.set_trace()

    all_data = requests.get(url)
    all_data = json.loads(all_data.text)
    print(all_data)
    pdb.set_trace()

    Model_IDs = []
    for d in all_data:
        Model_ID = str(d['Model_ID'])#,d['Directory_Path'])
        try:
            url = str('https://www.neuroml-db.org/GetModelZip?modelID=')+Model_ID+str('&version=NEURON')
            print(url)
            urllib.request.urlretrieve(url,Model_ID)
            Model_IDs.append(Model_ID)


        except:
            url = str('https://www.neuroml-db.org/GetModelZip?modelID=')+Model_ID+str('&version=NeuroML')
            print(url)

            urllib.request.urlretrieve(url,Model_ID)
            os.system(str('unzip ')+str(d['Model_ID'])+('*'))
            os.chdir(('*')+str(d['Model_ID'])+('*'))
            try:
                os.system(str('nrnivmodl *.mod'))
            except:
                print('No MOD files')
            try:
                os.system(str('pynml hhneuron.cell.nml -neuron'))
            except:
                print('No NeuroML files')

            Model_IDs.append(Model_ID)
    return Model_IDs

list_to_get =[ str('https://www.neuroml-db.org/api/search?q=traub'),
    str('https://www.neuroml-db.org/api/search?q=markram'),
    str('https://www.neuroml-db.org/api/search?q=Gouwens') ]

authors = {}
for url in list_to_get:
    #pdb.set_trace()

    Model_IDs = crawl_ids(url)
    parts = url.split('?q=')

    authors[parts[1]] = Model_IDs

try:
    #assert 1==2
    with open('static_models.p','rb') as f: sms = pickle.load(f)
    with open('waves.p','rb') as f: waves = pickle.load(f)
    #with open('waves.p','wb') as f:
    #    pickle.dump(waves,f)
except:
    def get_wave_forms(cell_id):
        url = str("https://www.neuroml-db.org/api/model?id=")+cell_id
        waveids = requests.get(url)
        waveids = json.loads(waveids.text)
        wlist = waveids['waveform_list']
        waves_to_get = []
        for wl in wlist:
            waves_to_test = {}
            wid = wl['ID']
            url = str("https://neuroml-db.org/api/waveform?id=")+str(wid)
            waves = requests.get(url)
            temp = json.loads(waves.text)
            if 'NOISE' in temp['Protocol_ID']:

                pass
            if 'RAMP' in temp['Protocol_ID']:
                pass
            if 'SHORT_SQUARE_TRIPPLE' in temp['Protocol_ID']:
                print((temp['Waveform_Label']))
                pass
            if type(temp['Waveform_Label']) is type(None):
                break
            if str("1.0xRB") in temp['Waveform_Label'] or str("1.25xRB") in temp['Waveform_Label']:
                print(temp['Protocol_ID'])
                temp_vm = list(map(float, temp['Variable_Values'].split(',')))
                waves_to_test['Times'] = list(map(float,temp['Times'].split(',')))
                waves_to_test['DURATION'] = temp['Time_End'] -temp['Time_Start']
                waves_to_test['DELAY'] = temp['Time_Start']
                waves_to_test['Time_End'] = temp['Time_End']
                waves_to_test['Time_Start'] = temp['Time_Start']
                dt = waves_to_test['Times'][1]- waves_to_test['Times'][0]
                waves_to_test['vm'] = AnalogSignal(temp_vm,sampling_period=dt*ms,units=mV)

                #waves_to_test['everything'] = temp

                sm = models.StaticModel(waves_to_test['vm'])
                sm.complete = None
                sm.complete = temp
                sm.complete['vm'] = waves_to_test['vm']

                #import pdb; pdb.set_trace()

                trace = {}
                DURATION = sm.complete['Time_End'] -sm.complete['Time_Start']

                trace['T'] = waves_to_test['Times']
                #trace['V'] = vm = [float(v[0]) for v in trace['V'].tolist()]

                trace['V'] = temp_vm
                #AnalogSignal(temp_vm,sampling_period=dt*ms,units=mV)
                trace['stim_start'] = [sm.complete['Time_Start']]#rtest.run_params[]

                trace['stim_end'] = [sm.complete['Time_End'] ]# list(sm.complete['duration'])
                traces = [trace]# Now we pass 'traces' to the efel and ask it to calculate the feature# values
                #import pdb; pdb.set_trace()
                print(temp['Spikes'])
                if temp['Spikes'] and np.min(temp_vm)<0:

                    traces_results = efel.getFeatureValues(traces,list(efel.getFeatureNames()))#
                    for v in traces_results:
                        for key,value in v.items():
                            if type(value) is not type(None):
                                #pass
                                print(key,value)
                else:
                    print(temp['Spikes'],np.min(temp_vm))
                    pass




            if 'SHORT_SQUARE' in temp['Protocol_ID'] and not 'SHORT_SQUARE_TRIPPLE' in temp['Protocol_ID'] or 'SHORT_SQUARE_TRIPPLE' in temp['Protocol_ID']:
                try:
                    parts = temp['Waveform_Label'].split(' ')

                    #import pdb; pdb.set_trace()
                    print(' value ',parts[0], ' units ',parts[1])

                    waves_to_test['prediction'] = float(parts[0])*nA# '1.1133 nA',
                    #print('yes')

                    temp_vm = list(map(float, temp['Variable_Values'].split(',')))

                    waves_to_test['Times'] = list(map(float,temp['Times'].split(',')))
                    waves_to_test['DURATION'] = temp['Time_End'] -temp['Time_Start']
                    waves_to_test['DELAY'] = temp['Time_Start']
                    waves_to_test['Time_End'] = temp['Time_End']
                    waves_to_test['Time_Start'] = temp['Time_Start']
                    dt = waves_to_test['Times'][1]- waves_to_test['Times'][0]
                    waves_to_test['vm'] = AnalogSignal(temp_vm,sampling_period=dt*ms,units=mV)
                    waves_to_test['everything'] = temp
                    waves_to_get.append(waves_to_test)

                except:

                    if temp['Waveform_Label'] is None:
                        pass
                    pass
        return waves_to_get
    waves = get_wave_forms(str('NMLCL001129'))
    with open('waves.p','wb') as f:
        pickle.dump(waves,f)


protocols = [ w['everything']['Protocol_ID'] for w in waves ]
ss = [ w['everything'] for w in waves if w['everything']['Protocol_ID'] == 'SHORT_SQUARE' ]
#not_current_injections = [ w for w in waves if w['everything']['Spikes'] > 1 ]

not_current_injections = [ w for w in waves if w['everything']['Variable_Name'] != str('Current') ]
sms = []
for w in not_current_injections:

    sm = models.StaticModel(w['vm'])
    sm.rheobase = {}
    sm.rheobase['mean'] = w['prediction']
    sm.complete = w
    sms.append(sm)
with open('static_models.p','wb') as f:
    pickle.dump(sms,f)
current_injections = [ w for w in waves if w['everything']['Variable_Name'] == str('Current') ]

electro_path = str(os.getcwd())+'/examples/pipe_tests.p'

assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    test_frame = pickle.load(f)

def generate_prediction(self,model):
    prediction = {}
    prediction['n'] = 1
    prediction['std'] = 1.0
    prediction['mean'] = model.rheobase['mean']
    return prediction
    #model.get_membrane_potential()

#import pdb; pdb.set_trace()
test_scores = []
tt = [tests for tests in test_frame[0][0] ]
#vtest[k] = active_values(keyed,dtc.rheobase)
for t in tt: t.generate_prediction = MethodType(generate_prediction,t)
params = {}

'''
def get_m_p(cls,params = {}):
    return model.get_membrane_potential()
for model in sms:
    model.inject_square_current = MethodType(get_m_p,model)#get_membrane_potential
'''
#def get_membrane_potential():

#for model in sms: model.inject_square_current = MethodType(model.get_membrane_potential,model)

#for model in sms: model.inject_square_current = MethodType(model.get_membrane_potential,model)
#for model in sms: print(model.inject_square_current())# = MethodType(model.get_membrane_potential,params)
#for t in tt: t.generate_prediction = MethodType(generate_prediction,t)

def active_values(keyed,rheobase,square = None):
    keyed['injected_square_current'] = {}
    if square == None:
        DURATION = 1000.0*pq.ms
        DELAY = 100.0*pq.ms
        if type(rheobase) is type({str('k'):str('v')}):
            keyed['injected_square_current']['amplitude'] = float(rheobase['value'])*pq.pA
        else:
            keyed['injected_square_current']['amplitude'] = rheobase
    else:
        DURATION = square['Time_End'] -square['Time_Start']
        DELAY = square['Time_Start']
        keyed['injected_square_current']['amplitude'] = square['prediction']#value'])*pq.pA

    keyed['injected_square_current']['delay']= DELAY
    keyed['injected_square_current']['duration'] = DURATION

    return keyed



tt = switch_logic(tt) # for tests in tt ]
for t in tt:
    for sm in sms:
        rheobase = sm.complete['prediction']
        t.params = {}
        t.params = active_values(t.params,rheobase,square=sm.complete)


flat_iter = [(t,sm) for t in tt for sm in sms]
try:
    from neuronunit.optimisation.optimisation_management import add_druckmann_properties_to_cells as dm
except:
    from neuronunit.optimisation.optimisation_management import add_dm_properties_to_cells as dm

for t,sm in flat_iter:
    sm._backend = None
    if t.active:
        t.params = {}
        t.params['injected_square_current'] = None

        #score = rtest.judge(model)
        results = sm.get_membrane_potential()
        value = float(np.max(current_injections[0]['vm']))

        dm_tests = init_dm_tests(value,1.5*value)

        predictions = [ dm.generate_prediction(sm) for dm in dm_tests ]
        #import pdb; pdb.set_trace()
        trace = {}
        DURATION = sm.complete['Time_End'] -sm.complete['Time_Start']

        trace['T'] = sm.complete['Times']
        trace['V'] = results
        trace['stim_start'] = [sm.complete['Time_Start']]#rtest.run_params[]

        trace['stim_end'] = [sm.complete['Time_End'] ]# list(sm.complete['duration'])
        traces = [trace]# Now we pass 'traces' to the efel and ask it to calculate the feature# values
        traces_results = efel.getFeatureValues(traces,list(efel.getFeatureNames()))#
        for v in traces_results:
            for key,value in v.items():
                if type(value) is not type(None):
                    #pass
                    print(key,value)
        #try:
            #import pdb; pdb.set_trace()
        #    sm.get_membrane_potential()
        #    test_scores.append(t.judge(sm))
        #    print(test_scores[-1],t.name)
        #except:
            #test_scores.append(t.judge(sm))

            #print(test_scores[-1],t.name)

            print('active skipped: ',t.name)
        print(test_scores)
        #all_models = json.loads(url.read().decode())
def get_all(Model_ID = str('NMLNT001592')):
    if Model_ID == None:
        try:
            # Obtains the cell threshold, rheobase, resting v, and bias currents for
            #with urllib.request.urlopen("https://www.neuroml-db.org/api/models") as url:
            #    all_models = json.loads(url.read().decode())


            url = str("https://www.neuroml-db.org/api/models")
            all_models = requests.get(url)
            all_models = json.loads(all_models.text)
            print(all_models)
            for d in all_models:
                print(d.keys())
            for d in all_models[0]:
                print(d['Model_ID'],d['Directory_Path'])
                url = str('https://www.neuroml-db.org/GetModelZip?modelID=')+str(d['Model_ID'])+str('&version=NeuroML')
                urllib.request.urlretrieve(url,Model_ID)
                #https://www.neuroml-db.org/api/models'
                #url = str('https://www.neuroml-db.org/GetModelZip?modelID=')+str(d['Model_ID'])+str('&version=NeuroML')
                #os.system('wget '+str(url))
                os.system(str('unzip *')+str(d['Model_ID'])+('*'))
                os.system('cd *')+str(d['Model_ID'])+('*'))
                os.system(str('pynml hhneuron.cell.nml -neuron'))
            return data

        except:
            pass
    else:
        d = {}
        d['Model_ID'] = Model_ID
        #print(d['Model_ID'],d['Directory_Path'])
        #https://www.neuroml-db.org/api/models'
        url = "https://www.neuroml-db.org/GetModelZip?modelID=NMLNT001592&version=NeuroML"
        #url = str('https://www.neuroml-db.org/GetModelZip?modelID=')+str(d['Model_ID'])+str('&version=NeuroML')
        urllib.request.urlretrieve(url,Model_ID)
        print(url)
        url = "https://www.neuroml-db.org/GetModelZip?modelID=NMLNT001592&version=NeuroML"
        os.system('wget '+str(url))
        os.system(str('unzip ')+str(d['Model_ID'])+('*'))
        os.system('cd *')+str(d['Model_ID'])+('*'))

        os.system(str('pynml hhneuron.cell.nml -neuron'))


def run_cell():
    from neuron import h
    h.load_file('hhneuron.hoc')
    cell = h.hhneuron
    d = {}
    d['Model_ID'] = str('NT001592')
    with urllib.request.urlopen(str('https://www.neuroml-db.org/api/model?id=')+str(d['Model_ID'])) as url:
        data_on_model = json.loads(url.read().decode())
