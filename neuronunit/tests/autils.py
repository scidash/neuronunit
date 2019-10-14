
##
#  The main method that does the aligned feature extraction is down the bottom.
# Two thirds of this file, it is called
# def three_feature_sets_on_static_models
##

##
# docker pull russelljarvis/efel_allen_dm
# I build it with the name russelljarvis/efel_allen_dm.
# This uses the docker file in this directory.
# I build it with the name efl.
# and launch it with this alias.
# alias efel='cd /home/russell/outside/neuronunit; sudo docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/russell/outside/neuronunit:/home/jovyan/neuronunit -v /home/russell/Dropbox\ \(ASU\)/AllenDruckmanData:/home/jovyan/work/allendata efel /bin/bash'
##

##
# This is how my travis script builds and runs:
# before_install:
# - docker pull russelljarvis/efel_allen_dm
# - git clone -b barcelona https://github.com/russelljjarvis/neuronunit.git
#
# Run the unit test
# script:
# show that running the docker container at least works.
#  - docker run -v neuronunit:/home/jovyan/neuronunit russelljarvis/efel_allen_dm python /home/jovyan/work/allendata/small_travis_run.py
#
##

from allensdk.ephys.ephys_extractor import EphysSweepSetFeatureExtractor
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
from types import MethodType
import quantities as pq
import pdb

from collections import Iterable, OrderedDict

import numpy as np
try:
    import efel
except:
    print('warning efel not installed')
import pickle
from allensdk.ephys.extract_cell_features import extract_cell_features
import pandas as pd
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from neuronunit.neuromldb import NeuroMLDBStaticModel

#import dm_test_interoperable #import Interoperabe
from dask import bag as db
import glob


import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('agg')
#import logging
#logging.info("test")
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.extract_cell_features import extract_cell_features
from collections import defaultdict
#from neuronunit.optimisation.optimisation_management import inject_rh_and_dont_plot, add_dm_properties_to_cells

from allensdk.core.nwb_data_set import NwbDataSet
import pickle


from neuronunit.tests.allen_tests_utils import three_feature_sets_on_static_models
import quantities as qt
import copy


def find_nearest(array, value):
    #value = float(value)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx], idx)


def sweep_to_analog_signal(sweep_data):
    temp_vm = sweep_data['response']
    injection = sweep_data['stimulus']
    sampling_rate = sweep_data['sampling_rate']
    vm = AnalogSignal(temp_vm,sampling_rate=sampling_rate*qt.Hz,units=qt.V)
    vm = vm.rescale(qt.mV)
    return vm

def get_15_30(model,rheobase,data_set=None):
    '''
    model, data_set is basically like a lookup table.
    '''

    if not hasattr(rheobase,'units'):
        rheobase = rheobase*qt.A
        rheobase = rheobase.rescale(qt.pA)

    if data_set == None:
        data_set = model.data_set
        sweeps = model.sweeps
    supras = [s for s in sweeps if s['stimulus_name'] == str('Square - 2s Suprathreshold')]
    supra_currents = [(s['stimulus_absolute_amplitude']*qt.pA).rescale(qt.pA) for s in supras]

    supra_numbers = [s['sweep_number'] for s in supras]
    zipped_current_numbers = list(zip(supra_currents,supra_numbers))

    index_15 = False
    for i,zi in enumerate(zipped_current_numbers):
        if zi[0]== np.median(supra_currents):
            index_15 = zi[1]
            model.druckmann2013_standard_current = supra_currents[i]

    print(supras)
    print(len(supras))

    model.druckmann2013_strong_current = supra_currents[-1]
    try:
        sweep_data30 = data_set.get_sweep(supra_numbers[-1])
        vm30 = sweep_to_analog_signal(sweep_data30)

    except:
        import pdb
        pdb.set_trace()

    if index_15 == False:
        index_15 = supra_numbers[int(len(supra_currents)/2.0)]
        model.druckmann2013_standard_current = supra_currents[int(len(supra_currents)/2.0)]

    sweep_data15 = data_set.get_sweep(index_15)
    vm15 = sweep_to_analog_signal(sweep_data15)

    model.vm30 = vm30
    model.vm15 = vm15

    return

def inject_square_current(model,current,data_set=None):
    '''
    model, data_set is basically like a lookup table.
    draws from a dictionary of stored models
    '''
    if type(current) is type({}):
        current = current['amplitude']

    if not hasattr(current,'units'):
        current = current*qt.A
        current = current.rescale(qt.pA)

    if hasattr(model,'fast_lookup'):
        if float(current) in model.fast_lookup.keys():
            return model.fast_lookup[float(current)]

    if data_set == None:
        data_set = model.data_set
        sweeps = model.sweeps
    if not hasattr(model,'all_long_square_injections'):
        supras = [s for s in sweeps if s['stimulus_name'] == str('Square - 2s Suprathreshold')]
        subs = [s for s in sweeps if s['stimulus_name'] == str('Square - 0.5ms Subthreshold')]
        sub_currents = [(s['stimulus_absolute_amplitude']*qt.pA).rescale(qt.pA) for s in subs]
        supra_currents = [(s['stimulus_absolute_amplitude']*qt.pA).rescale(qt.pA) for s in supras]
        supra_numbers = [s['sweep_number'] for s in supras]
        sub_numbers = [s['sweep_number'] for s in subs]
        everything = supra_currents
        everything.extend(sub_currents)
        everything_index = supra_numbers
        everything_index.extend(sub_numbers)
        model.all_long_square_injections = None
        model.all_long_square_injection_indexs = None
        model.fast_lookup = None
        model.all_long_square_injections = everything
        model.all_long_square_injection_indexs = everything_index
        model.fast_lookup = {}
    (nearest,idx) = find_nearest(model.all_long_square_injections,float(current))
    index = np.asarray(model.all_long_square_injection_indexs)[idx]
    sweep_data = data_set.get_sweep(index)
    model._vm = sweep_to_analog_signal(sweep_data)
    model.fast_lookup[float(current)] = model._vm
    return model._vm




def get_membrane_potential(model):
    return model._vm

"""Auxiliary helper functions for analysis of spiking."""

def get_spike_train(vm, threshold=0.0*mV):
    """
    Inputs:
     vm: a neo.core.AnalogSignal corresponding to a membrane potential trace.
     threshold: the value (in mV) above which vm has to cross for there
                to be a spike.  Scalar float.

    Returns:
     a neo.core.SpikeTrain containing the times of spikes.
    """
    from elephant.spike_train_generation import threshold_detection

    spike_train = threshold_detection(vm, threshold=threshold)
    return spike_train

def get_spike_count(model):
    vm = model.get_membrane_potential()
    train = get_spike_train(vm)
    return len(train)



def get_data_sets_from_cache(do_features=True):
    path_name = '../tests/data_nwbs'
    files = glob.glob(path_name+'/*.p')
    data_sets = []
    for temp_path in files:
        #print(temp_path)
        if os.path.exists(temp_path):
            with open(temp_path,'rb') as f:
                (data_set_nwb,sweeps,specimen_id) = pickle.load(f)
            data_sets.append((data_set_nwb,sweeps,specimen_id))
            if do_features == True:
                allen_to_model_and_features(data_sets[-1])
    return data_sets



def get_data_sets_from_remote(upper_bound=2,lower_bound=None):
    try:
        with open('all_allen_cells.p','rb') as f: cells = pickle.load(f)
        ctc = CellTypesCache(manifest_file='cell_types/manifest.json')

    except:
        ctc = CellTypesCache(manifest_file='cell_types/manifest.json')

        cells = ctc.get_cells()
        inhibitory_rodent = []
        excitatory_rodent = []
        for cell in cells:
            if str("Mus musculus") in cell['species']:
                print(cell['structure_layer_name'], cell['species'], cell['name'])
                if str("Pvalb-IRES-Cre") in cell['name']:
                    inhibitory_rodent.append(cell)
                else:
                    excitatory_rodent.append(cell)
        with open('all_allen_cells.p','wb') as f:
            pickle.dump(cells,f)
    data = []
    data_sets = []
    path_name = 'data_nwbs'

    try:
        os.mkdir(path_name)
    except:
        print('directory already made.')

    ids = [ c['id'] for c in cells ]
    if upper_bound == None and lower_bound is None:
        limited_range = ids[0:-1]
    elif upper_bound is not None and lower_bound is not None:
        limited_range = ids[lower_bound:upper_bound]
    else:
        limited_range = ids[0:upper_bound]
    cnt=0
    for specimen_id in limited_range:
        temp_path = str(path_name)+str('/')+str(specimen_id)+'.p'
        if os.path.exists(temp_path):
            cnt+=1
    for specimen_id in limited_range:
        temp_path = str(path_name)+str('/')+str(specimen_id)+'.p'
        if os.path.exists(temp_path):
            with open(temp_path,'rb') as f:
                (data_set_nwb,sweeps,specimen_id) = pickle.load(f)
            data_sets.append((data_set_nwb,sweeps,specimen_id))
        else:

            data_set = ctc.get_ephys_data(specimen_id)
            sweeps = ctc.get_ephys_sweeps(specimen_id)

            file_name = 'cell_types/specimen_'+str(specimen_id)+'/ephys.nwb'
            data_set_nwb = NwbDataSet(file_name)

            data_sets.append((data_set_nwb,sweeps,specimen_id))

            with open(temp_path,'wb') as f:
                pickle.dump((data_set_nwb,sweeps,specimen_id),f)
    return data_sets


def get_static_models_allen(content):
    data_set,sweeps,specimen_id = content
    try:
        sweep_numbers = data_set.get_sweep_numbers()
    except:
        return (False,specimen_id)
    for sn in sweep_numbers:
        spike_times = data_set.get_spike_times(sn)
        sweep_data = data_set.get_sweep(sn)
    ##
    try:
        cell_features = extract_cell_features(data_set, sweep_numbers_['Ramp'],sweep_numbers_['Short Square'],sweep_numbers_['Long Square'])
    except:
        print('did this used to work?')
    ##
    cell_features = None
    if cell_features is not None:
        spiking_sweeps = cell_features['long_squares']['spiking_sweeps'][0]
        multi_spike_features = cell_features['long_squares']['hero_sweep']
        biophysics = cell_features['long_squares']
        shapes =  cell_features['long_squares']['spiking_sweeps'][0]['spikes'][0]

    supras = [s for s in sweeps if s['stimulus_name'] == str('Square - 2s Suprathreshold')]
    if len(supras) == 0:
        return None
    supra_numbers = [s['sweep_number'] for s in supras]

    smallest_multi = 1000
    all_currents = []
    temp_vm = None
    for sn in supra_numbers:
        spike_times = data_set.get_spike_times(sn)
        sweep_data = data_set.get_sweep(sn)

        if len(spike_times) == 1:
            inj_rheobase = np.max(sweep_data['stimulus'])
            temp_vm = sweep_data['response']
            break
        if len(spike_times) < smallest_multi and len(spike_times) > 1:
            smallest_multi = len(spike_times)
            inj_multi_spike = np.max(sweep_data['stimulus'])
            inj_rheobase = inj_multi_spike
            temp_vm = sweep_data['response']

    spike_times = data_set.get_spike_times(supras[-1]['sweep_number'])
    sweep_data = data_set.get_sweep(supras[-1]['sweep_number'])
    sd = sweep_data['stimulus']
    # sampling rate is in Hz
    sampling_rate = sweep_data['sampling_rate']

    inj = AnalogSignal(sd,sampling_rate=sampling_rate*qt.Hz,units=qt.pA)

    indexs = np.where(sd==np.max(sd))[0]



    if temp_vm is None:
        return (None,None,None,None)
    vm = AnalogSignal(temp_vm,sampling_rate=sampling_rate*qt.Hz,units=qt.V)
    sm = models.StaticModel(vm)
    sm.allen = None
    sm.allen = True
    sm.protocol = {}
    sm.protocol['Time_Start'] = inj.times[indexs[0]]
    sm.protocol['Time_End'] = inj.times[indexs[-1]]

    sm.name = specimen_id
    sm.data_set = data_set
    sm.sweeps = sweeps
    sm.inject_square_current = MethodType(inject_square_current,sm)
    sm.get_membrane_potential = MethodType(get_membrane_potential,sm)


    sm.rheobase_current = inj_rheobase
    current = {}
    current['amplitude'] = sm.rheobase_current
    sm.vm_rheobase = sm.inject_square_current(current)

    try:
        import asciiplotlib as apl
        fig = apl.figure()
        fig.plot([float(t) for t in sm.vm30.times],[float(v) for v in sm.vm30], label="data", width=100, height=80)
        fig.show()

        import asciiplotlib as apl
        fig = apl.figure()
        fig.plot([float(t) for t in sm.vm15.times],[float(v) for v in sm.vm15], label="data", width=100, height=80)
        fig.show()
    except:
        pass
    sm.get_spike_count = MethodType(get_spike_count,sm)
    subs = [s for s in sweeps if s['stimulus_name'] == str('Square - 0.5ms Subthreshold')]
    sub_currents = [(s['stimulus_absolute_amplitude']*qt.A).rescale(qt.pA) for s in subs]
    if len(sub_currents) == 3:
        sm.druckmann2013_input_resistance_currents = [ sub_currents[0], sub_currents[1], sub_currents[2] ]

    elif len(sub_currents) == 2:

        sm.druckmann2013_input_resistance_currents = [ sub_currents[0], sub_currents[0], sub_currents[1] ]
    elif len(sub_currents) == 1:
        # unfortunately only one inhibitory current available here.
        sm.druckmann2013_input_resistance_currents = [ sub_currents[0], sub_currents[0], sub_currents[0] ]
    try:
        sm.inject_square_current(sub_currents[0])
    except:
        pass
    get_15_30(sm,inj_rheobase)
    #everything = (sm,sweep_data,cell_features,vm)
    with open(str('models')+str('/')+str(specimen_id)+'.p','wb') as f:
        pickle.dump(sm,f)
    return (True,specimen_id)


def allen_to_model_and_features(content,cell_features = None):
    data_set,sweeps,specimen_id = content
    try:
        sweep_numbers = data_set.get_sweep_numbers()
    except:
        print('erroneous deletion of relevant ephys.nwb file')
        file_name = 'cell_types/specimen_'+str(specimen_id)+'/ephys.nwb'
        data_set_nwb = NwbDataSet(file_name)
        try:
            sweep_numbers = data_set_nwb.get_sweep_numbers()
        except:
            return None

    for sn in sweep_numbers:
        spike_times = data_set.get_spike_times(sn)
        sweep_data = data_set.get_sweep(sn)


    ##
    #if cell_features is not None:


    if cell_features is not None:
        cell_features = extract_cell_features(data_set, sweep_numbers_['Ramp'],sweep_numbers_['Short Square'],sweep_numbers_['Long Square'])
    ##
        spiking_sweeps = cell_features['long_squares']['spiking_sweeps'][0]
        multi_spike_features = cell_features['long_squares']['hero_sweep']
        biophysics = cell_features['long_squares']
        shapes =  cell_features['long_squares']['spiking_sweeps'][0]['spikes'][0]

    supras = [s for s in sweeps if s['stimulus_name'] == str('Square - 2s Suprathreshold')]
    if len(supras) == 0:
        return None
    supra_numbers = [s['sweep_number'] for s in supras]

    smallest_multi = 1000
    all_currents = []
    temp_vm = None
    for sn in supra_numbers:
        spike_times = data_set.get_spike_times(sn)
        sweep_data = data_set.get_sweep(sn)

        if len(spike_times) == 1:
            inj_rheobase = np.max(sweep_data['stimulus'])
            temp_vm = sweep_data['response']
            break
        if len(spike_times) < smallest_multi and len(spike_times) > 1:
            smallest_multi = len(spike_times)
            inj_multi_spike = np.max(sweep_data['stimulus'])
            inj_rheobase = inj_multi_spike
            temp_vm = sweep_data['response']

    spike_times = data_set.get_spike_times(supras[-1]['sweep_number'])
    sweep_data = data_set.get_sweep(supras[-1]['sweep_number'])
    sd = sweep_data['stimulus']
    # sampling rate is in Hz
    sampling_rate = sweep_data['sampling_rate']

    inj = AnalogSignal(sd,sampling_rate=sampling_rate*qt.Hz,units=qt.pA)

    indexs = np.where(sd==np.max(sd))[0]



    if temp_vm is None:
        return (None,None,None,None)
    vm = AnalogSignal(temp_vm,sampling_rate=sampling_rate*qt.Hz,units=qt.V)
    sm = models.StaticModel(vm)
    sm.allen = None
    sm.allen = True
    sm.protocol = {}
    sm.protocol['Time_Start'] = inj.times[indexs[0]]
    sm.protocol['Time_End'] = inj.times[indexs[-1]]

    sm.name = specimen_id
    sm.data_set = data_set
    sm.sweeps = sweeps
    sm.inject_square_current = MethodType(inject_square_current,sm)
    sm.get_membrane_potential = MethodType(get_membrane_potential,sm)


    sm.rheobase_current = inj_rheobase
    current = {}
    current['amplitude'] = sm.rheobase_current
    sm.vm_rheobase = sm.inject_square_current(current)

    try:
        import asciiplotlib as apl
        fig = apl.figure()
        fig.plot([float(t) for t in sm.vm30.times],[float(v) for v in sm.vm30], label="data", width=100, height=80)
        fig.show()

        import asciiplotlib as apl
        fig = apl.figure()
        fig.plot([float(t) for t in sm.vm15.times],[float(v) for v in sm.vm15], label="data", width=100, height=80)
        fig.show()
    except:
        pass
    sm.get_spike_count = MethodType(get_spike_count,sm)
    subs = [s for s in sweeps if s['stimulus_name'] == str('Square - 0.5ms Subthreshold')]
    sub_currents = [(s['stimulus_absolute_amplitude']*qt.A).rescale(qt.pA) for s in subs]
    if len(sub_currents) == 3:
        sm.druckmann2013_input_resistance_currents = [ sub_currents[0], sub_currents[1], sub_currents[2] ]

    elif len(sub_currents) == 2:

        sm.druckmann2013_input_resistance_currents = [ sub_currents[0], sub_currents[0], sub_currents[1] ]
    elif len(sub_currents) == 1:
        # unfortunately only one inhibitory current available here.
        sm.druckmann2013_input_resistance_currents = [ sub_currents[0], sub_currents[0], sub_currents[0] ]
    try:
        sm.inject_square_current(sub_currents[0])
    except:
        pass
    get_15_30(sm,inj_rheobase)
    everything = (sm,sweep_data,cell_features,vm)
    return everything

def model_analysis(model):
    try:
        os.mkdir(str('allen_three_feature_folder'))
    except:
        print('directory already exists')
    if hasattr(model,'name'):
        temp_path = str('allen_three_feature_folder')+str('/')+str(model.name)+str('.p')
    else:
        return
    print('exclusion worked',os.path.exists(temp_path))

    if type(model) is not type(None) and not os.path.exists(temp_path):
        print('got passed exclusion \n\n\n\n\n')
        three_feature_sets = three_feature_sets_on_static_models(model)
        #try:
        #    assert type(model.name) is not None
        with open(str('allen_three_feature_folder')+str('/')+str(model.name)+'.p','wb') as f:
            pickle.dump(three_feature_sets,f)
        print(os.path.exists(temp_path), 'file wrote')
        #os.system('ls /allen_three_feature_folder | wc -l')

        #except:
        #    print('big error')
        #    import pdb; pdb.set_trace()
    return


def run_on_allen(number_d_sets=2):
    try:
        with open('allen_data.pkl','rb') as f:
            data_sets = pickle.load(f)
    except:
        data_sets = get_data_sets(number_d_sets=number_d_sets)
        with open('allen_data.pkl','wb') as f:
            pickle.dump(data_sets,f)

    models = []
    for data_set in data_sets:
        models.append(allen_to_model_and_features(data_set))
    models = [mod for mod in models if mod is not None]
    models = [mod[0] for mod in models]

    three_feature_sets = []

    for model in models:
        #if model is not None:
        temp_path = str('allen_three_feature_folder')+str('/')+str(model.name)+str('.p')
        if not os.path.exists(temp_path):
            model_analysis(model)
        #three_feature_sets.append(three_feature_sets_on_static_models(model))

def faster_run_on_allen(number_d_sets=1300):
    if os.path.isfile('allen_models.pkl'):
        with open('allen_models.pkl','rb') as f:
            models = pickle.load(f)
        if len(models) < number_d_sets:
            print(len(models),number_d_sets)
            data_sets = get_data_sets(lower_bound=len(models),upper_bound=number_d_sets)
            #models = []

            lazy_arrays = [dask.delayed(allen_to_model_and_features)(m) for m in data_sets]
            models = [ l.compute() for l in lazy_arrays ]


            #data_bag = db.from_sequence(data_sets,npartitions=8)
            #models = list(data_bag.map(allen_to_model_and_features).compute())
            models = [mod for mod in models if mod is not None]
            models = [mod[0] for mod in models]
            with open('allen_models.pkl','wb') as f:
                pickle.dump(models,f)
    else:
        data_sets = get_data_sets(lower_bound=0,upper_bound=number_d_sets)
        models = []
        data_bag = db.from_sequence(data_sets,npartitions=8)
        models = list(data_bag.map(allen_to_model_and_features).compute())
        models = [mod for mod in models if mod is not None]
        models = [mod[0] for mod in models]
        with open('allen_models.pkl','wb') as f:
            pickle.dump(models,f)

    #    model_analysis(model)

    lazy_arrays = [dask.delayed(model_analysis)(m) for m in models]
    [ l.compute() for l in lazy_arrays ]
    '''
    data_bag = db.from_sequence(models[0:int(len(models)/4.0)],npartitions=8)
    _ = list(data_bag.map(model_analysis).compute())
    data_bag = db.from_sequence(models[int(len(models)/4.0)+1:int(len(models)/2.0)],npartitions=8)
    _ = list(data_bag.map(model_analysis).compute())
    data_bag = db.from_sequence(models[int(len(models)/2.0)+1:3*int(len(models)/4.0)],npartitions=8)
    _ = list(data_bag.map(model_analysis).compute())
    data_bag = db.from_sequence(models[int(len(models)/4.0):-1],npartitions=8)
    _ = list(data_bag.map(model_analysis).compute())


    data_bag = db.from_sequence(models[0:int(len(models)/2.0)],npartitions=8)
    _ = list(data_bag.map(model_analysis).compute())
    data_bag = db.from_sequence(models[int(len(models)/2.0)+1:-1],npartitions=8)
    _ = list(data_bag.map(model_analysis).compute())
    '''
    return


def faster_run_on_allen_cached():


    data_sets = get_data_sets_from_cache()#(lower_bound=len(models),upper_bound=number_d_sets)
    models = []
    data_bag = db.from_sequence(data_sets,npartitions=8)
    models = list(data_bag.map(allen_to_model_and_features).compute())
    data_bag = db.from_sequence(models[0:int(len(models)/4.0)],npartitions=8)
    _ = list(data_bag.map(model_analysis).compute())
    data_bag = db.from_sequence(models[int(len(models)/4.0)+1:int(len(models)/2.0)],npartitions=8)
    _ = list(data_bag.map(model_analysis).compute())
    data_bag = db.from_sequence(models[int(len(models)/2.0)+1:3*int(len(models)/4.0)],npartitions=8)
    _ = list(data_bag.map(model_analysis).compute())
    data_bag = db.from_sequence(models[int(len(models)/4.0):-1],npartitions=8)
    _ = list(data_bag.map(model_analysis).compute())
    return
'''
data_bag = models[0:int(len(models)/4.0)]#,npartitions=8)
_ = list(map(model_analysis,data_bag))
data_bag = models[int(len(models)/4.0)+1:int(len(models)/2.0)]#,npartitions=8)
_ = list(map(model_analysis,data_bag))
data_bag = models[int(len(models)/2.0)+1:3*int(len(models)/4.0)]#,npartitions=8)
_ = list(map(model_analysis,data_bag))
data_bag = models[int(len(models)/4.0):-1]#,npartitions=8)
_ = list(map(model_analysis,data_bag))#compute())

data_bag = db.from_sequence(models[0:int(len(models)/2.0)],npartitions=8)
_ = list(data_bag.map(model_analysis).compute())
data_bag = db.from_sequence(models[int(len(models)/2.0)+1:-1],npartitions=8)
_ = list(data_bag.map(model_analysis).compute())
'''

'''
def faster_run_on_allen_revised():
    #import pdb; pdb.set_trace()
    try:
        #assert 1==2
        if os.path.isfile('allen_models.pkl'):
            with open('allen_models.pkl','rb') as f: models = pickle.load(f)

    except:
            #if len(models) < number_d_sets:
        data_sets = get_data_sets(lower_bound=0,upper_bound=-1)
        models = []
        data_bag = db.from_sequence(data_sets,npartitions=8)
        models = list(data_bag.map(allen_to_model_and_features).compute())
        #models = list(map(allen_to_model_and_features,data_sets))


        models = [mod for mod in models if mod is not None]
        models = [mod[0] for mod in models]
        with open('allen_models.pkl','wb') as f:
            pickle.dump(models,f)
    data_bag = db.from_sequence(models[0:int(len(models)/2.0)],npartitions=8)
    _ = list(data_bag.map(model_analysis).compute())
    data_bag = db.from_sequence(models[int(len(models)/2.0)+1:-1],npartitions=8)
    _ = list(data_bag.map(model_analysis).compute())
    return
'''

def analyze_models_from_cache(file_paths):
    models = []
    for f in file_paths:
        models.append(pickle.load(open(f,'rb')))
    models_bag = db.from_sequence(models,npartitions=8)
    list(models_bag.map(model_analysis).compute())

def faster_feature_extraction():
    all_the_NML_IDs =  pickle.load(open('cortical_NML_IDs/cortical_cells_list.p','rb'))
    file_paths = glob.glob("models/*.p")
    if file_paths:
        if len(file_paths)==len(all_the_NML_IDs):
            _ = analyze_models_from_cache(file_paths)
        else:
            _ = faster_make_model_and_cache()
    else:
        _ = faster_make_model_and_cache()
    file_paths = glob.glob("models/*.p")
    _ = analyze_models_from_cache(file_paths)



def more_challenging(model):
    '''
    Isolate harder code, still wrangling data types.
    When this is done, EFEL might be able to report back about input resistance.
    '''
    single_spike = {}
    single_spike['APWaveForm'] = [ float(v) for v in model.vm_rheobase]#temp_vm
    single_spike['T'] = [ float(t) for t in model.vm_rheobase.times.rescale('ms') ]
    single_spike['V'] = [ float(v) for v in model.vm_rheobase ]#temp_vm
    single_spike['stim_start'] = [ float(model.protocol['Time_Start']) ]
    single_spike['stimulus_current'] = [ model.model.rheobase_current ]
    single_spike['stim_end'] = [ trace15['T'][-1] ]

    single_spike = [single_spike]


    ##
    # How EFEL could learn about input resistance of model
    ##
    trace_ephys_prop = {}
    trace_ephys_prop['stimulus_current'] = model.druckmann2013_input_resistance_currents[0]# = druckmann2013_input_resistance_currents[0]
    trace_ephys_prop['V'] = [ float(v) for v in model.vminh ]
    trace_ephys_prop['T'] = [ float(t) for t in model.vminh.times.rescale('ms') ]
    trace_ephys_prop['stim_end'] = [ trace15['T'][-1] ]
    trace_ephys_prop['stim_start'] = [ float(model.inh_protocol['Time_Start']) ]# = in_current_filter[0]['Time_End']
    trace_ephys_props = [trace_ephys_prop]

    efel_results_inh = efel.getFeatureValues(trace_ephys_props,list(efel.getFeatureNames()))#
    efel_results_ephys = efel.getFeatureValues(trace_ephys_prop,list(efel.getFeatureNames()))#
    return efel_results_inh

def not_necessary_for_program_completion(DMTNMLO):
    '''
    Synopsis:
       Not necessary for feature extraction pipe line.
       More of a unit test.
    '''
    standard_current = DMTNMLO.model.nmldb_model.get_druckmann2013_standard_current()
    strong_current = DMTNMLO.model.nmldb_model.get_druckmann2013_strong_current()
    volt15 = DMTNMLO.model.nmldb_model.get_waveform_by_current(standard_current)
    volt30 = DMTNMLO.model.nmldb_model.get_waveform_by_current(strong_current)
    temp0 = np.mean(volt15)
    temp1 = np.mean(volt30)
    assert temp0 != temp1
    return
