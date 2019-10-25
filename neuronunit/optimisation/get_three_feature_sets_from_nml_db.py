
##
#  The main method that does the aligned feature extraction is down the bottom.
# Two thirds of this file, it is called
# def three_feature_sets_on_static_models
##

##
# I build the docker image with the name russelljarvis/efel_allen_dm.
# meaning that the command
# docker pull russelljarvis/efel_allen_dm should work
# This uses the docker file in this directory.
# I build it with the name russelljarvis/efel_allen_dm.
# and launch it with this alias.
# alias efel='cd /home/russell/outside/neuronunit; sudo docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/russell/outside/neuronunit:/home/jovyan/neuronunit -v /home/russell/Dropbox\ \(ASU\)/AllenDruckmanData:/home/jovyan/work/allendata russelljarvis/efel_allen_dm /bin/bash'
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
import efel
from types import MethodType
import quantities as pq
import pdb

from collections import Iterable, OrderedDict

import numpy as np
import efel
import pickle
from allensdk.ephys.extract_cell_features import extract_cell_features
import pandas as pd
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from neuromldb import NeuroMLDBStaticModel

from neuronunit.optimisation import dm_test_interoperable #import Interoperabe
from dask import bag as db
import glob

import copy
from elephant.spike_train_generation import threshold_detection
from allensdk.ephys.ephys_extractor import EphysSweepSetFeatureExtractor

import csv


import dask
import dask.array as da



def crawl_ids(url):
    ''' move to aibs '''
    all_data = requests.get(url)
    all_data = json.loads(all_data.text)
    Model_IDs = []
    for d in all_data:
        Model_ID = str(d['Model_ID'])
        Model_IDs.append(Model_ID)
    return Model_IDs

list_to_get =[ str('https://www.neuroml-db.org/api/search?q=traub'),
    str('https://www.neuroml-db.org/api/search?q=markram'),
    str('https://www.neuroml-db.org/api/search?q=Gouwens') ]

def get_all_cortical_cells(list_to_get):
    model_ids = {}
    for url in list_to_get:
        Model_IDs = crawl_ids(url)
        parts = url.split('?q=')
        try:
            model_ids[parts[1]].append(Model_IDs)
        except:
            model_ids[parts[1]] = []
            model_ids[parts[1]].append(Model_IDs)
    with open('cortical_cells_list.p','wb') as f:
        pickle.dump(model_ids,f)

    return model_ids



def get_waveform_current_amplitude(waveform):
    return float(waveform["Waveform_Label"].replace(" nA", "")) * pq.nA


def get_static_models(cell_id):
    """
    Inputs: NML-DB cell ids, a method designed to be called inside an iteration loop.

    Synpopsis: given a NML-DB id, query nml-db, create a NMLDB static model based on wave forms
        obtained for that NML-ID.
        get mainly just waveforms, and current injection values relevant to performing druckman tests
        as well as a rheobase value for good measure.
        Update the NML-DB model objects attributes, with all the waveform data/injection values obtained for the appropriate cell IDself.
    """


    url = str("https://www.neuroml-db.org/api/model?id=")+cell_id
    #import pdb
    #pdb.set_trace()
    model_contents = requests.get(url)
    model_contents = json.loads(model_contents.text)
    model = NeuroMLDBStaticModel(cell_id)

    wlist = model_contents['waveform_list']
    long_squares = [ w for w in wlist if w['Protocol_ID'] == 'LONG_SQUARE' ]
    applied_current_injections = [ w for w in wlist if w["Protocol_ID"] == "LONG_SQUARE" and w["Variable_Name"] == "Current" ]
    currents = [ w for w in wlist if w["Protocol_ID"] == "LONG_SQUARE" and w["Variable_Name"] == "Voltage" ]
    in_current_filter = [ w for w in wlist if w["Protocol_ID"] == "SQUARE" and w["Variable_Name"] == "Voltage" ]
    rheobases = []
    for wl in long_squares:
        wid = wl['ID']
        url = str("https://neuroml-db.org/api/waveform?id=")+str(wid)
        waves = requests.get(url)
        temp = json.loads(waves.text)
        if temp['Spikes'] >= 1:
            rheobases.append(temp)
    if len(rheobases) == 0:
        return None

    in_current = []
    for check in in_current_filter:
        amp = get_waveform_current_amplitude(check)
        if amp < 0 * pq.nA:
            in_current.append(amp)
    rheobase_current = get_waveform_current_amplitude(rheobases[0])
    druckmann2013_standard_current = get_waveform_current_amplitude(currents[-2])
    druckmann2013_strong_current = get_waveform_current_amplitude(currents[-1])
    druckmann2013_input_resistance_currents = in_current
    model.waveforms = wlist
    model.protocol = {}
    model.protocol['Time_Start'] = currents[-2]['Time_Start']
    model.protocol['Time_End'] = currents[-2]['Time_End']
    model.inh_protocol = {}
    model.inh_protocol['Time_End'] = in_current_filter[0]['Time_End']
    model.inh_protocol['Time_Start'] = in_current_filter[0]['Time_Start']
    model.druckmann2013_input_resistance_currents = druckmann2013_input_resistance_currents

    model.rheobase_current = rheobase_current
    model.druckmann2013_standard_current = druckmann2013_standard_current
    model.druckmann2013_strong_current = druckmann2013_strong_current
    current = {}
    current['amplitude'] = rheobase_current
    model.vm_rheobase = model.inject_square_current(current)
    current['amplitude'] = druckmann2013_standard_current
    model.vm15 = model.inject_square_current(current)
    current['amplitude'] = druckmann2013_strong_current
    model.vm30 = model.inject_square_current(current)
    current['amplitude'] = druckmann2013_input_resistance_currents[0]
    model.vminh =  model.inject_square_current(current)

    return model

def allen_format(volts,times,optional_vm=None):
    '''
    Synposis:
        At its most fundamental level, AllenSDK still calls a single trace a sweep.
        In otherwords there are no single traces, but there are sweeps of size 1.
        This is a bit like wrapping unitary objects in iterable containers like [times].

    inputs:
        np.arrays of time series: Specifically a time recording vector, and a membrane potential recording.
        in floats probably with units striped away
    outputs:
        a data frame of Allen features, a very big dump of features as they pertain to each spike in a train.

        to get a managable data digest
        we out put features from the middle spike of a spike train.

    '''
    if optional_vm is not None:


        spike_train = threshold_detection(optional_vm, threshold=0)

    ext = EphysSweepSetFeatureExtractor([times],[volts])

    ext.process_spikes()
    swp = ext.sweeps()[0]
    spikes = swp.spikes()
    if len(spikes)==0:
        return (None,None)

    meaned_features_1 = {}
    skeys = [ skey for skey in spikes[0].keys() ]
    for sk in skeys:
        if str('isi_type') not in sk:
            meaned_features_1[sk] = np.mean([ i[sk] for i in spikes if type(i) is not type(str(''))] )
    allen_features = {}
    meaned_features_overspikes = {}
    for s in swp.sweep_feature_keys():# print(swp.sweep_feature(s))
        if str('isi_type') not in s:
            allen_features[s] = swp.sweep_feature(s)

    allen_features.update(meaned_features_1)
    return meaned_features_overspikes, allen_features

    '''
    for sk in skeys:
        if str('isi_type') not in sk:
            meaned_features_1[sk] = np.mean([ i[sk] for i in spikes if type(i) is not type(str(''))] )
    allen_features = {}
    meaned_features_overspikes = {}
    for s in swp.sweep_feature_keys():# print(swp.sweep_feature(s))

        if str('isi_type') not in s:
            allen_features[s] = swp.sweep_feature(s)
            try:
                feature = swp.sweep_feature(s)
                if isinstance(feature, Iterable):
                    meaned_features_overspikes[s] = np.mean([i for i in feature if type(i) is not type(str(''))])
                else:
                    meaned_features_overspikes[s] = feature

            except:
                meaned_features_overspikes[s] = None #np.mean([i for i in swp.spike_feature(s) if type(i) is not type(str(''))])
                print(meaned_features_overspikes)
    for s in swp.sweep_feature_keys():
        print(swp.sweep_feature(s))


    meaned_features_overspikes.update(meaned_features_1)
    '''

#from elephant.spike_train_generation import threshold_detection
from neuronunit.optimisation.optimization_management import DataTC
def three_feature_sets_on_static_models(model,unit_test = False, challenging=False):
    '''
    Conventions:
        variables ending with 15 refer to 1.5 current injection protocols.
        variables ending with 30 refer to 3.0 current injection protocols.
    Inputs:
        NML-DB models, a method designed to be called inside an iteration loop, where a list of
        models is iterated over, and on each iteration a new model is supplied to this method.

    Outputs:
        A dictionary of dataframes, for features sought according to: Druckman, EFEL, AllenSDK

    '''

    ##
    # wrangle data in preperation for computing
    # Allen Features
    ##

    #import pdb; pdb.set_trace()
    if type(model) is type(DataTC()):
        temp = model.protocols
        model = model.dtc_to_model()
        model.inj = temp
    if not hasattr(model,'vm30'):
        # assume not static model anymore
        current = model.inj[0]['injected_square_current']
        current['amplitude']= model.rheobase*3.0
        model.inject_square_current(current)
        vm = model.get_membrane_potential()
        model.vm30 = vm
        model.druckmann2013_strong_current = current

        current['amplitude']= model.rheobase*1.5
        model.protocol = {}
        model.protocol['Time_Start'] = current['delay']
        model.inject_square_current(current)
        vm = model.get_membrane_potential()
        model.vm15 = vm
        model.druckmann2013_standard_current = current


    times = np.array([float(t) for t in model.vm30.times])
    volts = np.array([float(v) for v in model.vm30])
    try:
        import asciiplotlib as apl
        fig = apl.figure()
        fig.plot(times, volts, label="V_{m} (mV), versus time (ms)", width=100, height=80)
        fig.show()
    except:
        pass
    ##
    # Allen Features
    ##
    #frame_shape,frame_dynamics,per_spike_info, meaned_features_overspikes
    all_allen_features30, allen_features30 = allen_format(volts,times,optional_vm=model.vm30)
    #if frame30 is not None:
    #    frame30['protocol'] = 3.0
    ##

    # wrangle data in preperation for computing
    # Allen Features
    ##
    times = np.array([float(t) for t in model.vm15.times])
    volts = np.array([float(v) for v in model.vm15])

    ##
    # Allen Features
    ##

    all_allen_features15, allen_features15 = allen_format(volts,times,optional_vm=model.vm15)
    ##
    # Get Druckman features, this is mainly handled in external files.
    ##
    #if model.ir_currents
    DMTNMLO = dm_test_interoperable.DMTNMLO()

    if hasattr(model,'druckmann2013_input_resistance_currents') and not hasattr(model,'allen'):
        DMTNMLO.test_setup(None,None,model= model)

    else:
        DMTNMLO.test_setup(None,None,model= model,ir_current_limited=True)
    dm_test_features = DMTNMLO.runTest()
    print(dm_test_features)
    import pdb; pdb.set_trace()
    ##
    # Wrangle data to prepare for EFEL feature calculation.
    ##
    trace3 = {}
    trace3['T'] = [ float(t) for t in model.vm30.times.rescale('ms') ]
    trace3['V'] = [ float(v) for v in model.vm30.magnitude]#temp_vm
    trace3['stimulus_current'] = [ model.druckmann2013_strong_current ]
    if not hasattr(model,'allen'):
        trace3['stim_end'] = [ trace3['T'][-1] ]
        trace3['stim_start'] = [ float(model.protocol['Time_Start']) ]

    else:
        trace3['stim_end'] = [ float(model.protocol['Time_End'])*1000.0 ]
        trace3['stim_start'] = [ float(model.protocol['Time_Start'])*1000.0 ]

    traces3 = [trace3]# Now we pass 'traces' to the efel and ask it to calculate the feature# values

    trace15 = {}
    trace15['T'] = [ float(t) for t in model.vm15.times.rescale('ms') ]
    trace15['V'] = [ float(v) for v in model.vm15.magnitude ]#temp_vm

    if not hasattr(model,'allen'):
        trace15['stim_end'] = [ trace15['T'][-1] ]
        trace15['stim_start'] = [ float(model.protocol['Time_Start']) ]
    else:
        trace15['stim_end'] = [ float(model.protocol['Time_End'])*1000.0 ]
        trace15['stim_start'] = [ float(model.protocol['Time_Start'])*1000.0 ]
    trace15['stimulus_current'] = [ model.druckmann2013_standard_current ]
    trace15['stim_end'] = [ trace15['T'][-1] ]
    traces15 = [trace15]# Now we pass 'traces' to the efel and ask it to calculate the feature# values

    ##
    # Compute
    # EFEL features (HBP)
    ##
    efel.reset()

    if len(threshold_detection(model.vm15, threshold=0)):
        #pass
        threshold = float(np.max(model.vm15.magnitude)-0.5*np.abs(np.std(model.vm15.magnitude)))

        print(len(threshold_detection(model.vm15, threshold=threshold)))
        print(threshold,'threshold', np.max(model.vm15.magnitude),np.min(model.vm15.magnitude))

        #efel_15 = efel.getMeanFeatureValues(traces15,list(efel.getFeatureNames()))#
    else:
        threshold = float(np.max(model.vm15.magnitude)-0.2*np.abs(np.std(model.vm15.magnitude)))

        efel.setThreshold(threshold)
        print(len(threshold_detection(model.vm15, threshold=threshold)))
        print(threshold,'threshold', np.max(model.vm15.magnitude))

    if np.min(model.vm15.magnitude)<0:
        try:
            efel_15 = efel.getMeanFeatureValues(traces15,list(efel.getFeatureNames()))
        except:
            efel_15 = None

    else:
        efel_15 = None
    efel.reset()

    if len(threshold_detection(model.vm30, threshold=0)):
        threshold = float(np.max(model.vm30.magnitude)-0.5*np.abs(np.std(model.vm30.magnitude)))

        print(len(threshold_detection(model.vm30, threshold=threshold)))
        print(threshold,'threshold', np.max(model.vm30.magnitude),np.min(model.vm30.magnitude))

    #efel_30 = efel.getMeanFeatureValues(traces3,list(efel.getFeatureNames()))


    else:
        threshold = float(np.max(model.vm30.magnitude)-0.2*np.abs(np.std(model.vm30.magnitude)))
        efel.setThreshold(threshold)
        print(len(threshold_detection(model.vm15, threshold=threshold)))
        print(threshold,'threshold', np.max(model.vm15.magnitude))

    if np.min(model.vm30.magnitude)<0:
        #efel_30 = efel.getMeanFeatureValues(traces3,list(efel.getFeatureNames()))
        try:
            efel_30 = efel.getMeanFeatureValues(traces3,list(efel.getFeatureNames()))
        except:
            efel_30 = None

    else:
        efel_30 = None

    efel.reset()

    if hasattr(model,'information'):
        return {'model_id':model.name,'model_information':model.information,'efel_15':efel_15,'efel_30':efel_30,'dm':dm_test_features,'allen_15':all_allen_features15,'allen_30':all_allen_features30}
    else:
        return {'model_id':model.name,'model_information':'allen_data','efel_15':efel_15,'efel_30':efel_30,'dm':dm_test_features,'allen_15':all_allen_features15,'allen_30':all_allen_features30}



def nmlefel(nml_data,onefive=True):
    indexs = [ nml['model_id'] for nml in nml_data ]
    if onefive:
        rows = [ nml['efel_15'] for nml in nml_data ]
    else:
        rows = [ nml['efel_30'] for nml in nml_data ]
    list_of_dicts = []
    for r in rows:
        if r is None:
            # write in a data frame entry for a non spiking model
            temp = {}
            #print(rows[0])
            temp = copy.copy(rows[0][0])
            for k,v in temp.items():
                temp[k] = None
            list_of_dicts.append(temp)
        else:
            list_of_dicts.append(r[0])

    #list_of_dicts.append(r[0])
    df = pd.DataFrame(list_of_dicts,index=indexs)
    return df

def nmldm(nml_data):
    indexs = [ nml['model_id'] for nml in nml_data ]
    rows = [ nml['dm'] for nml in nml_data ]
    list_of_dicts = []
    #for r in rows:
    #    list_of_dicts.append(r)
    for r in rows:
        if r is None:
            # write in a data frame entry for a non spiking model
            temp = {}
            temp = copy.copy(rows[0])
            for k,v in temp.items():
                temp[k] = None
            list_of_dicts.append(temp)
        else:
            list_of_dicts.append(r)

    df = pd.DataFrame(list_of_dicts,index=indexs)
    return df
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
def nmlallen(nml_data,onefive=True):
    indexs = [ nml['model_id'] for nml in nml_data ]
    if onefive:
        rows = [ nml['allen_15'] for nml in nml_data ]
    else:
        rows = [ nml['allen_30'] for nml in nml_data ]
    #rows = [ nml['allen'] for nml in nml_data if len(nml)]
    list_of_dicts = []
    for r in rows:
        if r is None:
            # write in a data frame entry for a non spiking model
            temp = {}
            temp = copy.copy(rows[0])
            for k,v in temp.items():
                temp[k] = None
            list_of_dicts.append(temp)
        else:
            list_of_dicts.append(r)


    df = pd.DataFrame(list_of_dicts,index=indexs)
    return df
    #print(df)

def giant_frame(allen_analysis,nml_data,onefive=True,other_dir=None):
    dfe = nmlefel(nml_data,onefive)
    dfd = nmldm(nml_data)
    dfa = nmlallen(nml_data,onefive)

    dfea = nmlefel(allen_analysis,onefive)
    dfda = nmldm(allen_analysis)
    dfaa = nmlallen(allen_analysis,onefive)

    dfa = dfa.append(dfaa)
    dfe = dfe.append(dfea)
    dfd = dfd.append(dfda)
    #pdb.set_trace()

    merged = pd.merge(dfe, dfd, right_index=True, left_index=True)
    df = pd.merge(merged, dfa, right_index=True, left_index=True)
    if other_dir is None:
        if onefive:
            with open(str('onefive_df.pkl'),'wb') as f:
                pickle.dump(df,f)
        else:
            with open(str('three_df.pkl'),'wb') as f:
                pickle.dump(df,f)

    else:
        if onefive:
            with open(str(other_dir)+str('/onefive_df.pkl'),'wb') as f:
                pickle.dump(df,f)
        else:
            with open(str(other_dir)+str('/three_df.pkl'),'wb') as f:
                pickle.dump(df,f)
    # A function to convert all cells containing array (or other things) into floats.
    def f(x):
        try:
            return np.mean(x)
        except:
            try:
                return np.mean(x['pred'])
            except:
                print(x)
                raise e

    # Apply this function to each dataframe in order to convert all cells into floats.
    # Also call fillna() first to impute missing values with 0's.
    df_15x = df_15x.fillna(0).applymap(f)
    df_30x = df_30x.fillna(0).applymap(f)


    df = df_15x.join(df_30x, lsuffix='_1.5x', rsuffix='_3.0x')
    # Impute remaining missing values with 0.
    df = df.fillna(0)

    # Turn all features into Normal(0,1) variables
    # Important since features all have different scales

    df[:] = ss.fit_transform(df.values)
    model_idx = [idx for idx in df.index.values if type(idx)==str]
    model_no_trans_df = df[df.index.isin(model_idx)]
    model_no_trans_df.index.name = 'Cell_ID'
    model_df = model_no_trans_df.copy()
    model_df.index.name = 'Cell_ID'

    # make experiment dataframe
    experiment_idx = [idx for idx in df.index.values if type(idx)==int]

    experiment_no_trans_df = df[df.index.isin(experiment_idx)]

    experiment_df = experiment_no_trans_df.copy()
    print(len(experiment_df))
    return final


def recoverable_interuptable_batch_process():
    '''
    Synposis:
        slower serial mode but debug friendly and simple
        Mass download all the NML model waveforms for all cortical regions
        And perform three types of feature extraction on resulting waveforms.

    Inputs: None
    Outputs: None in namespace, yet, lots of data written to pickle.
    '''
    mid =  pickle.load(open('cortical_NML_IDs/cortical_cells_list.p','rb'))
    mid = mid[1:-1]


    path_name = str('three_feature_folder')
    try:
        os.mkdir(path_name)
    except:
        print('directory already made :)')
    try:
        ##
        # This is the index in the list to the last NML-DB model that was analyzed
        # this index is stored to facilitate recovery from interruption
        ##
        assert 1==2
        with open('last_index.p','rb') as f:
            index = pickle.load(f)
    except:
        index = 0
    until_done = len(mid[index:-1])
    cnt = 0
    ##
    # Do the batch job, with the background assumption that some models may
    # have already been run and cached.
    ##
    while cnt <until_done-1:
        for i,mid_ in enumerate(mid[index:-1]):
            until_done = len(mid[index:-1])
            model = get_static_models(mid_[1])
            if type(model) is not type(None):

                model.name = None
                model.name = str(mid_[1])
                model.information = mid_
                three_feature_sets = three_feature_sets_on_static_models(model)
                with open(str(path_name)+str('/')+str(mid_[1])+'.p','wb') as f:
                    pickle.dump(three_feature_sets,f)
            with open('last_index.p','wb') as f:
                pickle.dump(i,f)
            cnt+=1

#import numpy as np

def mid_to_model(mid_):
    try:
        os.mkdir(str('models'))
    except:
        pass
    path = str('models')+str('/')+str(mid_[1])+'.p'
    if os.path.exists(path):
        return

    model = get_static_models(mid_[1])
    if type(model) is not type(None):
        model.name = None
        model.name = str(mid_[1])
        model.information = mid_
        with open(str('models')+str('/')+str(mid_[1])+'.p','wb') as f:
            pickle.dump(model,f)
    return

def faster_make_model_and_cache():
    '''
    Synposis:

        Mass download all the NML model waveforms for all cortical regions
        And perform three types of feature extraction on resulting waveforms.

    Inputs: None
    Outputs: None in namespace, yet, lots of data written to pickle.
    '''
    try:

        mid =  pickle.load(open('cortical_NML_IDs/cortical_cells_list.p','rb'))
    except:
        with open('cortical_tags.csv','rt') as csvfile:
            reader = csv.reader(csvfile,delimiter=',',quotechar='|')
            mid = [row for row in reader]

            with open('cortical_NML_IDs/cortical_cells_list.p','wb') as f :
                pickle.dump(mid,f)
    size = len(mid[1:-1])
    mid =( m for m in mid[1:-1] ) # mid needs to be generator to not cause memory problems.
    path_name = str('models')
    try:
        os.mkdir(path_name)
    except:
        print('model directory already made :)')

    ##
    # Do the batch model download.
    ##
    lazy_arrays = [dask.delayed(mid_to_model)(m) for m in mid]
    [ l.compute() for l in lazy_arrays ]

    #[ l.compute() for l in lazy_arrays[int(size/4.0):int(size/2.0)] ]
    #[ l.compute() for l in lazy_arrays[int(size/2.0):3*int(size/4.0) ] ]
    #[ l.compute() for l in lazy_arrays[int(3.0*size/4.0):-1] ]


    #mid_bag = db.from_sequence(mid,npartitions=8)
    #list(mid_bag.map(mid_to_model).compute())

def model_analysis(model):
    if type(model) is not type(None):
        three_feature_sets = three_feature_sets_on_static_models(model)
        try:
            assert type(model.name) is not None
            with open(str('three_feature_folder')+str('/')+str(model.name)+'.p','wb') as f:
                pickle.dump(three_feature_sets,f)
        except:
            print('big error')
            import pdb; pdb.set_trace()
    return


def write_data_to_df():
    '''
    vertically and horizontally merge data frames
    '''
    file_paths = glob.glob("three_feature_folder/*.p")
    nml_data = []

    for f in file_paths:
        nml_data.append(pickle.load(open(f,'rb')))
    file_paths = glob.glob("allen_three_feature_folder/*.p")
    allen_analysis = []
    for f in file_paths:
        allen_analysis.append(pickle.load(open(f,'rb')))
    merged = runnable_nml.giant_frame(allen_analysis,nml_data,onefive=True,other_dir=os.getcwd())
    merged = runnable_nml.giant_frame(allen_analysis,nml_data,onefive=False,other_dir=os.getcwd())


    try:
        os.mkdir('just_data_frames')
    except:
        pass
    os.system('mv onefive_df.pkl just_data_frames/')
    os.system('mv three_df.pkl just_data_frames/')

def analyze_models_from_cache(models):
    data_bag = db.from_sequence(models,npartitions=8)
    analysis = list(data_bag.map(model_analysis).compute())


'''
    models = (pickle.load(open(f,'rb')) for f in paths)
    models = (m for m in models if m.vm30 is not None)
    import pdb; pdb.set_trace()
    m = [ m for m in models if not os.path.exists(str('three_feature_folder')+str('/')+str(m.name)+str('.p')) ]
    models = [ m for m in models if not os.path.exists(str('allen_three_feature_folder')+str('/')+str(m.name)+str('.p')) ]
    models.extend(m)
    cnt = len(models)
'''
'''
def analyze_models_from_cache(file_paths):
    models = [pickle.load(open(f,'rb')) for f in file_paths ]
    viable_paths = [ m for m in models if not os.path.exists(str('three_feature_folder')+str('/')+str(m.name)+str('.p')) ]


    models = (m for m in models if m.vm30 is not None)
    models = [ m for m in models if not os.path.exists(str('three_feature_folder')+str('/')+str(m.name)+str('.p')) ]


    file_paths = viable_paths
    data_bag = db.from_sequence((m for m in models[0:int(len(file_paths)/4.0)]))
    _ = list(data_bag.map(model_analysis).compute())
    data_bag = db.from_sequence((m for m in models[int(len(file_paths)/4.0)+1:int(len(file_paths)/2.0)]))
    _ = list(data_bag.map(model_analysis).compute())
    data_bag = db.from_sequence((m for m in models[int(len(file_paths)/2.0)+1:3*int(len(file_paths)/4.0)]))
    _ = list(data_bag.map(model_analysis).compute())
    data_bag = db.from_sequence((m for m in models[int(len(file_paths)/4.0):-1]))
    _ = list(data_bag.map(model_analysis).compute())
    lazy_arrays = [dask.delayed(model_analysis)(m) for m in models]
    _ = [ l.compute() for l in lazy_arrays ]

    #except:
'''

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
