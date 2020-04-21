#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np

import seaborn as sns
sns.set_context('notebook')
import matplotlib.pyplot as plt
plt.plot([0,1],[1,0])
#%matplotlib notebook
import matplotlib.pyplot as plt
#pip install natsort
from neuronunit.optimisation import dm_test_interoperable #import Interoperabe

#pip install igor
"""NeuronUnit module for interaction with the Blue Brain Project data."""
import os
import zipfile
import json
import IPython
import requests
import matplotlib.pyplot as plt
from neo.io import IgorIO
import natsort
from neuronunit.capabilities import spike_functions
import quantities as pq
import numpy as np
try:  # Python 3
    from io import BytesIO
    from urllib.request import urlopen, URLError
    MAJOR_VERSION = 3
except ImportError:  # Python 2
    from StringIO import StringIO
    from urllib2 import urlopen, URLError
    MAJOR_VERSION = 2


def is_bbp_up():
    """Check whether the BBP microcircuit portal is up."""
    url = "http://microcircuits.epfl.ch/released_data/B95_folder.zip"
    request = requests.get(url)
    return request.status_code == 200


def list_curated_data():
    """List all curated datasets as of July 1st, 2017.
    Includes those found at
    http://microcircuits.epfl.ch/#/article/article_4_eph
    """
    url = "http://microcircuits.epfl.ch/data/articles/article_4_eph.json"
    cells = []
    try:
        response = urlopen(url)
    except URLError:
        print ("Could not find list of curated data at %s" % URL)
    else:
        data = json.load(response)
        table = data['data_table']['table']['rows']
        for section in table:
            for row in section:
                if 'term' in row:
                    cell = row['term'].split(' ')[1]
                    cells.append(cell)
    return cells


def get_curated_data(data_id, sweeps=None):
    """Download curated data (Igor files) from the microcircuit portal.
    data_id: An ID number like the ones in 'list_curated_data()' that appears
    in http://microcircuits.epfl.ch/#/article/article_4_eph.
    """
    url = "http://microcircuits.epfl.ch/data/released_data/%s.zip" % data_id
    data = get_sweeps(url, sweeps=sweeps)
    return data


def get_uncurated_data(data_id, sweeps=None):
    """Download uncurated data (Igor files) from the microcircuit portal."""
    url = "http://microcircuits.epfl.ch/data/uncurated/%s_folder.zip" % data_id
    data = get_sweeps(url, sweeps=sweeps)
    return data


def get_sweeps(url, sweeps=None):
    """Get sweeps of data from the given URL."""
    print("Getting data from %s" % url)
    path = find_or_download_data(url)  # Base path for this data
    assert type(sweeps) in [type(None), list], "Sweeps must be None or a list."
    sweep_paths = list_sweeps(path)  # Available sweeps
    if sweeps is None:
        sweeps = sweep_paths
    else:
        sweeps = []
        for sweep_path in sweep_paths:
            if any([sweep_path.endswith(sweep for sweep in sweeps)]):
                sweeps.append(sweep_path)
        sweeps = set(sweeps)
    data = {sweep: open_data(sweep) for sweep in sweeps}
    return data


def find_or_download_data(url):
    """Find or download data from the given URL.
    Return a path to a local directory containing the unzipped data found
    at the provided url.  The zipped file will be downloaded and unzipped if
    the directory cannot be found.  The path to the directory is returned.
    """
    zipped = url.split('/')[-1]  # Name of zip file
    unzipped = zipped.split('.')[0]  # Name when unzipped
    z = None
    if not os.path.isdir(unzipped):  # If unzipped version not found
        if MAJOR_VERSION == 2:
            r = requests.get(url, stream=True)
            z = zipfile.ZipFile(StringIO(r.content))
        elif MAJOR_VERSION == 3:
            r = requests.get(url)
            z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall(unzipped)
    return unzipped


def list_sweeps(url, extension='.ibw'):
    """List all sweeps available in the file at the given URL."""
    path = find_or_download_data(url)  # Base path for this data
    sweeps = find_sweeps(path, extension=extension)
    return sweeps


def find_sweeps(path, extension='.ibw', depth=0):
    """Find sweeps available at the given path.
    Starting from 'path', recursively searches subdirectories and returns
    full paths to all files ending with 'extension'.
    """
    sweeps = []
    items = os.listdir(path)
    for item in items:
        new_path = os.path.join(path, item)
        if os.path.isdir(new_path):
            sweeps += find_sweeps(new_path, extension=extension, depth=depth+1)
        if os.path.isfile(new_path) and item.endswith(extension):
            sweeps += [new_path]
    return sweeps


def open_data(path):
    """Take a 'path' to an .ibw file and returns a neo.core.AnalogSignal."""
    igor_io = IgorIO(filename=path)
    analog_signal = igor_io.read_analogsignal()
    return analog_signal


def plot_data(signal,current):
    """Plot the data in a neo.core.AnalogSignal."""
    plt.clf()
    plt.plot(signal.times, signal)
    plt.xlabel(signal.sampling_period.dimensionality)
    plt.ylabel(signal.dimensionality)
    plt.title(np.max(current))
    plt.show()

threshes = []

def find_nearest(array,value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return (array[idx],idx)
def plot_rheobase(data):
    currents = {}
    vms = {}
    rekey3 = {}
    rekey0 = {}
    for k,v in data.items():
        if "Ch3" in k:
            rekey3[k] = v
        if "Ch0" in k:
            rekey0[k] = v
    vms = natsort.humansorted([(k,v) for k,v in rekey3.items()])
    injections = natsort.humansorted([(k,v) for k,v in rekey0.items()])
    threshes = []
    for j,(k,v) in enumerate(vms):
        threshold = spike_functions.threshold_detection(v)
        threshes.append(len(threshold))
    for j,(k,v) in enumerate(vms):
        threshold = spike_functions.threshold_detection(v)
        if len(threshold) == np.min(threshes) and np.max(v.times)>2.8*pq.s:

            target = j
            plot_data(vms[target][1],injections[target][1])
import quantities as pq
def like_plot_rheobase(data):
    currents = {}
    vms = {}
    rekey3 = {}
    rekey0 = {}
    for k,v in data.items():
        if "Ch3" in k:
            rekey3[k] = v
        if "Ch0" in k:
            rekey0[k] = v
    vms = natsort.humansorted([(k,v) for k,v in rekey3.items()])
    injections = natsort.humansorted([(k,v) for k,v in rekey0.items()])
    comprehendable = [np.max(ind[1]) for ind in injections if ind[1].units==pq.pA]
    print(comprehendable)
    threshes = []
    
    for _,(_,v) in enumerate(vms):
        threshold = spike_functions.threshold_detection(v)
        if np.max(v.times)>2.8*pq.s and len(threshold):            
            threshes.append(len(threshold))
        
    for j,(k,v) in enumerate(vms):
        threshold = spike_functions.threshold_detection(v)
        if len(threshold) == np.min(threshes) and np.max(v.times)>2.8*pq.s:
            rheobase_vm = v
            model = StaticModel(rheobase_vm)
            model.inj = []
            one_five = float(comprehendable[j])*1.5
            three = float(comprehendable[j])*3.0

            model.druckmann2013_standard_current = one_five
            model.druckmann2013_strong_current = three
            one_five = find_nearest(comprehendable,one_five)
            model.rheobase_current = comprehendable[j]
            model.vm15 = vms[one_five[1]]
            three = find_nearest(comprehendable,three)
            model.vm30 = vms[three[1]]
            return model, one_five,three




# In[54]:


from neuronunit.models import StaticModel
def get_15_30(data):
    currents = {}
    vms = {}
    rekey3 = {} # Membrane Potentials
    rekey0 = {} # Current injections
    for k,v in data.items():
        if "Ch3" in k:
            rekey3[k] = v
        if "Ch0" in k:
            rekey0[k] = v
    vms = natsort.humansorted([(k,v) for k,v in rekey3.items()])
    injections = natsort.humansorted([(k,v) for k,v in rekey0.items()])
    inj_list = [i for i in injections]

    threshes = []
    for j,(k,v) in enumerate(vms):
        threshold = spike_functions.threshold_detection(v)
        threshes.append(len(threshold))
    for j,(k,v) in enumerate(vms):
        threshold = spike_functions.threshold_detection(v)
        if len(threshold) == np.min(threshes) and np.max(v.times)>2.8*pq.s:
            target_idx = j
            rheobase = injections[target_idx][1]
            vmrh = (vms[target_idx][1],rheobase)
            model = StaticModel(vms[target_idx][1])
            plot_rheobase(vmrh)
            return (model,inj_list)


#injections
models = pickle.load(open('models.p','rb'))    

from neuronunit.optimisation.get_three_feature_sets_from_nml_db import three_feature_sets_on_static_models
    
np.shape(models)    
models[2][-1]


# In[65]:


import pickle

try:
    data = pickle.load(open('models.p','rb'))    
except:
    data_ids = list_curated_data()[0:5]
    tuples = []
    models = []
    results = []
    for di in data_ids:

        data = get_curated_data(di)
        out = like_plot_rheobase(data)        
        if out is not None:
            models.append([di,out[0]])

    pickle.dump(models,open('models.p','wb'))     
for mod in models:   
    model = None
    model = mod[1]
    model.name = mod[0]
    model.vm30 = model.vm30[1]
    model.vm15 = model.vm15[1]
    
    #import pdb
    #pdb.set_trace()
    
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
    from neuronunit.optimisation.get_three_feature_sets_from_nml_db import allen_format
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
out_dic = {'model_id':model.name,'model_information':'allen_data','efel_15':efel_15,'efel_30':efel_30,'dm':dm_test_features,'allen_15':all_allen_features15,'allen_30':all_allen_features30}
results.append(out_dic)
print(results)
import pdb
pdb.set_trace()

# In[66]:


import pickle

try:
    assert 1==2
    data = pickle.load(open('content.p','rb'))    
except:
    data_ids = list_curated_data()[0:5]
    tuples = []
    models = []
    for di in data_ids:

        data = get_curated_data(di)
        out = like_plot_rheobase(data)        
        if out is not None:
            models.append([di,out[0]])
    pickle.dump(models,open('models.p','wb'))        
