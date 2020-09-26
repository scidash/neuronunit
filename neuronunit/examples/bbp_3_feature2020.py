#!/usr/bin/env python
# coding: utf-8

import seaborn as sns
sns.set_context('notebook')
import matplotlib.pyplot as plt
plt.plot([0,1],[1,0])
#pip install natsort

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
    
    



import pickle
import pandas as pd
from neuronunit.optimisation.get_three_feature_sets_from_nml_db import three_feature_sets_on_static_models
import pickle

models_hbp = pickle.load(open("hbp_data2.p","rb"))
bbp = pd.DataFrame([m.everything for m in models_hbp if m is not None and hasattr(m,'everything')])
models_hbp = pickle.load(open("hbp_data.p","rb"))
for i,m in enumerate(models_hbp):
    if hasattr(m,'vm30') and m is not None and i!=190 and i!=86 and str(m)!=str('A87') and i!=189:
        print('crashed at index',i,m)#,m['model_information'])

        m.features = three_feature_sets_on_static_models(m,bbp=True)
        everything = {}
        if m.features['efel_15'] is not None and len(m.features['allen_30']) and m.features['efel_30'] is not None:
            everything.update(m.features['efel_15'][0])
            everything.update(m.features['efel_30'][0])
            everything.update(m.features['allen_30'])
        everything.update(m.features['allen_15'])
        everything.update(m.features['dm'])

        m.everything = everything
        pickle.dump(models_hbp,open("hbp_data3.p","wb"))
print(len(models_hbp))

import pickle
import pandas as pd
models_hbp = pickle.load(open("hbp_data3.p","rb"))
bbp = pd.DataFrame([m.everything for m in models_hbp if m is not None and hasattr(m,'everything')])
bbp



bbp = pd.DataFrame([m.everything for m in models_hbp])


features


# In[ ]:


#from neuronunit import bbp
#list_curated_data()


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
            model.protocol = None
            model.inj = []
            one_five = float(comprehendable[j])*1.5
            three = float(comprehendable[j])*3.0
            one_five = find_nearest(comprehendable,one_five)
            model.rheobase_current = comprehendable[j]
            model.vm15 = vms[one_five[1]]
            three = find_nearest(comprehendable,three)
            model.vm30 = vms[three[1]]
            return model, one_five,three




# In[ ]:


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

    #print(np.max([inj[1] for inj in injections ]))    
#get_15_30(data)    
    #np.closest()

#injections
import pickle
models = pickle.load(open('models.p','rb'))    

from neuronunit.optimisation.get_three_feature_sets_from_nml_db import three_feature_sets_on_static_models
'''
data_ids = list_curated_data()[0:5]
tuples = []
models = []
results = []
for di in data_ids:
    data = get_curated_data(di)
    plot_rheobase(data)
'''
from neo import AnalogSignal
for m in models:
    print(m)
    #print(AnalogSignal(m.vm30,)
    #print(m[0],m[1])
#    print(dir(m[0]))
#    print(type(m[0]))()
    #features = three_feature_sets_on_static_models(m)
    
#np.shape(models)    




import pickle

try:
    data = pickle.load(open('models.p','rb'))    
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


# In[ ]:


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
            features = three_feature_sets_on_static_models(models[-1])

        '''
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
        '''
        
    pickle.dump(models,open('models.p','wb'))        
        #out_tuple = get_15_30(data)    
        #tuples.append(out_tuple)


# In[ ]:


data = pickle.load(open('content.p','rb'))    
#data


# In[ ]:


data.keys


# In[ ]:


#plot_rheobase(data)
#print(out)
#out[1].
#efel_evaluation()
#injections[0]#.values()
#type(tuples[0][1][0#])

#plot_rheobase(data)


    #all_sweeps = list_sweeps(did)
    #break
models = pickle.load(open('models.p','rb'))    
#features = three_feature_sets_on_static_models(models[-1])
models[0].vm30[2]


# In[ ]:


out[0]


# In[ ]:


print(out[2])
one_five = float(out[2][0])*1.5
three = float(out[2][0])*3.0

one_five = find_nearest(out[2],one_five)
three = find_nearest(out[2],three)

#print([i[2] for i in out)
#len(out)
three


# In[ ]:


vms[1]


# In[ ]:





# In[ ]:





# In[ ]:


print(type(data))


# In[ ]:


data[list(data.keys())[1]].units


# In[ ]:


data[list(data.keys())[0]]


# In[ ]:


data[list(data.keys())[1]]


# In[ ]:





# In[ ]:


rekey3;


# In[ ]:


get_ipython().system('pwd')


# In[ ]:





# In[ ]:





# In[ ]:


help(natsort)


# In[ ]:


plot_rheobase(data)


# In[ ]:




