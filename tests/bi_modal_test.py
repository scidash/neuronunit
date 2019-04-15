import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import urllib.request, json

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

#from modality import calibrated_diptest, calibrated_bwtest, silverman_bwtest, hartigan_diptest, excess_mass_modes
#from modality.util import auto_interval
#import anchor
#https://www.neuroelectro.org/api/1/n/
purkinje ={"id": 18, "name": "Cerebellum Purkinje cell", "neuron_db_id": 271, "nlex_id": "sao471801888"}
fi_basket = {"id": 65, "name": "Dentate gyrus basket cell", "neuron_db_id": None, "nlex_id": "nlx_cell_100201"}
ca1_basket = {"id": 82, "name": "Hippocampus CA1 basket cell", "neuron_db_id": None, "nlex_id": "nlx_cell_091205"},
#{"id": 129, "name": "Olfactory bulb (main) mitral cell", "neuron_db_id": 267, "nlex_id": "nlx_anat_100201"}
## Draws a blank
# dg_basket = {"id": 65, "name": "Dentate gyrus basket cell", "neuron_db_id": null, "nlex_id": "nlx_cell_100201"}
##
pvis_cortex = {"id": 111, "name": "Neocortex pyramidal cell layer 5-6", "neuron_db_id": 265, "nlex_id": "nifext_50"}
#does not have rheobase
olf_mitral = {"id": 129, "name": "Olfactory bulb (main) mitral cell", "neuron_db_id": 267, "nlex_id": "nlx_anat_100201"}
#{"id": 129, "name": "Olfactory bulb (main) mitral cell", "neuron_db_id": 267, "nlex_id": "nlx_anat_100201"}
ca1_pyr = {"id": 85, "name": "Hippocampus CA1 pyramidal cell", "neuron_db_id": 258, "nlex_id": "sao830368389"}
pipe = [ ca1_basket, ca1_pyr, purkinje,  pvis_cortex,olf_mitral]

name_map ={}
name_map["Cerebellum Purkinje cell"] = "sao471801888"
name_map["Dentate gyrus basket cell"] = "nlx_cell_100201"

#name_map["Hippocampus CA1 basket cell"] = "nlx_cell_091205"
name_map["Neocortex pyramidal cell layer 5-6"] = "nifext_50"
name_map["Olfactory bulb (main) mitral cell"] = "nlx_anat_100201"
name_map["Hippocampus CA1 pyramidal cell"] = "sao830368389"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from neuronunit import neuroelectro
def neuroelectro_summary_observation(neuron,ontology):
    ephysprop_name = ''
    verbose = False

    reference_data = neuroelectro.NeuroElectroSummary(
        neuron = neuron, # Neuron type lookup using the NeuroLex ID.
        ephysprop = {'name': ontology['name']}, # Ephys property name in
        # NeuroElectro ontology.
    )
    reference_data.get_values() # Get and verify summary data
                                # from neuroelectro.org.
    return reference_data

def get_obs(pipe):
    with urllib.request.urlopen("https://neuroelectro.org/api/1/e/") as url:
        ontologies = json.loads(url.read().decode())
        #print(ontologies)
    #with urllib.request.urlopen("https://neuroelectro.org/api/1/e/") as url:
    #    ontologies = json.loads(url.read().decode())
    obs = []
    for p in pipe:
        for l in ontologies['objects']:
            try:
                obs.append(neuroelectro_summary_observation(p,l))
            except:
                pass
    return obs


def properties(pipe):
    #obs = get_obs(pipe)
    #import pdb; pdb.set_trace()
    neuron_values = {}
    ##
    # pipe = [ fi_basket, ca1_pyr, purkinje,  pvis_cortex,olf_mitral]
    ##
    for val in range(0,13):
        #import pdb; pdb.set_trace()
        for p in pipe:

            print(p["name"],"cell_name")
            url_to_get = str('https://neuroelectro.org/api/1/nedm/?nlex=')+str(p["nlex_id"])+str('&e=')+str(val)+str('&limit=100')
            with urllib.request.urlopen(url_to_get) as url:
                data = json.loads(url.read().decode())
            for objects in data['objects']:
                last_label = objects['ecm']['ref_text']
                #print(last_label)
            if len(data['objects']):
                p[last_label] = {}
            #p
            datax = [ objects['val'] for objects in data['objects'] ]
            datax = [ dx for dx in datax if type(dx) is not type(None) ]
            neuron_values[str(p["nlex_id"])] = {}
            neuron_values[str(p["nlex_id"])][val] = datax
            if len(datax):
                samples = [ n['n'] for n in data['objects'] ]
                samples = [ n for n in samples if type(n) is not type(None) ]
                sample_sizes = sum(samples)
                p[last_label]['values'] = datax
                p[last_label]['n'] = sample_sizes

                plt.clf()#hartigan_diptest(self.data)
                #n = 1000;
                #b = n//10;
                #h = np.histogram(datax,bins=b)
                #h = np.vstack((0.5*(h[1][:-1]+h[1][1:]),h[0])).T  # because h[0] and h[1] have different sizes.
                try:
                    datax = np.array(datax)
                    indexes = peakutils.indexes(datax, thres=0.02/max(cb), min_dist=100)
                    interpolatedIndexes = peakutils.interpolate(range(0, len(datax)), datax, ind=indexes)
                    kmeans = KMeans(n_clusters=2).fit(datax.reshape(len(datax),1))
                    print(kmeans.cluster_centers_, 'clusert centre')
                except:
                    pass
                try:

                    #plt.scatter(datax)
                    sns.distplot(sorted(datax), color="skyblue", label=str(last_label)+str(p["nlex_id"])+str(val))
                    plt.xlabel(str('sample_size: ')+str(sample_sizes))
                    plt.legend(loc="upper left")
                except:
                    plt.xlabel(str('sample_size: ')+str(sample_sizes))
                    plt.hist(sorted(datax))      #use this to draw histogram of your data
                    #plt.scatter(datax)
                try:
                    plt.savefig(str(last_label)+str(p["nlex_id"])+str(val)+str('.png'))
                except:
                    pass
    with open('tests_data.p','wb') as f:
        pickle.dump(pipe,f)
    return neuron_values

'''
{' RMP, mV ': 3,
 'AP current threshold (pA)': 8,
 'AP half width, ms': 6,
 'APAMP(mV)\n                           ': 5,
 'Access resistance (MΩ)': 12,
 'Afterspike hyperpolarization time ratio (AHPtr)': 10,
 'Amplitude, mV': 5,
 'Cm, pF': 1,
 'Duration (ms)*': 6,
 'Duration, s': 10,
 'Firing Frequency at 500 pA (Hz)': 9,
 'Frequency, Hz': 9,
 'Input resistance (MΩ)': 2,
 'Membrane capacitance (pF)': 1,
 'Membrane capacitance (pF)*': 1,
 'Membrane potential (mV)': 3,
 'RMP (mV)': 3,
 'Rheobase, pA': 8,
 'Rin (MΩ)': 2,
 'Somatic diameter, dsoma': 11,
 'Threshold (mV)': 7,
 'Threshold, mV': 7,
 'Time Constant (ms)': 4,
 'Time constant (ms)': 4,
 '\u2003Input resistance (MΩ)': 2}
'''

def specific_properties(obs_frame,test_frame,name_map):
    neuron_values = {}
    p = {}
    reverse_cell_map = {}
    test_name_map = {}
    for cell_name,test_values in test_frame.items():
        #for test in test_values:
        p["nlex_id"] = name_map[cell_name]
        neuron_values[str(p["nlex_id"])] = {}
        for val in range(0,13):
            #if val == 8 or val == 9 or val == 11:
            #    break


            reverse_cell_map[cell_name] = p["nlex_id"]

            #print(test.ephysprop_name)

            print(p["nlex_id"] ,"cell_name", cell_name)
            url_to_get = str('https://neuroelectro.org/api/1/nedm/?nlex=')+str(p["nlex_id"])+str('&e=')+str(val)+str('&limit=100')
            #print('fails at number: val: ',val)
            print(url_to_get)
            #import pdb; pdb.set_trace()
            with urllib.request.urlopen(url_to_get) as url:
                data = json.loads(url.read().decode())
            for objects in data['objects']:
                last_label = objects['ecm']['ref_text']
                print(last_label)
                #print(test.ephysprop_name in last_label)
            if not len(data['objects']):
                print('gets to fail here')
                break
            if not len(objects['ecm']['ref_text']):
                print('gets to fail here')

                break

            if len(data['objects']):
                p[last_label] = {}
            #p
            datax = [ objects['val'] for objects in data['objects'] ]
            datax = [ dx for dx in datax if type(dx) is not type(None) ]
            neuron_values[str(p["nlex_id"])][val] = datax

            if len(datax):
                samples = [ n['n'] for n in data['objects'] ]
                samples = [ n for n in samples if type(n) is not type(None) ]
                sample_sizes = sum(samples)
                p[last_label]['values'] = datax
                neuron_values[str(p["nlex_id"])]['n'] = sample_sizes

                #neuron_values[str(p["nlex_id"])][val] = datax
                #p.neuron_values =
                test_name_map[last_label] = val
                #p[last_label]['n'] = sample_sizes
                #test.sample_size = sample_sizes
                #test.data_distribution = datax
                with open('specific_test_data.p','wb') as f:
                    pickle.dump([neuron_values,test_name_map,name_map,reverse_cell_map],f)

                plt.clf()#hartigan_diptest(self.data)
                #n = 1000;
                #b = n//10;
                #h = np.histogram(datax,bins=b)
                #h = np.vstack((0.5*(h[1][:-1]+h[1][1:]),h[0])).T  # because h[0] and h[1] have different sizes.
                try:
                    datax = np.array(datax)
                    indexes = peakutils.indexes(datax, thres=0.02/max(cb), min_dist=100)
                    interpolatedIndexes = peakutils.interpolate(range(0, len(datax)), datax, ind=indexes)
                    kmeans = KMeans(n_clusters=2).fit(datax.reshape(len(datax),1))
                    print(kmeans.cluster_centers_, 'clusert centre')
                except:
                    pass
                try:

                    #plt.scatter(datax)
                    sns.distplot(sorted(datax), color="skyblue", label=str(last_label)+str(p["nlex_id"])+str(val))
                    plt.xlabel(str('sample_size: ')+str(sample_sizes))
                    plt.legend(loc="upper left")
                except:
                    plt.xlabel(str('sample_size: ')+str(sample_sizes))
                    plt.hist(sorted(datax))      #use this to draw histogram of your data
                    #plt.scatter(datax)
                try:
                    plt.savefig(str(last_label)+str(p["nlex_id"])+str(val)+str('.png'))
                except:
                    pass
    return (obs_frame,test_frame,neuron_values)
''''
pipe_tests_path = str(os.getcwd())+'/pipe_tests.p'
assert os.path.isfile(pipe_tests_path) == True
with open(pipe_tests_path,'rb') as f:
    pipe_tests = pickle.load(f)
'''
all_tests_path = str(os.getcwd())+'/all_tests.p'
assert os.path.isfile(all_tests_path) == True
with open(all_tests_path,'rb') as f:
    (obs_frame,test_frame) = pickle.load(f)
#import pdb; pdb.set_trace()



(obs_frame,test_frame,neuron_values) = specific_properties(obs_frame,test_frame,name_map)

with open('specific_test_data.p','rb') as f:
   contents = pickle.load(f)

cell_name_map = contents[2]
neuron_values = contents[0]



#np.random.seed(45)
#i = np.random.randint(0,2,n)
#x = i*np.random.normal(-2.0,0.8,n) + (1-i)*np.random.normal(2.0,0.8,n)
#_ = plt.hist(x,bins=b)

'''
neuron_values = properties(pipe)
with open('values_for_r.p','wb') as f:
    pickle.dump(neuron_values,f)
'''
