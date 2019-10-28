import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import urllib.request, json

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

import os
import neuronunit
anchor = neuronunit.__file__
anchor = os.path.dirname(anchor)
mypath = os.path.join(os.sep,anchor,'tests/russell_tests.p')

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
#nxids =
name_map ={}
name_map["Cerebellum Purkinje cell"] = "sao471801888"
name_map["Dentate gyrus basket cell"] = "nlx_cell_100201"
name_map["Hippocampus CA1 basket cell"] = "nlx_cell_091205"

#name_map["Hippocampus CA1 basket cell"] = "nlx_cell_091205"
name_map["Neocortex pyramidal cell layer 5-6"] = "nifext_50"
name_map["Olfactory bulb (main) mitral cell"] = "nlx_anat_100201"
name_map["Hippocampus CA1 pyramidal cell"] = "sao830368389"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from neuronunit import neuroelectro
from scipy.stats import mode
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

test_map = {1:'CapacitanceTest',2:'InputResistanceTest',3:'RestingPotentialTest',\
4:'TimeConstantTest',5:'InjectedCurrentAPAmplitudeTest',6:'InjectedCurrentAPWidthTest',\
7:'InjectedCurrentAPThresholdTest',8:'RheobaseTest'}

reverse_test_map = {
        ' RMP, mV ': 3,
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
        '\u2003Input resistance (MΩ)': 2
     }


test_map = {1:'.CapacitanceTest',2:'InputResistanceTest',3:'RestingPotentialTest',4:'TimeConstantTest',5:'InjectedCurrentAPAmplitudeTest',6:'InjectedCurrentAPWidthTest',7:'InjectedCurrentAPThresholdTest',8:'RheobaseTestp'}

def specific_properties(name_map,test_map):
    test_obs = {}
    neuron_values = {}
    p = {}
    reverse_cell_map = {}
    test_name_map = {}
    for cell_name,nx_id in name_map.items():
        #for test in test_values:
        p["nlex_id"] =nx_id # name_map[cell_name]
        neuron_values[str(p["nlex_id"])] = {}
        for val in range(1,13):
            if val==8:
                pass
            if val == 9 or val == 11:
                break
            reverse_cell_map[cell_name] = p["nlex_id"]
            url_to_get = str('https://neuroelectro.org/api/1/nedm/?nlex=')+str(p["nlex_id"])+str('&e=')+str(val)+str('&limit=100')
            with urllib.request.urlopen(url_to_get) as url:
                data = json.loads(url.read().decode())
            for objects in data['objects']:
                last_label = objects['ecm']['ref_text']
            if not len(data['objects']):
                break
            if not len(objects['ecm']['ref_text']):
                break

            if len(data['objects']):
                p[last_label] = {}
            datax = [ objects['val'] for objects in data['objects'] ]
            datax = [ dx for dx in datax if type(dx) is not type(None) ]
            neuron_values[str(p["nlex_id"])][val] = {}
            neuron_values[str(p["nlex_id"])][val]['values'] = datax
            #if val==8:
            #    print('rheobase encoded?')
            #neuron_values[str(p["nlex_id"])][val]['mode'] = mode(datax)
            neuron_values[str(p["nlex_id"])][val]['mean'] = np.mean(datax)
            neuron_values[str(p["nlex_id"])][val]['std'] = np.std(datax)
            test_obs[test_map[val]] = [np.mean(datax),np.std(datax),datax]

            if len(datax):
                samples = [ n['n'] for n in data['objects'] ]
                samples = [ n for n in samples if type(n) is not type(None) ]
                sample_sizes = sum(samples)
                neuron_values[str(p["nlex_id"])][val]['n'] = sample_sizes
                p[last_label]['values'] = datax

                test_name_map[last_label] = val
                with open('specific_test_data.p','wb') as f:
                    pickle.dump([neuron_values,test_name_map,name_map,reverse_cell_map],f)

    return neuron_values, test_obs


'''
all_tests_path = str(mypath)+'/all_tests.p'
assert os.path.isfile(all_tests_path) == True
with open(all_tests_path,'rb') as f:
    (obs_frame,test_frame) = pickle.load(f)
'''


neuron_values,test_obs = specific_properties(name_map,test_map)



#np.random.seed(45)
#i = np.random.randint(0,2,n)
#x = i*np.random.normal(-2.0,0.8,n) + (1-i)*np.random.normal(2.0,0.8,n)
#_ = plt.hist(x,bins=b)

'''
neuron_values = properties(pipe)
with open('values_for_r.p','wb') as f:
    pickle.dump(neuron_values,f)
'''
