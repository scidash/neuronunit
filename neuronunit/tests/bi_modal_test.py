import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import urllib.request, json

import matplotlib.pyplot as plt
import numpy as np
import pickle
#from modality import calibrated_diptest, calibrated_bwtest, silverman_bwtest, hartigan_diptest, excess_mass_modes
#from modality.util import auto_interval
#import anchor
purkinje ={"id": 18, "name": "Cerebellum Purkinje cell", "neuron_db_id": 271, "nlex_id": "sao471801888"}
fi_basket = {"id": 65, "name": "Dentate gyrus basket cell", "neuron_db_id": None, "nlex_id": "nlx_cell_100201"}
pvis_cortex = {"id": 111, "name": "Neocortex pyramidal cell layer 5-6", "neuron_db_id": 265, "nlex_id": "nifext_50"}
#does not have rheobase
olf_mitral = {"id": 129, "name": "Olfactory bulb (main) mitral cell", "neuron_db_id": 267, "nlex_id": "nlx_anat_100201"}
ca1_pyr = {"id": 85, "name": "Hippocampus CA1 pyramidal cell", "neuron_db_id": 258, "nlex_id": "sao830368389"}
pipe = [ fi_basket, ca1_pyr, purkinje,  pvis_cortex,olf_mitral]

def properties(pipe):
    neuron_values = {}
    for val in range(1,10):
        for p in pipe:
            url_to_get = str('https://neuroelectro.org/api/1/nedm/?nlex=')+str(p["nlex_id"])+str('&e=')+str(val)+str('&limit=100')
            with urllib.request.urlopen(url_to_get) as url:
                data = json.loads(url.read().decode())
            for objects in data['objects']:
                last_label = objects['ecm']['ref_text']
            #values = [ objects['val'] for objects in data['objects'] ]
            datax = [ objects['val'] for objects in data['objects'] ]
            datax = [ dx for dx in datax if type(dx) is not type(None) ]
            neuron_values[str(p["nlex_id"])] = {}
            neuron_values[str(p["nlex_id"])][val] = datax
            if len(datax):
                samples = [ n['n'] for n in data['objects'] ]
                samples = [ n for n in samples if type(n) is not type(None) ]
                sample_sizes = sum(samples)
                #reweighted = []
                #for i,j in zip(datax,samples):
                #    reweighted.append(i*j)

                plt.clf()#hartigan_diptest(self.data)
                try:
                    sns.distplot(sorted(datax), color="skyblue", label=str(last_label)+str(p["nlex_id"])+str(val))
                    plt.xlabel(str('sample_size: ')+str(sample_sizes))
                    plt.legend(loc="upper left")
                except:
                    plt.xlabel(str('sample_size: ')+str(sample_sizes))
                    plt.hist(sorted(datax))      #use this to draw histogram of your data

                plt.savefig(str(last_label)+str(p["nlex_id"])+str(val)+str('.png'))


    return neuron_values
neuron_values = properties(pipe)
with open('values_for_r.p','wb') as f:
    pickle.dump(neuron_values,f)
