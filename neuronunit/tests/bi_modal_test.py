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
    for val in range(1,20):
        for p in pipe:
            url_to_get = str('https://neuroelectro.org/api/1/nedm/?nlex=')+str(p["nlex_id"])+str('&e=')+str(val)+str('&limit=100')
            with urllib.request.urlopen(url_to_get) as url:
                data = json.loads(url.read().decode())
            for objects in data['objects']:
                last_label = objects['ecm']['ref_text']
            datax = [ objects['val'] for objects in data['objects'] ]
            datax = [ dx for dx in datax if type(dx) is not type(None) ]
            neuron_values[str(p["nlex_id"])] = {}
            neuron_values[str(p["nlex_id"])][val] = datax
            if len(datax):
                samples = [ n['n'] for n in data['objects'] ]
                samples = [ n for n in samples if type(n) is not type(None) ]
                sample_sizes = sum(samples)

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


                    sns.distplot(sorted(datax), color="skyblue", label=str(last_label)+str(p["nlex_id"])+str(val))
                    plt.xlabel(str('sample_size: ')+str(sample_sizes))
                    plt.legend(loc="upper left")
                except:
                    plt.xlabel(str('sample_size: ')+str(sample_sizes))
                    plt.hist(sorted(datax))      #use this to draw histogram of your data

                plt.savefig(str(last_label)+str(p["nlex_id"])+str(val)+str('.png'))


    return neuron_values



#np.random.seed(45)
#i = np.random.randint(0,2,n)
#x = i*np.random.normal(-2.0,0.8,n) + (1-i)*np.random.normal(2.0,0.8,n)
#_ = plt.hist(x,bins=b)


neuron_values = properties(pipe)
with open('values_for_r.p','wb') as f:
    pickle.dump(neuron_values,f)
