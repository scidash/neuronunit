import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import urllib.request, json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


import pandas

try:
    ne_raw = pandas.read_csv('article_ephys_metadata_curated.csv', delimiter='\t')
    os.system('!ls -ltr *.csv')
except:
    os.system('wget https://neuroelectro.org/static/src/article_ephys_metadata_curated.csv')
    ne_raw = pandas.read_csv('article_ephys_metadata_curated.csv', delimiter='\t')

blah = ne_raw[ne_raw['NeuronName'].str.match('Hippocampus CA1 pyramidal cell')]
print([i for i in blah.columns])
here = ne_raw[ne_raw['TableID']==18]
print(here['rheo_raw'])

#import pdb; pdb.set_trace()
from scipy.signal import find_peaks_cwt


from neuronunit import tests as _, neuroelectro
from neuronunit.tests import passive, waveform, fi
from neuronunit.tests.fi import RheobaseTestP
from neuronunit.tests import passive, waveform#, druckmann2013
import sciunit


with open('specific_test_data.p','rb') as f:
   contents = pickle.load(f)
cell_name_map = contents[2]
neuron_values = contents[0]
import quantities as pq
test_map = {1:'CapacitanceTest',2:'InputResistanceTest',3:'RestingPotentialTest',4:'TimeConstantTest',\
            5:'InjectedCurrentAPAmplitudeTest',6:'InjectedCurrentAPWidthTest',\
            7:'InjectedCurrentAPThresholdTest',8:'RheobaseTest'}

units_map = {1:pq.pF,2:pq.MOhm,3:pq.mV,4:pq.ms,\
            5:pq.mV,6:pq.ms,\
            7:pq.mV,8:pq.pA}

complete_map = {}
for k,v in units_map.items():
    complete_map[test_map[k]] = units_map[k]

name_map ={}
name_map["Cerebellum Purkinje cell"] = "sao471801888"
name_map["Dentate gyrus basket cell"] = "nlx_cell_100201"
name_map["Hippocampus CA1 basket cell"] = "nlx_cell_091205"
name_map["Neocortex pyramidal cell layer 5-6"] = "nifext_50"
name_map["Olfactory bulb (main) mitral cell"] = "nlx_anat_100201"
name_map["Hippocampus CA1 pyramidal cell"] = "sao830368389"
import seaborn
#ax = sns.violinplot(x="day", y="total_bill", data=tips)
inv_name_map = {v: k for k, v in name_map.items()}
executable_tests = {}
russell_tests = {}

for nlex_ids,values in neuron_values.items():
    #plt.legend(loc="upper left")
    #plt.savefig(str(cell_name)+str(test_map[i])+str('.png'))
    cell_name = inv_name_map[nlex_ids]
    executable_tests[cell_name] = {}#[test_map[i]]
    russell_tests[cell_name] = {}
    for i in neuron_values[nlex_ids].keys():
        neuron_values[nlex_ids][i]['modes'] = []
        plt.clf()

        fig, ax = plt.subplots()
        if i==8:
            print(i,test_map[i])
        # the histogram of the data
        n, bins, patches = ax.hist(sorted(neuron_values[nlex_ids][i]['values']), label=str(cell_name)+str(test_map[i]))
        #ax = sns.violinplot(x="day", y="total_bill", data=tips)
        plt.clf()
        ax = sns.violinplot(data=sorted(neuron_values[nlex_ids][i]['values']), palette="Set2", split=True,
                     scale="count", inner="stick")
        plt.legend(loc="upper left")
        plt.savefig(str(cell_name)+str(test_map[i])+str('_hist_.png'))

        plt.clf()
        plt.hist(sorted(neuron_values[nlex_ids][i]['values']), label=str(cell_name)+str(test_map[i]))
        plt.savefig(str(cell_name)+str(test_map[i])+str('_violin_.png'))

        mode0 = bins[np.where(n==np.max(n))[0][0]]
        neuron_values[nlex_ids][i]['modes'].append(mode0)
        half = (bins[1]-bins[0])/2.0


        try:
            #print(sorted(n))
            #import pdb; pdb.set_trace()

            mode1 = bins[np.where(n==sorted(n)[-2])[0][0]]
            neuron_values[nlex_ids][i]['modes'].append(mode1)
            #plt.scatter(mode1+half,sorted(n)[-2],c='r')

        except:
            pass
        #max_peakind = find_peaks_cwt(bins,np.arange(1,10))
        russell_tests[cell_name][test_map[i]] = (neuron_values[nlex_ids][i]['modes'],neuron_values[nlex_ids][i]['std'],neuron_values[nlex_ids][i]['values'])

        test_classes = [fi.RheobaseTest,
                         passive.InputResistanceTest,
                         passive.TimeConstantTest,
                         passive.CapacitanceTest,
                         passive.RestingPotentialTest,
                         waveform.InjectedCurrentAPWidthTest,
                         waveform.InjectedCurrentAPAmplitudeTest,
                         waveform.InjectedCurrentAPThresholdTest]#,
        import neuronunit
        for tt in test_classes:
            if test_map[i] in str(tt):
                anchor = neuronunit.__file__
                anchor = os.path.dirname(anchor)
                pipe_tests_path = os.path.join(os.sep,anchor,'unit_test/pipe_tests.p')
                #import pdb; pdb.set_trace()
                
                #pipe_tests_path = str(os.getcwd())+'/pipe_tests.p'
                assert os.path.isfile(pipe_tests_path) == True
                with open(pipe_tests_path,'rb') as f:
                    pipe_tests = pickle.load(f)
                
                observation = {}
                observation['mean'] = neuron_values[nlex_ids][i]['modes'][0]*units_map[i]
                observation['value'] = neuron_values[nlex_ids][i]['modes'][0]*units_map[i]
                observation['std'] = neuron_values[nlex_ids][i]['std']*units_map[i]
                observation['n'] = neuron_values[nlex_ids][i]['n']
                print(test_map[i])
            
                t = tt()#observation)#neuron_values[nlex_ids][i]['modes'][0]*units_map[i])
                t.observation = observation
                executable_tests[cell_name][test_map[i]] =  t
            
                assert test_map[i] == t.name
                executable_tests[cell_name][test_map[i]].data = None
                executable_tests[cell_name][test_map[i]].data = neuron_values[nlex_ids][i]['values']
pipe_tests_path = str(os.getcwd())+'/russell_tests.p'
#assert os.path.isfile(pipe_tests_path) == True
import pdb; pdb.set_trace()
executable_tests.pop('Dentate gyrus basket cell', None)
with open(pipe_tests_path,'wb') as f:
    pickle.dump([executable_tests,complete_map],f)
exit()

pipe_tests_path = str(os.getcwd())+'/pipe_tests.p'
assert os.path.isfile(pipe_tests_path) == True
with open(pipe_tests_path,'rb') as f:
    pipe_tests = pickle.load(f)
all_tests_path = str(os.getcwd())+'/all_tests.p'
assert os.path.isfile(all_tests_path) == True
with open(all_tests_path,'rb') as f:
    (obs_frame,test_frame) = pickle.load(f)
'''
import pdb; pdb.set_trace()
observations = {}
executable_tests = {}
for cell_name, local_tests in russell_tests.items():
    executable_tests[cell_name] = {}
    for test_name,tt in local_tests.items():
        for index, classic_tests in enumerate(test_classes):
            if tt in classic_tests:
                executable_tests[cell_name][test_name] = classic_tests(tt['modes'][0])
print(executable_tests)
import pdb; pdb.set_trace()
'''
#hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
#update amplitude at the location in sciunit thats its passed to, without any loss of generality.
suite = sciunit.TestSuite(tests,name="vm_suite")

import pdb; pdb.set_trace()
'''
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

data = [neuron_values['sao830368389']['Rin (MΩ)'],neuron_values['sao830368389']['AP current threshold (pA)'] ]

cnt= 0
for datax in data:
    cnt+=1
    plt.clf()
    try:

        #plt.scatter(datax)
        sns.distplot(sorted(datax), color="skyblue", label=str(last_label)+str(p["nlex_id"])+str(val))
        #plt.xlabel(str('sample_size: ')+str(sample_sizes))
        plt.legend(loc="upper left")
    except:
        #plt.xlabel(str('sample_size: ')+str(sample_sizes))
        plt.hist(sorted(datax))      #use this to draw histogram of your data
#plt.scatter(neuron_values['nifext_50']['\u2003Input resistance (MΩ)'],neuron_values['nifext_50']['Rheobase, pA'])
    plt.savefig(str(cnt)+'inR_vs_rheobase2.png')

plt.clf()
data = [neuron_values['nifext_50']['\u2003Input resistance (MΩ)'],neuron_values['nifext_50']['Rheobase, pA']]
cnt= 0
for datax in data:
    cnt+=1
    plt.clf()
    try:

        #plt.scatter(datax)
        sns.distplot(sorted(datax), color="skyblue", label=str(last_label)+str(p["nlex_id"])+str(val))
        #plt.xlabel(str('sample_size: ')+str(sample_sizes))
        plt.legend(loc="upper left")
    except:
        #plt.xlabel(str('sample_size: ')+str(sample_sizes))
        plt.hist(sorted(datax))      #use this to draw histogram of your data
#plt.scatter(neuron_values['nifext_50']['\u2003Input resistance (MΩ)'],neuron_values['nifext_50']['Rheobase, pA'])
    plt.savefig(str(cnt)+'inR_vs_rheobase.png')
'''
B
