import matplotlib
#matplotlib.use('Agg')
import seaborn as sns
import urllib.request, json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
with open('specific_test_data.p','rb') as f:
   contents = pickle.load(f)
cell_name_map = contents[2]
neuron_values = contents[0]
test_map = {1:'CapacitanceTest',2:'InputResistanceTest',3:'RestingPotentialTest',4:'TimeConstantTest',\
            5:'InjectedCurrentAPAmplitudeTest',6:'InjectedCurrentAPWidthTest',\
            7:'InjectedCurrentAPThresholdTest',8:'RheobaseTestp'}
name_map ={}
name_map["Cerebellum Purkinje cell"] = "sao471801888"
name_map["Dentate gyrus basket cell"] = "nlx_cell_100201"
name_map["Hippocampus CA1 basket cell"] = "nlx_cell_091205"
name_map["Neocortex pyramidal cell layer 5-6"] = "nifext_50"
name_map["Olfactory bulb (main) mitral cell"] = "nlx_anat_100201"
name_map["Hippocampus CA1 pyramidal cell"] = "sao830368389"

inv_name_map = {v: k for k, v in name_map.items()}

russell_tests = {}
for nlex_ids,values in neuron_values.items():
    cell_name = inv_name_map[nlex_ids]
    russell_tests[cell_name] = {}
    for i in neuron_values[nlex_ids].keys():
        russell_tests[cell_name][test_map[i]] = (neuron_values[nlex_ids][i]['mean'],neuron_values[nlex_ids][i]['std'],neuron_values[nlex_ids][i]['values'])
        plt.clf()
        try:
            sns.distplot(sorted(neuron_values[nlex_ids][i]['values']), color="skyblue", label=str(cell_name)+str(test_map[i]))
            plt.legend(loc="upper left")
            plt.savefig(str(cell_name)+str(test_map[i])+str('.png'))
        except:
            plt.scatter(neuron_values[nlex_ids][i]['mean'],4,c='r')
            plt.scatter(neuron_values[nlex_ids][i]['median'],4,c='g')

            plt.hist(sorted(neuron_values[nlex_ids][i]['values']), label=str(cell_name)+str(test_map[i]))
            plt.legend(loc="upper left")
            plt.savefig(str(cell_name)+str(test_map[i])+str('.png'))


pipe_tests_path = str(os.getcwd())+'/pipe_tests.p'
assert os.path.isfile(pipe_tests_path) == True
with open(pipe_tests_path,'rb') as f:
    pipe_tests = pickle.load(f)
all_tests_path = str(os.getcwd())+'/all_tests.p'
assert os.path.isfile(all_tests_path) == True
with open(all_tests_path,'rb') as f:
    (obs_frame,test_frame) = pickle.load(f)
#import pdb; pdb.set_trace()
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
