#!/usr/bin/env python
# coding: utf-8

# Here we remove all but one parameter from the dictionary of free parameters.
# 
# Parameters "popped" from this dictionary are frozen only 'a' is left and free to vary.
# 

# In[1]:


from neuronunit.optimisation.model_parameters import MODEL_PARAMS


# In[2]:


import pickle
import numpy as np
tests = pickle.load(open('allen_NU_tests.p','rb'))
names = [t.name for t in tests[3].tests ]
names;


# In[3]:


#yes_list = ['AHP_depth_abs_3.0x','sag_ratio2_3.0x','ohmic_input_resistance_3.0x','sag_ratio2_3.0x','peak_voltage_3.0x','voltage_base_3.0x','Spikecount_3.0x','ohmic_input_resistance_vb_ssse_3.0x']
simple_yes_list = ['all_ISI_values','ISI_log_slope','mean_frequency','adaptation_index2','first_isi','ISI_CV','median_isi','AHP_depth_abs','sag_ratio2','ohmic_input_resistance','sag_ratio2','peak_voltage','voltage_base','Spikecount','ohmic_input_resistance_vb_ssse']
#new_list = [t for t in tests[3].tests if t.name in yes_list]


# # Get and plot this experiment

# In[4]:


tests[3].tests;


# In[5]:


tests[3].name


# In[6]:


names = [t.observation for t in tests[3].tests if "Spike" in t.name]
names


# In[7]:


specimen_id = tests[4].name

#with open(str(specimen_id)+'later_allen_NU_tests.p','rb') as f:
#    suite = pickle.load(f) 


# In[ ]:




from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.extract_cell_features import extract_cell_features
from collections import defaultdict
from allensdk.core.nwb_data_set import NwbDataSet
from neuronunit.optimisation.optimization_management import efel_evaluation,rekeyed
import numpy as np 
from neuronunit.make_allen_tests import AllenTest
from sciunit import TestSuite
import matplotlib.pyplot as plt
from neuronunit.models import StaticModel 

# initialize the cacher
specimen_id = tests[3].name
def allen_id_to_sweeps(specimen_id):
    ctc = CellTypesCache(manifest_file='cell_types/manifest.json')

    specimen_id = int(specimen_id)
    data_set = ctc.get_ephys_data(specimen_id)
    sweeps = ctc.get_ephys_sweeps(specimen_id)
    sweep_numbers = defaultdict(list)
    for sweep in sweeps:
        sweep_numbers[sweep['stimulus_name']].append(sweep['sweep_number'])
    return sweep_numbers,data_set,sweeps

sweep_numbers,data_set,sweeps = allen_id_to_sweeps(specimen_id)
        

def closest(lst, K):       
     lst = np.asarray(lst) 
     idx = (np.abs(lst - K)).argmin() 
     return idx
      
currents={}
allen_test_suites = []

def get_rheobase(numbers,sets):
    rheobase_numbers = [sweep_number for sweep_number in numbers if len(sets.get_spike_times(sweep_number))==1]
    sweeps = [sets.get_sweep(n) for n in rheobase_numbers ]
    temp = [ (i,np.max(s['stimulus'])) for i,s in zip(rheobase_numbers,sweeps) if 'stimulus' in s.keys()]# if np.min(s['stimulus'])>0 ]
    temp = sorted(temp,key=lambda x:[1],reverse=True)
    rheobase = temp[0][1]
    index = temp[0][0]
    return rheobase,index

from neo.core import AnalogSignal
import quantities as qt
def get_models(data_set,sweep_numbers,specimen_id,simple_yes_list):
    sweep_numbers = sweep_numbers['Square - 2s Suprathreshold']
    rheobase = -1
    above_threshold_sn = []
    for sn in sweep_numbers:
        sweep_data = data_set.get_sweep(sn)

        spike_times = data_set.get_spike_times(sn)

        # stimulus is a numpy array in amps
        stimulus = sweep_data['stimulus']

        if len(spike_times) == 1:
            if np.max(stimulus)> rheobase and rheobase==-1:
                rheobase = np.max(stimulus)
                stim = rheobase
                currents['rh']=stim
                sampling_rate = sweep_data['sampling_rate']
                vmrh = AnalogSignal(sweep_data['response'],sampling_rate=sampling_rate*qt.Hz,units=qt.V)

        if len(spike_times) >= 1:
            reponse = sweep_data['response']
            sampling_rate = sweep_data['sampling_rate']
            vmm = AnalogSignal(sweep_data['response'],sampling_rate=sampling_rate*qt.Hz,units=qt.V)
            above_threshold_sn.append((np.max(stimulus),sn,vmm))
            print(len(spike_times))

    myNumber = 3.0*rheobase
    currents_ = [t[0] for t in above_threshold_sn]
    indexvm30 = closest(currents_, myNumber)
    stim = above_threshold_sn[indexvm30][0]
    currents['30']=stim
    vm30 = above_threshold_sn[indexvm30][2]
    myNumber = 1.5*rheobase
    currents_ = [t[0] for t in above_threshold_sn]
    indexvm15 = closest(currents_, myNumber)
    stim = above_threshold_sn[indexvm15][0]
    currents['15']=stim
    vm15 = above_threshold_sn[indexvm15][2]
    sm = StaticModel(vm = vmrh)
    sm.rheobase = rheobase
    sm.vm15 = vm15
    sm.vm30 = vm30
    sm = efel_evaluation(sm,thirty=False)

    sm = efel_evaluation(sm,thirty=True)
    sm = rekeyed(sm)
    useable = False
    sm.vmrh = vmrh
    plt.show()
    allen_tests = []
    if sm.efel_15 is not None:
        for k,v in sm.efel_15[0].items():
            try:
                if "SpikeCount" in k:
                    print(v)
                    import pdb
                    pdb.set_trace()

                at = AllenTest(name=str(k)+'_1.5x')
                at.set_observation(v)
                allen_tests.append(at)
                if k in simple_yes_list:
                    useable = True
            except:
                pass
    if sm.efel_30 is not None:
        for k,v in sm.efel_30[0].items():
            try:
                if "SpikeCount" in k:
                    print(v)
                    import pdb
                    pdb.set_trace()

                if k in simple_yes_list:
                    useable = True
                at = AllenTest(name=str(k)+'_3.0x')
                at.set_observation(v)
                allen_tests.append(at)
            except:
                pass

    suite = TestSuite(allen_tests,name=str(specimen_id))
    suite.traces = None
    suite.traces = {}
    suite.traces['rh_current'] = sm.rheobase
    suite.traces['vmrh'] = sm.vmrh
    suite.traces['vm15'] = sm.vm15
    suite.traces['vm30'] = sm.vm30
    suite.model = None
    suite.useable = useable
    suite.model = sm
    suite.stim = None

    suite.stim = currents
      
    return suite,specimen_id
try: 
    #assert 1==2
    with open(str(specimen_id)+'later_allen_NU_tests.p','rb') as f:
        suite = pickle.load(f) 
except:
    suite,specimen_id = get_models(data_set,sweep_numbers,specimen_id,simple_yes_list)
    with open(str(specimen_id)+'later_allen_NU_tests.p','wb') as f:
        pickle.dump(suite,f) 
        
        


# In[ ]:


from neuronunit.optimisation.optimization_management import dtc_to_rheo, inject_and_plot_model30,check_bin_vm30,check_bin_vm15
simple_yes_list = ['all_ISI_values_3.0x','ISI_log_slope_3.0x','mean_frequency_3.0x','adaptation_index2_3.0x','first_isi_3.0x','ISI_CV_3.0x','median_isi_3.0x','AHP_depth_abs_3.0x','sag_ratio2_3.0x','ohmic_input_resistance_3.0x','sag_ratio2_3.0x','peak_voltage_3.0x','voltage_base_3.0x','Spikecount_3.0x','ohmic_input_resistance_vb_ssse_3.0x']

class obj(object):
    def __init__(self):
        self.vm30 = None
        self.vm15 = None
        
target = obj() #DataTC(backend="ADEXP")
target.vm30 = suite.traces['vm30'] 
target.vm15 = suite.traces['vm15'] 

nu_tests = suite.tests;
check_bin_vm30(target,target)


# In[ ]:



for t in nu_tests:
    if hasattr(t.observation['mean'],'units'):
        t.observation['mean'] = np.mean(t.observation['mean'])*t.observation['mean'].units
        t.observation['std'] = np.mean(t.observation['mean'])*t.observation['mean'].units
    else:
        t.observation['mean'] = np.mean(t.observation['mean'])*t.observation['mean']
        t.observation['std'] = np.mean(t.observation['mean'])*t.observation['mean']


# In[ ]:


from neuronunit.tests.target_spike_current import SpikeCountSearch
from neuronunit.optimisation.data_transport_container import DataTC
from neuronunit.optimisation.optimization_management import dtc_to_rheo, rekeyed
attrs = {k:np.mean(v) for k,v in MODEL_PARAMS["ADEXP"].items()}
dtc = DataTC(backend="ADEXP",attrs=attrs)
for t in nu_tests:
    if t.name == 'Spikecount_3.0x':
        spk_count = float(t.observation['mean'])
        print(spk_count,'spike_count')
        break
observation_range={}
observation_range['value'] = spk_count
scs = SpikeCountSearch(observation_range)
target_current = scs.generate_prediction(dtc.dtc_to_model())


# In[ ]:


target_current['value'] =target_current['value'] /3.0


# In[ ]:


import efel
import pandas as pd
import seaborn as sns
list(efel.getFeatureNames());
from utils import dask_map_function

import bluepyopt as bpop
import bluepyopt.ephys as ephys
import pickle
from sciunit.scores import ZScore
from sciunit import TestSuite
from sciunit.scores.collections import ScoreArray
import sciunit
import numpy as np
from neuronunit.optimisation.optimization_management import dtc_to_rheo, switch_logic,active_values
from neuronunit.tests.base import AMPL, DELAY, DURATION

import quantities as pq
PASSIVE_DURATION = 500.0*pq.ms
PASSIVE_DELAY = 200.0*pq.ms
import matplotlib.pyplot as plt
from bluepyopt.ephys.models import ReducedCellModel
import numpy
from neuronunit.optimisation.optimization_management import test_all_objective_test
from neuronunit.optimisation.optimization_management import check_binary_match, three_step_protocol,inject_and_plot_passive_model
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
import copy

import numpy as np
from make_allen_tests import AllenTest

from sciunit.scores import ZScore
from collections.abc import Iterable

simple_cell = ephys.models.ReducedCellModel(
        name='simple_cell',
        params=MODEL_PARAMS["ADEXP"],backend="ADEXP")  
simple_cell.backend = "ADEXP"
simple_cell.allen = None
simple_cell.allen = True


model = simple_cell
model.params = {k:np.mean(v) for k,v in model.params.items() }

features = None
allen = True


# In[ ]:


from sciunit.scores import ZScore
yes_list = simple_yes_list 
class NUFeatureAllenMultiSpike(object):
    def __init__(self,test,model,cnt,target,check_list,spike_obs,print_stuff=False):
        self.test = test
        self.model = model
        self.check_list = check_list
        self.spike_obs = spike_obs
        self.cnt = cnt
        self.target = target
        self.print_stuff = print_stuff
    def calculate_score(self,responses):
        
        if not 'features' in responses.keys():# or not 'model' in responses.keys():
            return 1000.0
        features = responses['features']

        check_list = self.check_list
    

        if False:
            self.test.set_prediction(responses['rheobase'])   
            self.test.prediction['value'] = self.test.prediction['mean']
            self.test.observation['std'] = np.abs(np.mean(self.target.rheobase['mean']))
            self.test.observation['mean'] = np.mean(self.test.observation['mean'])   
            self.test.observation['value'] = np.mean(self.test.observation['value'])   

            self.test.score_type = ZScore

            score_gene = self.test.compute_score(self.test.prediction,self.test.observation)
            if score_gene is not None:
                if score_gene.log_norm_score is not None:
                    delta = np.abs(float(score_gene.log_norm_score))
                else:
                    if score_gene.raw is not None:
                        delta = np.abs(float(score_gene.raw))
                    else:
                        delta = None
            else:
                delta = None
            if delta is None:
                delta = np.abs(np.float(responses['rheobase'])-np.mean(self.test.observation['mean']))
            if np.nan==delta or delta==np.inf:
                delta = np.abs(np.float(responses['rheobase'])-np.mean(self.test.observation['mean']))

            return delta

        feature_name = self.test.name
        #rint(feature_name)
        delta0 = np.abs(features['Spikecount_3.0x']-np.mean(self.spike_obs[0]['mean']))
        delta1 = np.abs(features['Spikecount_1.5x']-np.mean(self.spike_obs[1]['mean']))
        if feature_name not in features.keys():
            return 1000.0+(delta0+delta1)
        
        if features[feature_name] is None:
            return 1000.0+(delta0+delta1)
            
        if type(features[self.test.name]) is type(Iterable):
            features[self.test.name] = np.mean(features[self.test.name])
        self.test.observation['std'] = np.abs(np.mean(self.test.observation['mean']))
        self.test.observation['mean'] = np.mean(self.test.observation['mean'])   
        self.test.set_prediction(np.mean(features[self.test.name]))

        if 'Spikecount_3.0x'==feature_name or 'Spikecount_1.5x'==feature_name:
            delta = np.abs(features[self.test.name]-np.mean(self.test.observation['mean']))
            if np.nan==delta or delta==np.inf:
                delta = 1000.0

            
            return delta
        else:


            if feature_name in check_list:
                if features[feature_name] is None:
                    print('gets here')
                    return 1000.0+(delta0+delta1)
                self.test.score_type = ZScore
                score_gene = self.test.feature_judge()
                if score_gene is not None:
                    if score_gene.log_norm_score is not None:
                        delta = np.abs(float(score_gene.log_norm_score))
                    else:
                        if score_gene.raw is not None:
                            delta = np.abs(float(score_gene.raw))
                        else:
                            delta = None

                else:
                    delta = None
                        #if delta==np.inf or np.isnan(delta):
                        #    if score_gene.raw is not None:
                        #        delta =  np.abs(float(score_gene.raw))
                if delta is None:
                    delta = np.abs(features[self.test.name]-np.mean(self.test.observation['mean']))


                if np.nan==delta or delta==np.inf:
                    delta = np.abs(features[self.test.name]-np.mean(self.test.observation['mean']))
                if np.nan==delta or delta==np.inf:
                    delta = 1000.0
                return delta#+delta2
            else:
                return 0.0
   
objectives = []
spike_obs = []
for tt in nu_tests:
    if 'Spikecount_3.0x' == tt.name:
        spike_obs.append(tt.observation)
    if 'Spikecount_1.5x' == tt.name:
        spike_obs.append(tt.observation)
spike_obs = sorted(spike_obs, key=lambda k: k['mean'],reverse=True)

#check_list["RheobaseTest"] = target.rheobase['value']
for cnt,tt in enumerate(nu_tests):
    feature_name = '%s' % (tt.name)
    if feature_name in simple_yes_list:
        if 'Spikecount_3.0x' == tt.name or 'Spikecount_1.5x' == tt.name:
            ft = NUFeatureAllenMultiSpike(tt,model,cnt,yes_list,yes_list,spike_obs)
            objective = ephys.objectives.SingletonObjective(
                feature_name,
                ft)
            objectives.append(objective)

       
score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives) 
      
lop={}
from bluepyopt.parameters import Parameter
for k,v in MODEL_PARAMS["ADEXP"].items():
    p = Parameter(name=k,bounds=v,frozen=False)
    lop[k] = p

simple_cell.params = lop


# In[ ]:


objectives


# In[ ]:


from utils import dask_map_function

sweep_protocols = []
for protocol_name, amplitude in [('step1', 0.05)]:

    protocol = ephys.protocols.SweepProtocol(protocol_name, [None], [None])
    sweep_protocols.append(protocol)
twostep_protocol = ephys.protocols.SequenceProtocol('twostep', protocols=sweep_protocols)

simple_cell.params_by_names(MODEL_PARAMS["ADEXP"].keys())
simple_cell.params;

MODEL_PARAMS["ADEXP"]
cell_evaluator = ephys.evaluators.CellEvaluator(
        cell_model=simple_cell,
        param_names=MODEL_PARAMS["ADEXP"].keys(),
        fitness_protocols={twostep_protocol.name: twostep_protocol},
        fitness_calculator=score_calc,
        sim='euler')

simple_cell.params_by_names(MODEL_PARAMS["ADEXP"].keys())
simple_cell.params;
simple_cell.seeded_current = target_current['value']


# In[ ]:



no_list = pickle.load(open("too_rippled_b.p","rb"))


objectives2 = []
for cnt,tt in enumerate(nu_tests):
    feature_name = '%s' % (tt.name)
    if (feature_name not in no_list) and (feature_name in simple_yes_list):
        if feature_name != "time_constant_1.5x" and feature_name != "RheobaseTest":
            ft = NUFeatureAllenMultiSpike(tt,model,cnt,yes_list,yes_list,spike_obs,print_stuff=True)
            objective = ephys.objectives.SingletonObjective(
                feature_name,
                ft)
            objectives2.append(objective)
objectives2
score_calc2 = ephys.objectivescalculators.ObjectivesCalculator(objectives2) 
objectives2
print(objectives2)


# In[ ]:



MODEL_PARAMS["ADEXP"]
cell_evaluator2 = ephys.evaluators.CellEvaluator(
        cell_model=simple_cell,
        param_names=list(MODEL_PARAMS["ADEXP"].keys()),
        fitness_protocols={twostep_protocol.name: twostep_protocol},
        fitness_calculator=score_calc2,
        sim='euler')
simple_cell.params_by_names(MODEL_PARAMS["ADEXP"].keys())

simple_cell.params;


MODEL_PARAMS["ADEXP"]
cell_evaluator2 = ephys.evaluators.CellEvaluator(
        cell_model=simple_cell,
        param_names=list(MODEL_PARAMS["ADEXP"].keys()),
        fitness_protocols={twostep_protocol.name: twostep_protocol},
        fitness_calculator=score_calc2,
        sim='euler')

MU =12

optimisation = bpop.optimisations.DEAPOptimisation(
        evaluator=cell_evaluator2,
        offspring_size = MU,
        map_function = map,
        selector_name='IBEA',mutpb=0.1,cxpb=0.35)
final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=25)
cp = {}
cp['final_pop'] = final_pop
cp['hall_of_fame'] = hall_of_fame





# In[ ]:



best_ind = hall_of_fame[0]
best_ind_dict = cell_evaluator2.param_dict(best_ind)
model = cell_evaluator2.cell_model
cell_evaluator2.param_dict(best_ind)

model.attrs = {str(k):float(v) for k,v in cell_evaluator2.param_dict(best_ind).items()}
opt = model.model_to_dtc()
opt.attrs = {str(k):float(v) for k,v in cell_evaluator2.param_dict(best_ind).items()}
target = copy.copy(opt)
target.vm30 = suite.traces['vm30'] 
target.vm15 = suite.traces['vm15'] 


#vm301,vm151,_,_,target = inject_and_plot_model30(target)
vm302,vm152,_,_,opt = inject_and_plot_model30(opt)

check_bin_vm30(opt,opt)
check_bin_vm15(opt,opt)


# In[ ]:


check_bin_vm30(target,target)


# In[ ]:


check_bin_vm15(target,target)


# In[ ]:




gen_numbers = logs.select('gen')
min_fitness = logs.select('min')
max_fitness = logs.select('max')
avg_fitness = logs.select('avg')
plt.plot(gen_numbers, max_fitness, label='max fitness')
plt.plot(gen_numbers, avg_fitness, label='avg fitness')
plt.plot(gen_numbers, min_fitness, label='min fitness')

plt.plot(gen_numbers, min_fitness, label='min fitness')
plt.semilogy()
plt.xlabel('generation #')
plt.ylabel('score (# std)')
plt.legend()
plt.xlim(min(gen_numbers) - 1, max(gen_numbers) + 1) 
plt.show()


# In[ ]:




optimisation = bpop.optimisations.DEAPOptimisation(
        evaluator=cell_evaluator2,
        offspring_size = MU,
        map_function = dask_map_function,
        selector_name='IBEA',mutpb=0.1,cxpb=0.35,seeded_pop=[cp['final_pop'],cp['hall_of_fame']])#,seeded_current=target_current)
final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=50)


# In[ ]:


numbers = sweep_number# = numbers #= sets[-1].get_sweep_numbers()  
sweeps;


# In[ ]:


import glob
sets = []
#/home/user/Dropbox (ASU)/AllenDruckmanData/cell_types/
temp_paths = list(glob.glob("/home/user/Dropbox (ASU)/AllenDruckmanData/cell_types/*/*.nwb"))#[cnt:1035+cnt] 
#print(temp_paths)
tests[3].name


# In[ ]:


temp_paths


# In[ ]:


#import glob

for f in temp_paths:
    try:
        sets = NwbDataSet(f)
        #print(dir(sets))

        numbers = sets.get_sweep_numbers()  


        #numbers = sweep_numbers['Long Square']
        #print(numbers)
        #break
        #greater_than_rheobase_numbers = [sweep_number for sweep_number in numbers if len(sets.get_spike_times(sweep_number))>1]
        rheobase,rh_index = get_rheobase(numbers,sets)
        sweeps = [sets.get_sweep(n) for n in numbers ]
        mapped_current_sweeps = [ (i,np.max(s['stimulus'])) for i,s in zip(numbers,sweeps) if 'stimulus' in s.keys()]#
        myNumber = 3.0*rheobase
        currents_ = [t[1] for t in mapped_current_sweeps]
        indexvm30 = closest(currents_, myNumber)
        sweep_data = sets.get_sweep(indexvm30)
        stim = sweep_data['stimulus']
        currents['30']=stim

        sampling_rate = sweep_data['sampling_rate']
        vm30 = AnalogSignal(sweep_data['response'],sampling_rate=sampling_rate*qt.Hz,units=qt.mV)
        plt.plot(vm30.times,vm30.magnitude)


        myNumber = 1.5*rheobase
        indexvm15 = closest(currents_, myNumber)
        sweep_data = sets.get_sweep(indexvm15)
        stim = sweep_data['stimulus']
        currents['15']=stim

        sampling_rate = sweep_data['sampling_rate']
        vm15 = AnalogSignal(sweep_data['response'],sampling_rate=sampling_rate*qt.Hz,units=qt.mV)
        plt.plot(vm15.times,vm15.magnitude)

        sweep_data = sets.get_sweep(rh_index)
        stim = sweep_data['stimulus']
        currents['rh']=stim
        sampling_rate = sweep_data['sampling_rate']
        vmrh = AnalogSignal(sweep_data['response'],sampling_rate=sampling_rate*qt.Hz,units=qt.mV)

        sm = StaticModel(vm = vmrh)
        sm.rheobase = rheobase
        sm.vm15 = vm15
        sm.vm30 = vm30
        sm = efel_evaluation(sm,thirty=False)

        sm = efel_evaluation(sm,thirty=True)
        sm = rekeyed(sm)
        useable = False
        sm.vmrh = vmrh
        plt.show()
        allen_tests = []
        if sm.efel_15 is not None:
            for k,v in sm.efel_15[0].items():
                try:
                    at = AllenTest(name=str(k)+'_1.5x')
                    at.set_observation(v)
                    allen_tests.append(at)
                    if "SpikeCount" in k:
                        print(v)
                        import pdb
                        pdb.set_trace()

                    if k in simple_yes_list:
                        useable = True

                except:
                    pass
        if sm.efel_30 is not None:
            for k,v in sm.efel_30[0].items():
                try:
                    if k in simple_yes_list:
                        print(k,v)
                        useable = True

                    if "SpikeCount" in k:
                        print(v)
                        import pdb
                        pdb.set_trace()
                    at = AllenTest(name=str(k)+'_3.0x')
                    at.set_observation(v)
                    allen_tests.append(at)
                except:
                    pass

        suite = TestSuite(allen_tests,name=str(f))
        suite.traces = None
        suite.traces = {}
        suite.traces['rh_current'] = sm.rheobase
        suite.traces['vmrh'] = sm.vmrh
        suite.traces['vm15'] = sm.vm15
        suite.traces['vm30'] = sm.vm30
        suite.model = None
        suite.useable = useable
        suite.model = sm

        suite.stim = None

        suite.stim = currents

        allen_test_suites.append(suite)

        #print(TSD(suite))
        pickle.dump(allen_test_suites,open('later_allen_NU_tests.p','wb'))    
    except:
        continue
'''
temp = [ (i,np.abs(s['stimulus'])) for i,s in zip(numbers,sweeps) if np.min(s['stimulus'])<0 ]
temp = sorted(temp,key=lambda x:[1],reverse=True)
print(sets[-1].get_sweep_metadata(temp[0][0]))

sweep_data = sets[-1].get_sweep(temp[0][0])
#sm = StaticModel() 
#print(sweep_data['index_range'])
stim = sweep_data['stimulus']
delay = sweep_data['index_range'][0]
duration = sweep_data['index_range'][1]
before = sweep_data['stimulus'][0:delay]
after = sweep_data['stimulus'][delay:duration]
#inj = AnalogSignal(sweep_data['stim'],sampling_rate=sampling_rate*qt.Hz,units=qt.pA)
r_in = (np.mean(after)-np.mean(before))/np.min(sweep_data['stimulus'])
print(r_in*qt.MOhm)
# sampling rate is in Hz
sampling_rate = sweep_data['sampling_rate']
#it= InputResistanceTest(observation={'mean':0*qt.MOhm}) 
#it.generate_prediction(sm)
vmm = AnalogSignal(sweep_data['response'],sampling_rate=sampling_rate*qt.Hz,units=qt.mV)
'''


# In[ ]:


2.5*3


# In[ ]:


cell_features = extract_cell_features(data_set,
                                      sweep_numbers['Ramp'],
                                      sweep_numbers['Short Square'],
                                      sweep_numbers['Long Square'])


# In[ ]:


get_ipython().run_cell_magic('capture', '', '#print([t.observation for t in tests[0].tests])\n#print([t.name for t in tests[0].tests])')


# In[ ]:





# Next we pick a random point between 
# a=0.01 and a=0.03
# we instance a model at this random location in a. 
# We then find rheobase and take efel measurements at 1.5 and 3.0 rheobase for the Izhikitich model. 
# 
# Then we use GA optimization to find the model that produced the measurements, as a type of inversion.
# 
# Jump to cells 27 and 28. You can see there that 
# 
# the error surface is highly intractable for some measurements, and for other measurements its a simple well with no ripples.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
big_list = []
for tt in aug_nu_tests.values():
    tt.core = None
    tt.core = True
    big_list.append(tt)
big_list.extend(nu_tests)
nu_tests = big_list
aug_nu_tests = list(aug_nu_tests.values())[0:4]
'''


# In[ ]:


#!ls -ltr *.p
import pickle

no_list = pickle.load(open("too_rippled_b.p","rb"))
yes_list = pickle.load(open("tame_b.p","rb"))
#yes_list ='RheobaseTest'
#yes_list = ['AHP_depth_abs_3.0x','sag_ratio2_3.0x','ohmic_input_resistance_1.5x','sag_ratio2_1.5x','peak_voltage_3.0x','peak_voltage_1.5x','voltage_base_3.0x','voltage_base_1.5x','Spikecount_1.5x','Spikecount_3.0x','ohmic_input_resistance_vb_ssse_1.5x']
#yes_list
yes_list = ['Spikecount_3.0x','ohmic_input_resistance_1.5x','sag_ratio2_1.5x','peak_voltage_1.5x','voltage_base_1.5x','Spikecount_1.5x','ohmic_input_resistance_vb_ssse_1.5x']


# In[ ]:


#yes_list = pickle.load(open("tame_b.p","rb"))
#yes_list


# In[ ]:


#yes_list = ['peak_voltage_3.0x','voltage_base_1.5x','ohmic_input_resistance_vb_ssse_1.5x','Spint(kecount_1.5x','Spikecount_3.0x']
#yes_list = pickle.load(open("tame_b.p","rb"))

#yes_list


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


check_bin_vm30(opt,opt)


# In[ ]:


check_bin_vm30(target,target)


# In[ ]:


print([o.name for o in objectives2])


# In[ ]:


opt.rheobase


# In[ ]:


target.rheobase


# In[ ]:


#def hof_to_euclid
def hof_to_euclid_4(hof,MODEL_PARAMS,target,ranges=True):
    lengths = {}
    tv = 1
    cnt = 0
    constellation0 = hof[0]
    constellation1 = hof[1]
    subset = list(MODEL_PARAMS.keys())
    tg = target.dtc_to_gene(subset_params=subset)
    if len(MODEL_PARAMS)==1:
        
        ax = plt.subplot()
        for k,v in MODEL_PARAMS.items():
            lengths[k] = np.abs(np.abs(v[1])-np.abs(v[0]))

            x = sorted([h[cnt] for h in hof])
            y = [np.sum(h.fitness.values) for h in hof]
            tgene = tg[cnt]
            yg = 0
        plt.plot(x,y)
            

        ax.scatter(x, y, c='b', marker='o',label='samples')
        ax.scatter(tgene, yg, c='r', marker='*',label='target')
        if ranges:
            ax.set_xlim(np.min(hof),np.max(hof))
            ax.set_xlabel(k)
        #ax.plot(x,y)

        ax.legend()

        plt.show()

#def animate(i):

def movie(hof,MODEL_PARAMS,target,max_height):
    lengths = {}
    tv = 1
    cnt = 0
    subset = list(MODEL_PARAMS.keys())
    tg = target.dtc_to_gene(subset_params=subset)
    if len(MODEL_PARAMS)==1:
        plt.clf()
        ax = plt.subplot()

        x =  hof[cnt] 
        y = sum(hof.fitness.values)
        tgene = tg[cnt]
        yg = 0

        ax.scatter(x, y, c='b', marker='o',label='samples')
        #ax.plot(x,y)
        ax.scatter(tgene, yg, c='r', marker='*',label='target')
        ax.set_xlim(np.min(MODEL_PARAMS[subset[0]]),np.max(MODEL_PARAMS[subset[0]]))
        ax.set_ylim(0,max_height)

        ax.set_xlabel(k)

        return ax,plt


# In[ ]:




def threshold_detection2(signal, threshold=0.0, sign='above'):
    """
    Returns the times when the analog signal crosses a threshold.
    Usually used for extracting spike times from a membrane potential. 
    Adapted from version in NeuroTools.   

    Parameters
    ----------
    signal : neo AnalogSignal object
        'signal' is an analog signal.
    threshold : A quantity, e.g. in mV  
        'threshold' contains a value that must be reached 
        for an event to be detected.
    sign : 'above' or 'below'
        'sign' determines whether to count thresholding crossings
        that cross above or below the threshold.  
    format : None or 'raw'
        Whether to return as SpikeTrain (None) 
        or as a plain array of times ('raw').

    Returns
    -------
    result_st : neo SpikeTrain object
        'result_st' contains the spike times of each of the events (spikes)
        extracted from the signal.  
    """

    assert threshold is not None, "A threshold must be provided"

    if sign is 'above':
        cutout = np.where(signal > threshold)[0]
    elif sign in 'below':
        cutout = np.where(signal < threshold)[0]

    if len(cutout) <= 0:
        events = np.zeros(0)
        return np.array([0])
    else:
        take = np.where(np.diff(cutout)>1)[0]+1
        take = np.append(0,take)
        return take
        #time = signal.times
        #events = time[cutout][take]
   # return len(events)


# In[ ]:


sub_MODEL_PARAMS = copy.copy(MODEL_PARAMS['IZHI'])


sub_MODEL_PARAMS
hof_to_euclid_4(hall_of_fame,sub_MODEL_PARAMS,target)
sub_MODEL_PARAMS
hof_to_euclid_4(list(hist.genealogy_history.values()),sub_MODEL_PARAMS,target)#,ranges=small)
subset = list(sub_MODEL_PARAMS.keys())
tg = target.dtc_to_gene(subset_params=subset)
tg
MODEL_PARAMS['IZHI']
tg


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

import pickle


from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
#untractable=[]
#tractable=[]
def ripple_detection(hof,MODEL_PARAMS,target,objectives2,max_height, show_hard=True):
    lengths = {}
    tv = 1
    cnt = 0
    subset = list(MODEL_PARAMS.keys())
    tg = target.dtc_to_gene(subset_params=subset)
    good_for_opt = []
    know_for_future_opt = []
    for i,f in enumerate(hof[0].fitness.values):    
        plt.clf()
        lengths[k] = np.abs(np.abs(v[1])-np.abs(v[0]))
        
        # debug change this 200.
        x_plot = np.linspace(np.abs(v[1]),np.abs(v[0]),200)

        X_plot = x_plot[:, np.newaxis]

        hof_x = sorted([ h for h in hof ])

        y = [ h.fitness.values[i] for h in hof_x ] 
        tgene = tg[cnt]
        yg = 0
        model = make_pipeline(PolynomialFeatures(6), Ridge())

        model.fit(hof_x, y)
        model.predict(X_plot)
        y_plot = model.predict(X_plot)
        y_thresh = np.mean(y_plot)
        n_crosses = threshold_detection2(y, threshold=y_thresh, sign='above')
        
        if n_crosses is not None:# is not None:
            untractable.append(objectives2[i].name)
            if show_hard:
                if len(n_crosses)>4:
                    ax = plt.subplot()
                    plt.title(objectives2[i].name)


                    plt.plot(hof_x,y)
                    ax.scatter(hof_x, y, c='b', marker='o',label='samples')
                    ax.scatter(tgene, yg, c='r', marker='*',label='target')
                    ax.set_xlim(np.min(hof),np.max(hof))
                    ax.set_xlabel(k)

                    plt.plot(X_plot, y_plot)
                    ax.legend()
                    plt.show()
                    know_for_future_opt.append(objectives2[i].name)
                
        if len(n_crosses)<5:
            tractable.append(tractable)
            if not show_hard:
                ax = plt.subplot()

                plt.title(objectives2[i].name)


                plt.plot(hof_x,y)
                ax.scatter(hof_x, y, c='b', marker='o',label='samples')
                ax.scatter(tgene, yg, c='r', marker='*',label='target')
                ax.set_xlim(np.min(hof),np.max(hof))
                ax.set_xlabel(k)

                plt.plot(X_plot, y_plot)
                ax.legend()
                plt.show()
                good_for_opt.append(objectives2[i].name)

    if show_hard:
        pickle.dump(know_for_future_opt,open('too_rippled.p','wb'))
    else:
        pickle.dump(good_for_opt,open('tame.p','wb'))

    return 


# In[ ]:


pop = list(hist.genealogy_history.values())
max_height = np.max([sum(i.fitness.values) for i in pop])

ripple_detection(pop,sub_MODEL_PARAMS,target,objectives2,max_height)#,ranges=small)


# In[ ]:


temp = list(hist.genealogy_history.values())
max_height = np.max([sum(i.fitness.values) for i in temp])
ripple_detection(list(hist.genealogy_history.values()),sub_MODEL_PARAMS,target,objectives2,max_height,show_hard=False)


# In[ ]:





# In[ ]:


pred_collector = []
sum_pred_collector = []


model.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}
opt = model.model_to_dtc()
opt.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}

opt.tests = aug_nu_tests

plotxvec = np.linspace(MODEL_PARAMS['IZHI']['a'][0],MODEL_PARAMS['IZHI']['a'][1],20)
for a in plotxvec:
    opt.attrs['a'] = a
    opt.self_evaluate()
    for t in opt.tests:
        try:
            t.prediction['value'] = t.prediction['mean']
        except:
            pass
    values = [test.prediction['value'] for test in opt.tests]
    pred_collector.append(values)
    sum_pred_collector.append(np.sum([values]))
#plt.title('sum of errors')
#plt.plot(plotxvec,sum_pred_collector)
#plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



def spikes2thresholds_debug(spike_waveforms):
    """
    IN:
     spike_waveforms: Spike waveforms, e.g. from get_spike_waveforms().
        neo.core.AnalogSignal
    OUT:
     1D numpy array of spike thresholds, specifically the membrane potential
     at which 1/10 the maximum slope is reached.

    If the derivative contains NaNs, probably because vm contains NaNs
    Return an empty list with the appropriate units

    """
    try:
        n_spikes = spike_waveforms.shape[1]
    except:
        return None
        #return thresholds * spike_waveforms.units


    thresholds = []
    if n_spikes > 1:
        # good to know can handle multispikeing
        pass
    for i in range(n_spikes):
        s = spike_waveforms[:, i].squeeze()
        s = np.array(s)
        dvdt = np.diff(s)
        for j in dvdt:
            if math.isnan(j):
                return thresholds * spike_waveforms.units
        try:
            trigger = dvdt.max()/10.0
        except:
            return None
            # try this next.
            # return thresholds * spike_waveforms.units

        try:
            x_loc = np.where(dvdt >= trigger)[0][0]
            
            thresh = (s[x_loc]+s[x_loc+1])/2
            
        except:
            thresh = None
        thresholds.append(thresh)
        plt.plot(s.times,s.magnitude)        
        plt.scatter(x_loc,thresh)
        plt.show()
        #plt.savefig("debug_threshold.png")
        
    return thresholds * spike_waveforms.units


# In[ ]:



best_ind = hall_of_fame[0]
best_ind_dict = cell_evaluator.param_dict(best_ind)
model = cell_evaluator.cell_model
cell_evaluator.param_dict(best_ind)
model.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}
opt = model.model_to_dtc()
opt.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}
opt = opt.attrs

#opt.tests = aug_nu_tests


# How do measurements
# from the standard suite vary 
# with changes in 'a'?
# 
# Note these are not scores or like in the above tests, but simply how do measurements change with changes in 'a'
# 
# In this we see sometimes 'a' causes ripples in very basic measurements.

# In[ ]:


names = [test.name for test in opt.tests]
import matplotlib.pyplot as plt


def smoothTriangle(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

for i in range(0,len(pred_collector[0])):
    plt.clf()
    plt.title(names[i])
    measurement = [val[i] for val in pred_collector]
    plt.plot(plotxvec,measurement)
    
    #from scipy.interpolate import spline
    #from scipy.interpolate import interp1d

   # xnew = np.linspace(plotxvec[0],plotxvec[1],300) #300 represents number of points to make between T.min and T.max

    #measurement_smooth = interp1d(T,measurement,xnew)

    
    #x = np.linspace(0, 10, num=11, endpoint=True)
    #y = np.cos(-x**2/9.0)
    #f1 = interp1d(plotxvec, measurement, kind='nearest')
    #f2 = interp1d(plotxvec, measurement, kind='zero')
    #f3 = interp1d(plotxvec, measurement, kind='quadratic')

    
    plt.plot(plotxvec,smoothTriangle(measurement,14))
    #plt.plot(plotxvec,f2)
    #plt.plot(plotxvec,f3)

    plt.show()


# In[ ]:


attrs = {k:np.mean(v) for k,v in MODEL_PARAMS["IZHI"].items()}
from neuronunit.capabilities.spike_functions import get_spike_waveforms, spikes2widths, spikes2thresholds
from quantities import ms
from neuronunit.tests.base import AMPL, DELAY, DURATION
from neuronunit.optimisation.data_transport_container import DataTC
for slider_value in np.linspace(0.01, 0.1, 100):
  #print(slider_value)

  dtc = DataTC(backend="IZHI",attrs=attrs)
  dtc.attrs['a'] = slider_value
  dtc = dtc_to_rheo(dtc)
  model = dtc.dtc_to_model()
  model.attrs = model._backend.default_attrs
  model.attrs.update(dtc.attrs)
  #print(model.attrs)
  uc = {'amplitude':dtc.rheobase,'duration':DURATION,'delay':DELAY}
  model._backend.inject_square_current(uc)
  vm = model.get_membrane_potential()
  #print(np.max(vm)-np.min(vm))
  snippets1 = get_spike_waveforms(vm)#,width=20*ms)
  spikes2thresholds_debug(snippets1)
  #plt.plot(snippets1.times,snippets1.magnitude) 
  plt.plot(vm.times,vm.magnitude) 
  #snippets1 = get_spike_waveforms(vm,width=10*ms)
  #print(spikes2widths(snippets1)[0])
  print(np.max(vm)-spikes2thresholds(snippets1)[0])
  #plt.plot(snippets1.times,snippets1.magnitude) 

plt.show()


# In[ ]:


attrs = {k:np.mean(v) for k,v in MODEL_PARAMS["IZHI"].items()}
from neuronunit.capabilities.spike_functions import get_spike_waveforms, spikes2widths, spikes2thresholds
from quantities import ms
from neuronunit.tests.base import AMPL, DELAY, DURATION
from neuronunit.optimisation.data_transport_container import DataTC
for slider_value in np.linspace(0.01, 0.1, 100):
  #print(slider_value)

  dtc = DataTC(backend="IZHI",attrs=attrs)
  dtc.attrs['a'] = slider_value
  dtc = dtc_to_rheo(dtc)
  model = dtc.dtc_to_model()
  model.attrs = model._backend.default_attrs
  model.attrs.update(dtc.attrs)
  #print(model.attrs)
  uc = {'amplitude':dtc.rheobase*1.5,'duration':DURATION,'delay':DELAY}
  model._backend.inject_square_current(uc)
  vm = model.get_membrane_potential()
  #print(np.max(vm)-np.min(vm))
  #snippets1 = get_spike_waveforms(vm)#,width=20*ms)
  plt.clf()
  #plt.plot(snippets1.times,snippets1.magnitude) 
  plt.plot(vm.times,vm.magnitude) 
  #snippets1 = get_spike_waveforms(vm,width=10*ms)
  #print(spikes2widths(snippets1)[0])
  #print(spikes2thresholds(snippets1)[0])
  #plt.plot(snippets1.times,snippets1.magnitude) 

  plt.show()

  plt.clf()
  uc = {'amplitude':dtc.rheobase*3.0,'duration':DURATION,'delay':DELAY}
  model._backend.inject_square_current(uc)
  vm = model.get_membrane_potential()
    
  #plt.plot(snippets1.times,snippets1.magnitude) 
  #plt.plot(vm.times,vm.magnitude) 
  snippets1 = get_spike_waveforms(vm,width=10*ms)
  print(spikes2widths(snippets1)[0])
  print(spikes2thresholds(snippets1)[0])
  #plt.plot(snippets1.times,snippets1.magnitude) 

  #plt.show()


# In[ ]:


plt.clf()
#measurement = [float(val[0])-155 for val in pred_collector]
#plt.plot(plotxvec,measurement)
#plt.semilogy()
plotxvec = np.linspace(MODEL_PARAMS['IZHI']['a'][0],MODEL_PARAMS['IZHI']['a'][1]*2,200)

measurement = [val[-1] for val in pred_collector]
plt.plot(plotxvec,measurement)

measurement = [float(val[-2])-75 for val in pred_collector]
plt.plot(plotxvec,measurement)
measurement = [float(val[-3]) for val in pred_collector]
plt.plot(plotxvec,measurement)

#plt.semilogy()
plt.show()


# In[ ]:


m = opt.dtc_to_model()
plotyvec = []
plotxvec = np.linspace(MODEL_PARAMS['IZHI']['a'][0],MODEL_PARAMS['IZHI']['a'][1]*2,100)
for a in plotxvec:
    m.attrs['a'] = a
    #print(opt.tests[-2].params)

    plotyvec.append(opt.tests[-2].generate_prediction(m)['value'])
plt.plot(plotxvec,plotyvec)
plt.show()


# In[ ]:


from neuronunit.optimisation.optimization_management import inject_and_plot_model, dtc_to_rheo
plotxvec = np.linspace(MODEL_PARAMS['IZHI']['a'][0],MODEL_PARAMS['IZHI']['a'][1],100)
for a in plotxvec:
    opt.attrs['a'] = a
    opt = dtc_to_rheo(opt)
    opt.self_evaluate()
    try:
        out = inject_and_plot_model(opt,plotly=False)
        print(out)
        #vm.show()

    #except:
        plt = inject_and_plot_model(opt,plotly=False)
        print(opt.rheobase)
    #for t in opt.tests:
    #    try:
    #        t.prediction['value'] = t.prediction['mean']
    #    except:
    #        pass
    #values = [test.prediction['value'] for test in opt.tests]
    #pred_collector.append(values)
    #sum_pred_collector.append(np.sum([values]))
#plt.title('sum of errors')
#plt.plot(plotxvec,sum_pred_collector)
#plt.show()


# In[ ]:



plotxvec = np.linspace(MODEL_PARAMS['IZHI']['a'][0],MODEL_PARAMS['IZHI']['a'][1],100)
for a in plotxvec:
    opt.attrs['a'] = a
    opt = dtc_to_rheo(opt)
    opt.self_evaluate()
    vm,plt = inject_and_plot_model(opt)
    plt.show()
    for t in opt.tests:
        try:
            t.prediction['value'] = t.prediction['mean']
        except:
            pass
    values = [test.prediction['value'] for test in opt.tests]
    pred_collector.append(values)
    sum_pred_collector.append(np.sum([values]))
plt.title('sum of errors')
plt.plot(plotxvec,sum_pred_collector)
plt.show()


# # are changes in b just as dramatic?

# In[ ]:


#from neuronunit.optimisation.model_parameters import MODEL_PARAMS as mp
print(mp['IZHI'])
pred_collector = []
sum_pred_collector = []

plotxvec = np.linspace(-2,15,100)
opt.attrs['a'] = np.mean(MODEL_PARAMS['IZHI']['a'])
for b in plotxvec:
    
    opt.attrs['b'] = b
    opt.self_evaluate()
    for t in opt.tests:
        try:
            t.prediction['value'] = t.prediction['mean']
        except:
            pass
    values = [test.prediction['value'] for test in opt.tests]
    pred_collector.append(values)
    sum_pred_collector.append(np.sum([values]))


# In[ ]:


#plt.title('sum of errors')
#plt.plot(plotxvec,sum_pred_collector)
#plt.show()

names = [test.name for test in opt.tests]

for i in range(0,len(pred_collector[0])):
    plt.clf()
    plt.title(names[i])
    plt.plot(plotxvec,[val[i] for val in pred_collector])
    plt.show()


# And **c**?

# In[ ]:


#from neuronunit.optimisation.model_parameters import MODEL_PARAMS as mp
print(mp['IZHI'])
pred_collector = []
sum_pred_collector = []

opt.attrs['a'] = np.mean(MODEL_PARAMS['IZHI']['a'])
opt.attrs['b'] = np.mean(plotxvec)
plotxvec = np.linspace(-60,-40,100)
 
for c in plotxvec:
    
    opt.attrs['c'] = c
    opt.self_evaluate()
    for t in opt.tests:
        try:
            t.prediction['value'] = t.prediction['mean']
        except:
            pass
    values = [test.prediction['value'] for test in opt.tests if test.prediction['value'] is not None]
    pred_collector.append(values)
    sum_pred_collector.append(np.sum([values]))

names = [test.name for test in opt.tests]

for i in range(0,len(pred_collector[0])):
    plt.clf()
    plt.title(names[i])
    plt.plot(plotxvec,[val[i] for val in pred_collector])
    plt.show()


# And **C**?

# In[ ]:


from neuronunit.optimisation.model_parameters import MODEL_PARAMS as mp
print(mp['IZHI'])
pred_collector = []
sum_pred_collector = []

#opt.attrs['a'] = np.mean(MODEL_PARAMS['IZHI']['a'])
#opt.attrs['b'] = np.mean(plotxvec)
opt.attrs['c'] = np.mean(plotxvec)

plotxvec = np.linspace(50,200,100)
 
for C in plotxvec:
    
    opt.attrs['C'] = C
    opt.self_evaluate()
    for t in opt.tests:
        try:
            t.prediction['value'] = t.prediction['mean']
        except:
            pass
    values = [test.prediction['value'] for test in opt.tests if test.prediction['value'] is not None]
    pred_collector.append(values)
    sum_pred_collector.append(np.sum([values]))

names = [test.name for test in opt.tests]

for i in range(0,len(pred_collector[0])):
    plt.clf()
    plt.title(names[i])
    plt.plot(plotxvec,[val[i] for val in pred_collector])
    plt.show()


# In[ ]:


from neuronunit.optimisation.model_parameters import MODEL_PARAMS as mp
print(mp['IZHI'])
pred_collector = []
sum_pred_collector = []

#opt.attrs['a'] = np.mean(MODEL_PARAMS['IZHI']['a'])
#opt.attrs['b'] = np.mean(plotxvec)
opt.attrs['C'] = np.mean(plotxvec)

plotxvec = np.linspace(10,150,100)
 
for d in plotxvec:
    
    opt.attrs['d'] = d
    opt.self_evaluate()
    for t in opt.tests:
        try:
            t.prediction['value'] = t.prediction['mean']
        except:
            pass
    values = [test.prediction['value'] for test in opt.tests if test.prediction['value'] is not None]
    pred_collector.append(values)
    sum_pred_collector.append(np.sum([values]))

names = [test.name for test in opt.tests]

for i in range(0,len(pred_collector[0])):
    plt.clf()
    plt.title(names[i])
    plt.plot(plotxvec,[val[i] for val in pred_collector])
    plt.show()


# In[ ]:


from neuronunit.optimisation.model_parameters import MODEL_PARAMS as mp
print(mp['IZHI'])
pred_collector = []
sum_pred_collector = []

#opt.attrs['a'] = np.mean(MODEL_PARAMS['IZHI']['a'])
#opt.attrs['b'] = np.mean(plotxvec)
opt.attrs['d'] = np.mean(plotxvec)

plotxvec = np.linspace(0.7,1.6,100)
#print(plotxvec)
for k in plotxvec:
    
    opt.attrs['k'] = k
    #print(opt.attrs)
    opt.self_evaluate()
    for t in opt.tests:
        try:
            t.prediction['value'] = t.prediction['mean']
        except:
            pass
    values = [test.prediction['value'] for test in opt.tests if test.prediction['value'] is not None]
    pred_collector.append(values)
    sum_pred_collector.append(np.sum([values]))

names = [test.name for test in opt.tests]

for i in range(0,len(pred_collector[0])):
    plt.clf()
    plt.title(names[i])
    plt.plot(plotxvec,[val[i] for val in pred_collector])
    plt.show()


# In[ ]:



from matplotlib import animation, rc
from IPython.display import HTML

get_ipython().run_line_magic('matplotlib', 'inline')
import time
import pylab as pl
from IPython import display
#for i in range(10):

temp = list(hist.genealogy_history.values())
max_height = np.max([sum(i.fitness.values) for i in temp])
#movies = True
#if movies:
for i,current in enumerate(list(hist.genealogy_history.values())):
    max_height
    ax,pl = movie(current,sub_MODEL_PARAMS,target,max_height)
    pl.title('gene number {0}'.format(i))
    display.clear_output(wait=True)
    display.display(pl.gcf())
    time.sleep(0.05)
    

    
    
    
#anim = animation.FuncAnimation(fig, animate, init_func=init,
#                               frames=100, interval=20, blit=True)    
    
#HTML(anim.to_html5_video())


# In[ ]:





# In[ ]:


target.attrs


# In[ ]:


hall_of_fame[1]




# In[ ]:





# In[ ]:





# In[ ]:





MU = 50
cp = {}
cp['halloffame'] = hall_of_fame
cp['population'] = final_pop
#seed_pop = cp['halloffame']
#cp = pickle.load(open('results_100.p', "rb"))

#seed_pop.extend(cp['pop'])
optimisation = bpop.optimisations.DEAPOptimisation(
        evaluator=cell_evaluator2,
        offspring_size = MU,
        map_function = dask_map_function,
        selector_name='IBEA',mutpb=0.1,cxpb=0.2),
        #seeded_pop=[cp['halloffame'],cp['population']])
final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=50)


# In[ ]:





# In[ ]:




# The optimisation has return us 4 objects: final population, hall of fame, statistical logs and history. 
# 
# The final population contains a list of tuples, with each tuple representing the two parameters of the model


#print('Final population: ', final_pop)


# The best individual found during the optimisation is the first individual of the hall of fame

best_ind = hall_of_fame[0]
#print('Best individual: ', best_ind)
#print('Fitness values: ', best_ind.fitness.va


# We can evaluate this individual and make use of a convenience function of the cell evaluator to return us a dict of the parameters


best_ind_dict = cell_evaluator.param_dict(best_ind)
#print(cell_evaluator.evaluate_with_dicts(best_ind_dict))


model = cell_evaluator.cell_model
cell_evaluator.param_dict(best_ind)
model.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}



opt = model.model_to_dtc()
opt.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}

check_binary_match(target,opt)
inject_and_plot_passive_model(opt,second=target,plotly=False)


# In[ ]:


gen_numbers = logs.select('gen')
min_fitness = logs.select('min')
max_fitness = logs.select('max')
avg_fitness = logs.select('avg')

plt.clf()
plt.plot(gen_numbers, max_fitness, label='max fitness')
plt.plot(gen_numbers, avg_fitness, label='avg fitness')
plt.plot(gen_numbers, min_fitness, label='min fitness')

plt.xlabel('generation #')
plt.ylabel('score (# std)')
plt.yscale('log')

plt.legend()
plt.xlim(min(gen_numbers) - 1, max(gen_numbers) + 1) 
#plt.ylim(0.9*min(min_fitness), 1.1 * max(min_fitness)) 
plt.show()


# In[ ]:


gen_numbers = logs.select('gen')
min_fitness = logs.select('min')
max_fitness = logs.select('max')
avg_fitness = logs.select('avg')

plt.clf()
fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(1,1)

plt.plot(gen_numbers, min_fitness, label='min fitness')
#ax.semilogy()
#ax.set_yscale('log')
plt.yscale('log')

plt.xlabel('generation #')
plt.ylabel('score (# std)')
plt.legend()
plt.xlim(min(gen_numbers) - 1, max(gen_numbers) + 1) 
#plt.ylim(0.9*min(min_fitness), 1.1 * max(min_fitness)) 
plt.show()


# In[ ]:


inject_and_plot_passive_model(opt,second=target,plotly=False)
best_ind_dict = cell_evaluator.param_dict(best_ind)
objectives = cell_evaluator.evaluate_with_dicts(best_ind_dict)


# In[ ]:


best_ind_dict = cell_evaluator.param_dict(best_ind)
objectives = cell_evaluator.evaluate_with_dicts(best_ind_dict)

model = cell_evaluator.cell_model
cell_evaluator.param_dict(best_ind)
model.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}

opt = model.model_to_dtc()
opt.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}
from neuronunit.optimisation.optimization_management import dtc_to_rheo, inject_and_plot_model30,check_bin_vm30,check_bin_vm15
opt = dtc_to_rheo(opt)
opt.rheobase
opt.attrs;


# In[ ]:






# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def hof_to_euclid(hof,MODEL_PARAMS,target):
    lengths = {}
    tv = 1
    cnt = 0
    constellation0 = hof[0]
    constellation1 = hof[1]
    subset = list(sub_MODEL_PARAMS.keys())
    tg = target.dtc_to_gene(subset_params=subset)
    if len(MODEL_PARAMS)==1:
        
        ax = plt.subplot()
        for k,v in MODEL_PARAMS.items():
            lengths[k] = np.abs(np.abs(v[1])-np.abs(v[0]))

            x = [h[cnt] for h in hof]
            y = [0 for h in hof]
            ax.set_xlim(v[0],v[1])
            ax.set_xlabel(k)
            tgene = tg[cnt]
            yg = 0

        ax.scatter(x, y, c='b', marker='o',label='samples')
        ax.scatter(tgene, yg, c='r', marker='*',label='target')
        ax.legend()

        plt.show()
    
    
    if len(MODEL_PARAMS)==2:
        
        ax = plt.subplot()
        for k,v in MODEL_PARAMS.items():
            lengths[k] = np.abs(np.abs(v[1])-np.abs(v[0]))
                
            if cnt==0:
                tgenex = tg[cnt]
                x = [h[cnt] for h in hof]
                ax.set_xlim(v[0],v[1])
                ax.set_xlabel(k)
            if cnt==1:
                tgeney = tg[cnt]

                y = [h[cnt] for h in hof]
                ax.set_ylim(v[0],v[1])
                ax.set_ylabel(k)
            cnt+=1

        ax.scatter(x, y, c='r', marker='o',label='samples',s=5)
        ax.scatter(tgenex, tgeney, c='b', marker='*',label='target',s=11)
        ax.legend()
        plt.show()
    if len(MODEL_PARAMS)==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for k,v in MODEL_PARAMS.items():
            lengths[k] = np.abs(np.abs(v[1])-np.abs(v[0]))
        
            if cnt==0:
                tgenex = tg[cnt]

                x = [h[cnt] for h in hof]
                ax.set_xlim(v[0],v[1])
                ax.set_xlabel(k)
            if cnt==1:
                tgeney = tg[cnt]

                y = [h[cnt] for h in hof]
                ax.set_ylim(v[0],v[1])
                ax.set_ylabel(k)
            if cnt==2:
                tgenez = tg[cnt]

                z = [h[cnt] for h in hof]
                ax.set_zlim(v[0],v[1])
                ax.set_zlabel(k)

            cnt+=1
        ax.scatter(x, y, z, c='r', marker='o')
        ax.scatter(tgenex, tgeney,tgenez, c='b', marker='*',label='target',s=21)

        plt.show()
        
sub_MODEL_PARAMS = copy.copy(MODEL_PARAMS['IZHI'])


sub_MODEL_PARAMS
hof_to_euclid(hall_of_fame,sub_MODEL_PARAMS,target)
sub_MODEL_PARAMS
hof_to_euclid(final_pop,sub_MODEL_PARAMS,target)
subset = list(sub_MODEL_PARAMS.keys())
tg = target.dtc_to_gene(subset_params=subset)
tg
MODEL_PARAMS['IZHI']
tg


# In[ ]:


#for gene in list(hist.genealogy_history.values()):
hof_to_euclid(list(hist.genealogy_history.values()),sub_MODEL_PARAMS,target)
sub_MODEL_PARAMS


# In[ ]:


from utils import basic_expVar


# In[ ]:


opt = dtc_to_rheo(opt)
print(opt.rheobase)
print(target.rheobase)


# In[ ]:



check_binary_match(opt,target,plotly=False,snippets=False)
check_binary_match(opt,target,plotly=False,snippets=True)
print(basic_expVar(target.vmrh, opt.vmrh), 'variancce explained ratio at rheobase')


# In[ ]:


params = {}
params['injected_square_current'] = {}
#if v.name in str('RestingPotentialTest'):
params['injected_square_current']['delay'] = PASSIVE_DELAY
params['injected_square_current']['duration'] = PASSIVE_DURATION
params['injected_square_current']['amplitude'] = 0.0*pq.pA    


# In[ ]:



opt_model = opt.dtc_to_model()
opt_model.inject_square_current(params)
opt_vm = opt_model.get_membrane_potential()
opt_vm[-1]


# In[ ]:


target_model = target.dtc_to_model()
target_model.inject_square_current(params)
target_vm = target_model.get_membrane_potential()
target_vm[-1]


# In[ ]:


best_ind = hall_of_fame[1]
#print('Best individual: ', best_ind)
#print('Fitness values: ', best_ind.fitness.values)


# We can evaluate this individual and make use of a convenience function of the cell evaluator to return us a dict of the parameters


best_ind_dict = cell_evaluator.param_dict(best_ind)
#print(cell_evaluator.evaluate_with_dicts(best_ind_dict))


model = cell_evaluator.cell_model
dtc= model.model_to_dtc()
opt = dtc_to_rheo(opt)
print(opt.rheobase)
print(target.rheobase)


# In[ ]:


#model.rheobase
objectives


# In[ ]:



best_ind_dict = cell_evaluator.param_dict(best_ind)
objectives = cell_evaluator.evaluate_with_dicts(best_ind_dict)

objectives2 = cell_evaluator2.evaluate_with_dicts(best_ind_dict)


# In[ ]:


import pandas as pd
import seaborn as sns
logbook = logs
#scores = [ m for m in logs ]
'''
list_of_dicts = []
df1 = pd.DataFrame()
genes=[]
for _,v in hist.genealogy_history.items():
    genes.append(v.fitness.values)
for j,i in enumerate(objectives.keys()):
    index = i.split('.')[0]
    df1[str(index)] = pd.Series(genes).values[j]#, index=df1.index)

'''
#MU =14
genes=[]
min_per_generations = []
for i,v in hist.genealogy_history.items():
    if i%MU==0:
        min_per_gen = sorted([(gene,np.min(gene)) for gene in genes],key=lambda x: x[1])
        min_per_generations.append(min_per_gen[0][0])
        genes =[]
    genes.append(v.fitness.values)
    
df2 = pd.DataFrame()
scores = []
for j,i in enumerate(objectives.keys()):
    index = i.split('.')[0]
    print([i[j] for i in min_per_generations ])
    df2[index] = pd.Series([i[j] for i in min_per_generations ])#, index=df1.index)
df2    


# In[ ]:


number=int(np.sqrt(len(df2.columns)))


# In[ ]:


import math
box = int(np.sqrt(len(objectives)))
fig,axes = plt.subplots(box,box+1,figsize=(20,20))#math.ceil(len(objectives)/2+1),figsize=(20,20))
#axes[0,0].plot(scores)
#axes[0,0].plot(gen_numbers, min_fitness, label='min fitness')

axes[0,0].set_title('Observation/Prediction Disagreement')
for i,c in enumerate(df2.columns):
    ax = axes.flat[i+1]
    history = df2[c]
    #mn = mean[k.name] 
    #st = std[k.name] 
    #history = [(j[i]-mn)/st for j in scores ]
    #ax.axhline(y=mn , xmin=0.02, xmax=0.99,color='red',label='best candidate sampled')

    #ax.axvline(x=min_x , ymin=0.02, ymax=0.99,color='blue',label='best candidate sampled')
    ax.plot(history)
    ax.set_title(str(c))
    #bigger = np.max([np.max(history),mn])
    #smaller = np.max([np.min(history),mn])

    #ax.set_ylim([np.min(history),np.max(history)])
    #ax.set_ylabel(str(front[0].dtc.tests[i].observation['std'].units))
axes[0,0].set_xlabel("Generation")
axes[0,0].set_ylabel("standardized error")

plt.tight_layout()
#if figname is not None:
#    plt.savefig(figname)
#else:
plt.show()


# In[ ]:



logbook = logs
#scores = [ m for m in logs ]
list_of_dicts = []
df1 = pd.DataFrame()
genes=[]
for _,v in hist.genealogy_history.items():
    genes.append(v.fitness.values)
for j,i in enumerate(objectives2.keys()):
    index = i.split('.')[0]
    df1[str(index)] = pd.Series(genes).values[j]#, index=df1.index)


    
#if normalize:
#    a = (a - mean(a)) / (std(a) * len(a))
#    v = (v - mean(v)) /  std(v)

df1=(df1-df1.mean())/df1.std()

corr = df1.corr()#.normalize()
fig =plt.figure(figsize=(10,10))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)#, annot=True)
plt.show()
print(np.sum(np.sum(corr)))


# In[ ]:



    
logbook = logs
list_of_dicts = []
df1 = pd.DataFrame()
genes=[]
for _,v in hist.genealogy_history.items():
    genes.append(v.fitness.values)
for j,i in enumerate(objectives2.keys()):
    index = i.split('.')[0]
    df1[str(index)] = pd.Series(genes).values[j]#, index=df1.index)


    
#if normalize:
#    a = (a - mean(a)) / (std(a) * len(a))
#    v = (v - mean(v)) /  std(v)

df1=(df1-df1.mean())/df1.std()

corr = df1.corr()#.normalize()
fig =plt.figure(figsize=(10,10))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)#, annot=True)
plt.show()
print(np.sum(np.sum(corr)))


# In[ ]:


hist.genealogy_history.keys();


# In[ ]:





# In[ ]:


hist.genealogy_tree.keys();


# In[ ]:


import networkx
graph = networkx.DiGraph(hist.genealogy_tree)
graph = graph.reverse()     # Make the graph top-down
per_generation = [(gen,hist.genealogy_history[gen].fitness.values) for gen in graph]


# In[ ]:


per_generation


# In[ ]:


graph.nodes


# In[ ]:


cp['halloffame'][-1].fitness.values


# In[ ]:


cp['halloffame'][0].fitness.values


# In[ ]:


cp['population'][-1].fitness.values


# In[ ]:


final_pop 
hall_of_fame[-1].fitness.values


# In[ ]:


len(hall_of_fame)


# In[ ]:


plt.plot([i for i in range(len(hall_of_fame),0,-1)],[np.sum(i.fitness.values) for i in hall_of_fame])


# In[ ]:


plt.plot([i for i in range(0,len(hall_of_fame))],[np.sum(hall_of_fame[i].fitness.values) for i in range(len(hall_of_fame)-1,-1,-1)])


# In[ ]:


genes=[]

for v in hall_of_fame:
    #v = hall_of_fame[j]
    genes.append(v.fitness.values)

#for i in 
#    plt.plot([i for i in range(0,len(hall_of_fame))],[hall_of_fame[i].fitness.values[j] for i in range(len(hall_of_fame)-1,-1,-1)])
#    plt.show()
df1 = pd.DataFrame()


# In[ ]:


for j,i in enumerate(objectives.keys()):
    index = i.split('.')[0]
    df1[str(index)] = pd.Series(genes).values[j]#, index=df1.index)

df1=(df1-df1.mean())/df1.std()

corr = df1.corr()
fig =plt.figure(figsize=(10,10))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)#, annot=True)
plt.show()


# In[ ]:


for j,i in enumerate(objectives2.keys()):
    index = i.split('.')[0]
    df1[str(index)] = pd.Series(genes).values[j]#, index=df1.index)

df1=(df1-df1.mean())/df1.std()

corr = df1.corr()
fig =plt.figure(figsize=(10,10))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)#, annot=True)
plt.show()


# In[ ]:





# In[ ]:





# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:




