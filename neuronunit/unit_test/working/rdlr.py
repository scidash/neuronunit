
# coding: utf-8

# # Set up the environment

import matplotlib.pyplot as plt
import matplotlib
import hide_imports
from neuronunit.optimisation.opt_man import inject_and_plot_model, inject_and_plot_passive_model, check_binary_match
import copy
import pickle
from neuronunit.optimisation.opt_man import TSD
from collections import OrderedDict
import pickle
import sys
from neuronunit.optimisation.opt_man import contrast

# # Design simulated data tests
from neuronunit.optimisation.opt_man import check_match_front, jrt
#from neuronunit.optimisation.optimization_management import get_agreement
    
def data_driven_tests(backend,MU,NGEN,t_index):
    test_frame = pickle.load(open('processed_multicellular_constraints.p','rb'))


    od = OrderedDict(test_frame)
    test_name = list(od.keys())[t_index]
    lt = list(od.values())[t_index]
    if not len(lt.tests): 
        return
    this_test = TSD(lt.tests)

    #hc = {'C':this_test['CapacitanceTest'].observation['mean'],\
    #        'vr':this_test['RestingPotentialTest'].observation['mean'],\
    #        'vt':this_test['InjectedCurrentAPThresholdTest'].observation['mean'],\
    #        'vPeak':float(this_test['InjectedCurrentAPAmplitudeTest'].observation['mean'])}

    #def test_and_destroy_cycles(free_params,hc):_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    #    for i in range(0,3):


    tt = copy.copy(this_test)
    ga_out = tt.optimize(backend=backend,protocol={'allen': False, 'elephant': True},
                                        MU=MU,NGEN=NGEN,free_params=['a','b'],
                                        figname='two_param')

    opt1 = ga_out['pf'][0].dtc
    OM = opt1.dtc_to_opt_man()

    opt1= OM.get_agreement(opt1)
    model = ga_out['pf'][0].dtc.dtc_to_model()
    temp = model.default_attrs
    temp1 = copy.copy(ga_out['pf'][0].dtc.attrs)
    temp.update(temp1)
    temp.pop('C',None)    
    temp.pop('vt',None)    
    temp.pop('vPeak',None)    
    temp['vr'] = float(temp['vr'])
    hc = temp    
    for d in ga_out['pf']: 
        del d.dtc
    with open('protect_RAM.p','wb') as f:
        pickle.dump(ga_out['pf'],f)

    del_list = [ga_out,tt, OM]
    for d in del_list:
        del d
    del ga_out
    
    del tt
    del OM
    with open('protect_RAM.p','rb') as f:
        seed_pop = pickle.load(f)
    tt = copy.copy(this_test)


    import sys
    for i in dir():exec('sys.getsizeof('+str(i)+')')

    ga_out = this_test.optimize(seed_pop=seed_pop,backend=backend,\
                                protocol={'allen': False, 'elephant': True},\
                                MU=MU,NGEN=NGEN,free_params=['C','d'],figname='other_two_param')
    ga_out['DO'] = None
    front = ga_out['pf']
    for d in front:
        d.dtc.tests.DO = None
        d.dtc.tests = d.dtc.tests.to_dict()
    front = [ind.dtc for ind in ga_out['pf']]

    opt = ga_out['pf'][0].dtc
    OM = opt.dtc_to_opt_man()
 
    opt = OM.get_agreement(opt)

    hist = ga_out['history'].genealogy_history
    hist_val = hist.values()
    
    get_max = [(sum(j.fitness.values),i) for i,j in enumerate(hist_val)]
    worst = sorted(get_max,key = lambda x: x[0])[-1][1]+1
    worst_dtc = ga_out['history'].genealogy_history[worst].dtc
    contrast(opt,worst_dtc,figname='contrast_best_worst'+str('MU_')+str(MU)+('_NGEN_')+str(NGEN)+str(backend)+'_.png')

    inject_and_plot_model(opt,figname ='optimal_active_waveform'+str('MU_')+str(MU)+('_NGEN_')+str(NGEN)+str(backend)+'_.png')
    inject_and_plot_passive_model(opt,figname ='optimal_passive_wave_form'+str('MU_')+str(MU)+('_NGEN_')+str(NGEN)+str(backend)+'_.png')#,figname=None)
    #with open('.p','wb') as f:
    #    pickle.dump([opt.obs_preds],f)


    #sim_data = pickle.load(open('sim data.p','rb'))
    return [ga_out['log'],front,opt,test_name]

#MUrange =
NGEN = 10
MU = 10

backend = str("RAW")

test_frame = pickle.load(open('processed_multicellular_constraints.p','rb'))

#for MU in range(20,140,20):
for t_index in range(0,len(test_frame)):
    
    break
#import pdb
#pdb.set_trace()
t_index+=1
out = data_driven_tests(backend,MU,NGEN,t_index)
#if type(out) is type(None):
#    continue        
ga_out,front,opt,test_name = out
results = [ga_out,front,opt,test_name]
with open('real_data_'+str(MU)+str(test_name)+'_.p','wb') as f:
    pickle.dump(results,f)
sys.exit()
#backend = str("HH")
#ga_out,target,front = sim_data_tests(backend,MU,NGEN)
