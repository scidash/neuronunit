
# coding: utf-8

# # Bootstrap the ipython cluster via operating system shell utils:
# This notebook needs 3 kernel restart and runs before it runs to completion for reasons that are unclear. Wait about 15 seconds between each kernel restart.

# In[1]:


#import os
#os.system('ipcluster start -n 8 --profile=default & sleep 15 ; python stdout_worker.py &')
import sys
import logging
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# In[2]:


import matplotlib
#get_ipython().run_line_magic('matplotlib', 'inline')
import ipyparallel as ipp
rc = ipp.Client(profile='default')
from ipyparallel import depend, require, dependent
dview = rc[:]


# ################
# # GA parameters:
# about 25*15=375 models will be made, excluding rheobase search.
# ################

# In[3]:


MU = 15; NGEN = 50; CXPB = 0.9
USE_CACHED_GA = False


# ################
# # Grid search parameters:
# 2^10=1024 models, will be made excluding rheobase search
# ################

# In[4]:
npoints = 2
nparams = 10


#npoints = 2
#nparams = 10
from neuronunit.optimization.model_parameters import model_params
provided_keys = list(model_params.keys())
USE_CACHED_GS = True


if USE_CACHED_GS:
       #dtcpopg = pickle.load(open('grid_dump.p','rb'))
    import pickle
    #grid_dump_second_3rd.p
    first_third = pickle.load(open('grid_dump_first_3rd.p','rb'))
    second_third = pickle.load(open('grid_dump_second_3rd.p','rb'))
    final_third = pickle.load(open('grid_dump_final_3rd.p','rb'))

    second_third.extend(first_third)
    second_third.extend(final_third)
    dtcpopg = second_third
else:
    from neuronunit.optimization import exhaustive_search
    grid_points = exhaustive_search.create_grid(npoints = npoints,nparams = nparams)#,provided_keys = provided_keys )

    #grid_points = exhaustive_search.create_grid(npoints = npoints,nparams = nparams,provided_keys = provided_keys )
    dlist = list(dview.map_sync(exhaustive_search.update_dtc_pop,grid_points))
    from neuronunit.optimization import get_neab
    for d in dlist:
        d.model_path = get_neab.LEMS_MODEL_PATH
        d.LEMS_MODEL_PATH = get_neab.LEMS_MODEL_PATH

    # this is a big load on memory so divide it into thirds.

    dlist_first_third = dlist[0:int(len(dlist)/3)]
    dlist_second_third = dlist[int(len(dlist)/3):int(2*len(dlist)/3)]
    dlist_final_third = dlist[int(2*len(dlist)/3):-1]
    from neuronunit.optimization.exhaustive_search import dtc_to_rheo
    from neuronunit.optimization.nsga_parallel import nunit_evaluation


    def compute_half(dlist_half):
        dlist_half = list(map(dtc_to_rheo,dlist_half))
        dlist_half = dview.map_sync(nunit_evaluation,dlist_half)
        return dlist_half

    dlist_first_3rd = compute_half(dlist_first_third)
    import pickle
    with open('grid_dump_first_3rd.p','wb') as f:
       pickle.dump(dlist_first_3rd,f)
    # Garbage collect a big memory burden.
    dlist_first_3rd = None
    dlist_second_3rd = compute_half(dlist_second_third)

    with open('grid_dump_second_3rd.p','wb') as f:
       pickle.dump(dlist_second_3rd,f)
   # Garbage collect a big memory burden.
    dlist_second_3rd = None

    dlist_final_3rd = compute_half(dlist_final_third)
    with open('grid_dump_final_3rd.p','wb') as f:
       pickle.dump(dlist_final_3rd,f)
    # Garbage collect a big memory burden.
    dlist_final_3rd = None
    first_third = pickle.load(open('grid_dump_first_3rd.p','rb'))
    second_third = pickle.load(open('grid_dump_second_3rd.p','rb'))
    final_third = pickle.load(open('grid_dump_final_3rd.p','rb'))

    second_third.extend(first_third)
    second_third.extend(final_third)
    dtcpopg = second_third

dtcpopg = [ dtc for dtc in dtcpopg if not None in (dtc.scores.values()) ]
dtcpopg = [ (dtc,sum(list(dtc.scores.values()))) for dtc in dtcpopg ]
sorted_grid = sorted(dtcpopg,key=lambda x:x[1])
sorted_grid = [dtc[0] for dtc in sorted_grid]

'''
sorted_grid = sorted([(dtc,sum(list(dtc.scores.values()))) for dtc in dtcpopg],key=lambda x:x[1])
sorted_grid = [dtc[0] for dtc in sorted_grid]
'''
minimagr = sorted_grid[0]

# An oppurtunity to improve grid search, by increasing resolution of search intervals given a first pass:

# In[5]:


REFINE_GRID = True


# In[6]:


import pickle
import numpy as np


# In[7]:


from neuronunit.optimization.nsga_object import NSGA
from neuronunit.optimization import exhaustive_search as es
#from neuronunit.optimization import evaluate_as_module as eam


# In[8]:



if USE_CACHED_GA:
    from deap import creator
    from deap import base
    #from neuronunit.optimization import evaluate_as_module as eam
    #NSGAO = NSGA(CXPB)
    #NSGAO.setnparams(nparams=nparams,provided_keys=provided_keys)

    from bluepyopt.deapext.optimisations import DEAPOptimisation
    DO = DEAPOptimisation()
    DO.setnparams(nparams = nparams, provided_keys = provided_keys)
    # not a real run.
    #pop, hof, log, history, DO.td  = DO.run(offspring_size = 2, max_ngen = 0)
    #print(DO.td)

    [pop, log, history, hof, td] = pickle.load(open('ga_dump.p','rb'))


else:
    from bluepyopt.deapext.optimisations import DEAPOptimisation
    print(DEAPOptimisation)
    DO = DEAPOptimisation()
    DO.setnparams(nparams = nparams, provided_keys = provided_keys)


    #continue_cp=continue_cp,
    #)

    pop, hof, log, history, td = DO.run(offspring_size = MU, max_ngen = NGEN, cp_frequency=4,cp_filename='checkpointedGA.p')
    with open('ga_dump.p','wb') as f:
       pickle.dump([pop, log, history, hof, td],f)

def pop2dtc(pop1,DO):
    '''
    This function takes the DEAP population data type, and converts it to a more convenient
    data transport object, which can more readily be used in plotting functions.
    This a wasteful, recompute, which is in part necessitated because
    deaps pareto front object, only returns gene individual objects (elements of population)
    '''
    #from neuronunit.optimization import evaluate_as_module as eam
    from neuronunit.optimization import nsga_parallel
    return_package = nsga_parallel.update_pop(pop1,DO.td)
    dtc_pop = []
    for i,r in enumerate(return_package):
        dtc_pop.append(r[0])
        dtc_pop[i].error = None
        dtc_pop[i].error = np.sqrt(np.mean(np.square(list(pop1[i].fitness.values))))
    #sorted_list  = sorted([(dtc,_) for dtc, dtc.error in dtc_pop],key=lambda x:x[1])
    sorted_list  = sorted([(dtc,dtc.error) for dtc in dtc_pop],key=lambda x:x[1])
    dtc_pop = [dtc[0] for dtc in sorted_list]
    print(dtc_pop,sorted_list)
    return dtc_pop

DO.td = td
dtc_pop = pop2dtc(hof,DO)

miniga = dtc_pop[0].error
maxiga = dtc_pop[-1].error
maximaga = dtc_pop[-1]
minimaga = dtc_pop[0]

# In[9]:


import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size':16})
import matplotlib.pyplot as plt
all_inds = history.genealogy_history.values()
sums = np.array([np.sum(ind.fitness.values) for ind in all_inds])
quads = []
#from neuronunit.optimization import evaluate_as_module as eam
#td = eam.get_trans_dict(DO.params)

#td = eam.get_trans_dict(NSGAO.subset)
for k in range(1,9):
    for i,j in enumerate(td):
        if i+k < 10:
            quads.append((td[i],td[i+k],i,i+k))
all_inds1 = list(history.genealogy_history.values())


def plot_surface(x,z,td):
    ab = [ (all_inds1[x],all_inds1[z]) for y in all_inds1 ]
    xs = np.array([ind[x] for ind in all_inds])
    ys = np.array([ind[z] for ind in all_inds])
    min_ys = ys[np.where(sums == np.min(sums))]
    min_xs = xs[np.where(sums == np.min(sums))]
    plt.clf()
    fig_trip, ax_trip = plt.subplots(1, figsize=(10, 5), facecolor='white')
    trip_axis = ax_trip.tripcolor(xs,ys,sums,20,norm=matplotlib.colors.LogNorm())
    plot_axis = ax_trip.plot(list(min_xs), list(min_ys), 'o', color='lightblue',label='global minima')
    fig_trip.colorbar(trip_axis, label='Sum of Objective Errors ')
    ax_trip.set_xlabel('Parameter '+str(list(td.values())[x]))
    ax_trip.set_ylabel('Parameter '+str(list(td.values())[z]))
    plot_axis = ax_trip.plot(list(min_xs), list(min_ys), 'o', color='lightblue')
    fig_trip.tight_layout()
    plt.savefig('surface'+str(list(td.values())[z])+'.png')
    #fig_trip.show()


# # Below two error surface slices from the hypervolume are plotted.
# The data that is plotted consists of the error as experienced by the GA.
# Note: the GA performs an incomplete, and efficient sampling of the parameter space, and therefore sample points are irregularly spaced. Polygon interpolation is used to visualize error gradients. Existing plotting code from the package BluePyOpt has been extended for this purpose.
# Light blue dots indicate local minima's of error experienced by the NSGA algrorithm.

# In[10]:

plot_surface(4,-3,td)


# In[11]:

plot_surface(1,2,td)


# In[12]:

# # A small administrative burden to those who intend to extend this code
#
# The precomputed grid run, loads in two large pickle files, as they were initially divided into halves
# Code has been re-written to use data blocks divided in thirds, and to collate the thirds into one, however,
# An appropriate pickle load for loading and collating third data blocks has not been re-written, and it is assumed to be a trivial task.

# In[13]:

def error(dtc):
    """
    Overall error function for a DTC
    Returns the root-mean-square error over all the tests
    """
    return np.sqrt(np.mean(np.square(list(dtc.scores.values()))))

def sorted_dtcs(dtcpop):
    """
    Returns dtc,error tuples sorted from low to high error
    """
    return sorted([(dtc,error(dtc)) for dtc in dtcpop],key=lambda x:x[1])


def sorted_history(pop):
    """
    Returns dtc,error tuples sorted from low to high error
    """
    return sorted([ind.fitness for ind in pop],key=lambda x:x[1])


minimagr_dtc, maxi = sorted_dtcs(dtcpopg)[-1]
minimagr_dtc, mini = sorted_dtcs(dtcpopg)[0]

minimagr_dtc_1, mini_1 = sorted_dtcs(dtcpopg)[1]
minimagr_dtc_2, mini_2 = sorted_dtcs(dtcpopg)[2]

# Here we have the best, second and third best model parameters, and create a news
# fine grained new search interval
# using them.

from neuronunit.optimization.exhaustive_search import create_refined_grid
refined_grid = create_refined_grid(minimagr_dtc, minimagr_dtc_1,minimagr_dtc_2)
print(refined_grid)




# The following plot shows how population diversity (std deviation) increases,
# and simultaneously mean error follows a net downward
# Trajectory.

# In[ ]:

do_once = list(sorted_dtcs(dtcpopg))
minimagr_dtc, mini = do_once[0]
maximagr, maxi = do_once[-1]
minimagr_dtc_1, mini_1 = do_once[1]
minimagr_dtc_2, mini_2 = do_once[2]

# In[ ]:




#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')
gen_numbers =[ i for i in range(0,len(log.select('gen'))) ]
mean = np.array([ np.sum(i) for i in log.select('avg')])
std = np.array([ np.sum(i) for i in log.select('std')])
minimum = np.array([ np.sum(i) for i in log.select('min')])
stdminus = mean - std
stdplus = mean + std

axes.plot(
    gen_numbers,
    mean,
    color='black',
    linewidth=2,
    label='population average')
axes.fill_between(gen_numbers, stdminus, stdplus)
axes.plot(gen_numbers, stdminus, label='std variation lower limit')
axes.plot(gen_numbers, stdplus, label='std variation upper limit')
axes.set_xlim(np.min(gen_numbers) - 1, np.max(gen_numbers) + 1)
axes.set_xlabel('Generations')
axes.set_ylabel('Sum of objectives')
axes.set_ylim([np.min(stdminus), np.max(stdplus)])
axes.legend()
fig.tight_layout()
fig.savefig('evolution.png')

# In[ ]:

#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
plt.clf()
plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')
gen_numbers =[ i for i in range(0,len(log.select('gen'))) ]
mean = np.array([ np.sqrt(np.mean(np.square(i))) for i in log.select('avg')])
std = np.array([ np.sqrt(np.mean(np.square(i))) for i in log.select('std')])
minimum = np.array([ np.sqrt(np.mean(np.square(i))) for i in log.select('min')])
best_line = np.array([ np.sqrt(np.mean(np.square(list(h.fitness.values))))  for h in hof])
bl = [b for b in best_line if not np.isnan(b) ]
bl = [ min(bl) for i in gen_numbers]
blg = [ mini for i in gen_numbers ]


stdminus = mean - std
stdplus = mean + std
try:
    assert len(gen_numbers) == len(stdminus) == len(stdplus)
except:
    pass

axes.plot(
    gen_numbers,
    mean,
    color='black',
    linewidth=2,
    label='population average')
axes.fill_between(gen_numbers, stdminus, stdplus)
axes.plot(gen_numbers, bl, linewidth=2, label='pareto front error')
axes.plot(gen_numbers, blg, linewidth=2, label='grid search error')

axes.plot(gen_numbers, stdminus, label='std variation lower limit')
axes.plot(gen_numbers, stdplus, label='std variation upper limit')
axes.set_xlim(np.min(gen_numbers) - 1, np.max(gen_numbers) + 1)
axes.set_xlabel('Generations')
axes.set_ylabel('Sum of objectives')
axes.legend()
fig.tight_layout()
fig.savefig('evolution_of_objectives.png')

# In[ ]:


def use_dtc_to_plotting(dtcpop,minimagr):
    from neuronunit.capabilities import spike_functions
    import matplotlib.pyplot as plt
    import numpy as np
    plt.clf()
    plt.style.use('ggplot')
    fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')
    stored_min = []
    stored_max = []
    for dtc in dtcpop[1:-1]:
        plt.plot(dtc.tvec, dtc.vm0,linewidth=3.5, color='grey')
        stored_min.append(np.min(dtc.vm0))
        stored_max.append(np.max(dtc.vm0))

    from neuronunit.models.reduced import ReducedModel
    from neuronunit.optimization.get_neab import tests as T
    from neuronunit.optimization import get_neab
    #from neuronunit.optimization import evaluate_as_module
    from neuronunit.optimization.nsga_object import pre_format
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    import neuron
    model._backend.reset_neuron(neuron)
    model.set_attrs(minimagr.attrs)
    model.rheobase = minimagr.rheobase['value']
    minimagr = pre_format(minimagr)
    parameter_list = list(minimagr.vtest.values())
    #print(parameter_list[0])
    model.inject_square_current(parameter_list[0])
    model._backend.local_run()
    assert model.get_spike_count() == 1
    print(model.get_spike_count(),bool(model.get_spike_count() == 1))
    brute_best = list(model.results['vm'])

    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    import neuron
    model._backend.reset_neuron(neuron)
    model.set_attrs(dtc_pop[0].attrs)
    model.rheobase = dtc_pop[0].rheobase['value']
    dtc_pop[0] = pre_format(dtc_pop[0])
    parameter_list = list(dtc_pop[0].vtest.values())
    #print(parameter_list[0])
    model.inject_square_current(parameter_list[0])
    model._backend.local_run()
    assert model.get_spike_count() == 1
    print(model.get_spike_count(),bool(model.get_spike_count() == 1))
    ga_best = list(model.results['vm'])

    #minimagr
    plt.plot(dtcpop[0].tvec, brute_best,linewidth=1, color='blue',label='best candidate via Grid')#+str(mini))
    plt.plot(dtcpop[0].tvec, ga_best,linewidth=1, color='red',label='best candidate via GA')#+str(minimagr))
    plt.legend()
    plt.ylabel('$V_{m}$ mV')
    plt.xlabel('ms')
    #axes.legend()
    fig.tight_layout()
    plt.savefig('grid_versus_bpo.png')
    #plt.show()
from neuronunit import plottools
from neuronunit.plottools import dtc_to_plotting


CACHE_PLOTTING = False
if CACHE_PLOTTING == False:
    #sorted_list_pf, pareto_dtc = pop2dtc(pf,NSGAO)
    dtc_pop = dview.map_sync(dtc_to_plotting, dtc_pop )
    with open('plotting_dump.p','wb') as f:
       pickle.dump(dtc_pop,f)
else:
     invalid_dtc  = pickle.load(open('plotting_dump.p','rb'))

use_dtc_to_plotting(dtc_pop,minimagr_dtc)
'''
invalid_dtc = dview.map_sync(dtc_to_plotting, pareto_dtc)
use_dtc_to_plotting(invalid_dtc,minimagr_dtc)
'''

# # Quantize distance between minimimum error and maximum error.
# This step will allow the GA's performance to be located within or below the range of error found by grid search.
#

# In[ ]:


quantize_distance = list(np.linspace(mini,maxi,10))

# check that the nsga error is in the bottom 1/5th of the entire error range.
print('Report: ')
print("Success" if bool(miniga < quantize_distance[0]) else "Failure")
print("The nsga error %f is in the bottom 1/5th of the entire error range" % miniga)
print("Minimum = %f; 20th percentile = %f; Maximum = %f" % (mini,quantize_distance[2],maxi))


# The code below reports on the differences between between attributes of best models found via grid versus attributes of best models found via GA search:
#

# In[ ]:



from neuronunit.optimization import model_parameters as modelp
mp = modelp.model_params
for k,v in minimagr.attrs.items():
    #hvgrid = np.linspace(np.min(mp[k]),np.max(mp[k]),10)
    dimension_length = np.max(mp[k]) - np.min(mp[k])
    solution_distance_in_1D = np.abs(float(minimaga.attrs[k]))-np.abs(float(v))
    relative_distance = dimension_length/solution_distance_in_1D
    print('the difference between brute force candidates model parameters and the GA\'s model parameters:')
    print(float(minimaga.attrs[k])-float(v),minimaga.attrs[k],v,k)
    print('the relative distance scaled by the length of the parameter dimension of interest:')
    print(relative_distance)




# In[ ]:


print('the difference between the bf error and the GA\'s error:')
print('grid search:')
from numpy import square, mean, sqrt
rmsg = sqrt(mean(square(list(minimagr.scores.values()))))
print(rmsg)
print('ga:')
rmsga = sqrt(mean(square(list(minimaga.scores.values()))))
print(rmsga)


# If any time is left over, may as well compute a more accurate grid, to better quantify GA performance in the future.

# In[ ]:


print(refined_grid)
REFINED_GRID = True
if REFINE_GRID:
    #maximagr_dtc, maxi = sorted_dtcs(dtcpopg)[-1]
    from neuronunit.optimization import exhaustive_search

    dlist = list(dview.map_sync(exhaustive_search.update_dtc_pop,refined_grid))
    from neuronunit.optimization import get_neab
    for d in dlist:
        d.model_path = get_neab.LEMS_MODEL_PATH
        d.LEMS_MODEL_PATH = get_neab.LEMS_MODEL_PATH

    # this is a big load on memory so divide it into thirds.

    dlist_first_third = dlist[0:int(len(dlist)/3)]
    dlist_second_third = dlist[int(len(dlist)/3):int(2*len(dlist)/3)]
    dlist_final_third = dlist[int(2*len(dlist)/3):-1]
    from neuronunit.optimization.exhaustive_search import dtc_to_rheo
    from neuronunit.optimization.nsga_parallel import nunit_evaluation


    def compute_half(dlist_half):
        dlist_half = list(map(dtc_to_rheo,dlist_half))
        dlist_half = dview.map_sync(nunit_evaluation,dlist_half)
        return dlist_half

    dlist_first_3rd = compute_half(dlist_first_third)

    with open('grid_dump_first_3rd.p','wb') as f:
       pickle.dump(dlist_first_3rd,f)
    # Garbage collect a big memory burden.
    dlist_first_3rd = None
    dlist_second_3rd = compute_half(dlist_second_third)

    with open('grid_dump_second_3rd.p','wb') as f:
       pickle.dump(dlist_second_3rd,f)
    # Garbage collect a big memory burden.
    dlist_second_3rd = None

    dlist_final_3rd = compute_half(dlist_final_third)
    with open('grid_dump_final_3rd.p','wb') as f:
       pickle.dump(dlist_final_3rd,f)
    # Garbage collect a big memory burden.
    dlist_final_3rd = None
