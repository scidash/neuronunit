from IPython.lib.deepreload import reload
import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]

import pickle
import deap
from deap import base
from deap import creator

toolbox = base.Toolbox()

def get_trans_dict(param_dict):
    trans_dict = {}
    for i,k in enumerate(list(param_dict.keys())):
        trans_dict[i]=k
    return trans_dict
import model_parameters
param_dict = model_parameters.model_params

def vm_to_ind(vm,td):
    '''
    Re instanting Virtual Model at every update vmpop
    is Noneifying its score attribute, and possibly causing a
    performance bottle neck.
    '''

    ind =[]
    for k in td.keys():
        ind.append(vm.attrs[td[k]])
    ind.append(vm.rheobase)
    return ind

class Individual(object):
     '''
     When instanced the object from this class is used as one unit of chromosome or allele by DEAP.
     Extends list via polymorphism.
     '''
     def __init__(self, *args):
         list.__init__(self, *args)
         self.error=None
         self.results=None
         self.name=''
         self.attrs = {}
         self.params=None
         self.score=None
         self.fitness=None
         self.lookup={}
         self.rheobase=None
         self.fitness = creator.FitnessMin

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)
import os
os.system('sudo /opt/conda/bin/pip install --upgrade networkx')

cd = pickle.load(open('complete_dump.p','rb'))
print(len(cd))

#unpack = [vmoffspring,history,logbook,rheobase_values,best_worst,vmhistory,hvolumes ]
vmoffspring,history,logbook,rheobase_values,best_worst,vmhistory,hvolumes = cd[0], cd[1], cd[2], cd[3], cd[4], cd[5], cd[6]
#from IPython import get_ipython
#ipython = get_ipython()
#ipython.magic("load_ext autoreload")
#ipython.magic("autoreload 2")

import net_graph
vmoffspring = net_graph.speed_up(vmoffspring)
#print(vmhistory)
#print(history)
net_graph.plotly_graph(history,vmhistory)
best = best_worst[0]
worst = best_worst[1]
net_graph.plot_log(logbook)
net_graph.plot_objectives_history(logbook)
net_graph.bar_chart(best)
net_graph.not_just_mean(logbook,hvolumes)
plot_evaluate(best,worst,names=['best','worst'])
net_graph.shadow(vmoffspring,best_worst[0])

#best, worst = net_graph.best_worst(history)
#listss = [best , worst]

#from nsga_parallel import update_vm_pop, check_rheobase

#best_worst = update_vm_pop(listss,td)
#best_worst , _ = check_rheobase(best_worst)
