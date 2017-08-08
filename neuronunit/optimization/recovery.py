import pickle
import deap
from deap import base
from deap import creator

toolbox = base.Toolbox()

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
vmpop = cd[0]
history = cd[1]
logbook = cd[2]
import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]
import net_graph
best, worst = net_graph.best_worst(history)
listss = [best , worst]
best_worst = update_vm_pop(listss,td)
best_worst , _ = check_rheobase(best_worst)
net_graph.speed_up(vmpop)
net_graph.shadow(vmoffspring,best_worst[0])
net_graph.plotly_graph(history,vmhistory)
