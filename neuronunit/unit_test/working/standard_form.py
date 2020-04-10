#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt

import array
import random
import json

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

from neuronunit.optimisation import optimisations

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Problem definition
# Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1]
BOUND_LOW, BOUND_UP = 0.0, 1.0

# Functions zdt4 has bounds x1 = [0, 1], xn = [-5, 5], with n = 2, ..., 10
# BOUND_LOW, BOUND_UP = [0.0]  [-5.0]*9, [1.0]  [5.0]*9

# Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
NDIM = 30

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", benchmarks.zdt1)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

#def main(seed=None):

    #return pop, logbook
import dask
from dask import delayed
from distributed import Client
from sklearn.utils import parallel_backend
import sys
import pickle


NGEN = 250
MU = 100
CXPB = 0.8

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("mean", numpy.mean, axis=0)
stats.register("std", numpy.std, axis=0)
stats.register("min", numpy.min, axis=0)
stats.register("max", numpy.max, axis=0)

logbook = tools.Logbook()
logbook.header = "gen", "evals", "std", "min", "mean", "max"

pop = toolbox.population(n=MU)

# Evaluate the individuals with an invalid fitness
invalid_ind = [ind for ind in pop if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = fit
    #print(ind)
# This is just to assign the crowding distance to the individuals
# no actual selection is done
pop = toolbox.select(pop, len(pop))

record = stats.compile(pop)
logbook.record(gen=0, evals=len(invalid_ind), **record)
print(logbook.stream)

# Begin the generational process
for gen in range(1, NGEN):
    # Vary the population
    offspring = tools.selTournamentDCD(pop, len(pop))
    offspring = [toolbox.clone(ind) for ind in offspring]
    
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() <= CXPB:
            toolbox.mate(ind1, ind2)
            CXPB = 0.8
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    dtcbag = [ dask.delayed(toolbox.evaluate(d)) for d in invalid_ind ]
    fitnesses = [ dtc.compute(scheduler='distributed') for dtc in dtcbag ]
    #fitnesses = dask.compute(*dtcbag)
    #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Select the next generation population
    #pop = toolbox.select(pop,offspring, MU)
    pop = toolbox.select(pop + offspring, MU)
    #    pop = toolbox.select(pop, len(pop))
    #pop = toolbox.select(pop, len(pop))
    record = stats.compile(pop)
    logbook.record(gen=gen, evals=len(invalid_ind), **record)
    print(logbook.stream)


'''
try:
    assert 1==2
    logbook = pickle.load(open('logbook.p','rb'))
except:            
'''    
#if __name__ == "__main__":
    #from dask.distributed import Client, LocalCluster
    #cluster = LocalCluster()
    #client = Client(cluster)
    #client = Client(processes=True)
    #client = Client("process")
    #ncores = sum([c for _, c in client.ncores().items()])
    
    #print(ncores)
    #parallel_backend("dask")
    #parallel_backend("multiprocessing")

    #`random.seed()


    
import matplotlib
fig, axes = plt.subplots(figsize=(30, 30), facecolor='white')
matplotlib.rcParams.update({'font.size': 55})
gen_numbers =[ i+1 for i in range(0,len(logbook.select('gen'))) ]
import numpy as np
mean = np.array(logbook.select('mean'))
std = np.array(logbook.select('std'))
minimum = logbook.select('min')
stdminus = mean - std
stdplus = mean - std
assert len(gen_numbers) == len(stdminus) == len(stdplus)
minimum = [np.sum(i) for i in minimum]

mean = [np.sum(i) for i in mean]
stdminus = [np.sum(i) for i in stdminus]
stdplus = [np.sum(i) for i in stdplus]
axes.plot(
    gen_numbers,
    mean,
    color='black',
    linewidth=2,
    label='population average')
axes.fill_between(gen_numbers, stdminus, stdplus)
axes.plot(gen_numbers, mean, label='mean')
mean_ = np.array(logbook.select('mean'))
meanx = [i[0] for i in mean_]
#print(mean_,meanx)
axes.plot(
    gen_numbers,
    minimum,
    color='black',
    linewidth=2,
    label='population minimum')
axes.plot(gen_numbers, stdminus, label='std variation lower limit')
axes.plot(gen_numbers, stdplus, label='std variation upper limit')
#axes.set_xlim(np.min(gen_numbers) - 1, np.max(gen_numbers)  1)
axes.tick_params(labelsize=50)
axes.set_xlabel('Generations')
#axes.legend()
fig.tight_layout()
import numpy as np
import seaborn as sns
#plt.style.use('ggplot')
#fig, axes = plt.subplots(figsize=(50, 50), facecolor='white')
figname = 'benchmark'
plt.savefig(str(figname)+str('avg_converg_over_gen_.png'))#str('_')str(MU)str('_')str(NGEN)str('_')str('.png'))
plt.show()

plt.style.use('ggplot')
#fig, axes = plt.subplots(figsize=(50, 50), facecolor='white')
fig, axes = plt.subplots(figsize=(30, 30), facecolor='white')
matplotlib.rcParams.update({'font.size': 12})
#logbook = ga_out['log']

plt.clf()
plt.figure()
sns.set_style("darkgrid")

avg, max_, min_, std_ = logbook.select("mean", "max", "min","std")
all_over_gen = {}
pf_loc = 0

fitevol = [ m['min'] for m in logbook ]

get_min = [(np.sum(j),i) for i,j in enumerate(fitevol)]
min_x = sorted(get_min,key = lambda x: x[0])[0][1]+1
plt.clf()
fig2, ax2 = plt.subplots(2,1)

for i,f in enumerate(range(0,2)):
    ax2[i].plot(gen_numbers,[j[i] for j in fitevol ])#,label=("NeuronUnit Test: {0}".format(str(k)str(' ')str(v)), fontsize = 35.0)
plt.savefig(str(figname)+str('components_over_gen_.png'))#str('_')str(MU)str('_')str(NGEN)str('_')str('.png'))

plt.show()
