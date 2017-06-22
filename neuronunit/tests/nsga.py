import matplotlib as mpl
mpl.use('agg',warn=False)
from matplotlib import pyplot as plt
import time
import pdb
import array
import random
import sys
sys.path.insert(0,"../../../sciunit")

import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from deap import algorithms
from deap import base
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

import scoop

try:
    import get_neab
except ImportError:
    from neuronunit.tests import get_neab

import quantities as qt
import os
import os.path
from scoop import utils
from scoop import futures

import sciunit.scores as scores
history = tools.History()



init_start=time.time()
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0,
                                                    -1.0, -1.0, -1.0, -1.0))
creator.create("Individual",list, fitness=creator.FitnessMin)

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
        self.s_html=None
        self.lookup={}
        self.rheobase=None
toolbox = base.Toolbox()
try:
    import model_parameters
except ImportError:
    from neuronunit.tests import model_parameters as params

vr = np.linspace(-75.0,-50.0,1000)
a = np.linspace(0.015,0.045,1000)
b = np.linspace(-3.5*10E-9,-0.5*10E-9,1000)
k = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,1000)
C = np.linspace(1.00000005E-4-1.00000005E-5,1.00000005E-4+1.00000005E-5,1000)

c = np.linspace(-55,-60,1000)
d = np.linspace(0.050,0.2,1000)
v0 = np.linspace(-75.0,-45.0,1000)
vt =  np.linspace(-50.0,-30.0,1000)
vpeak= np.linspace(20.0,30.0,1000)

param=['vr','a','b','C','c','d','v0','k','vt','vpeak']
rov=[]
rov.append(vr)
rov.append(a)
rov.append(b)
rov.append(C)
rov.append(c)

rov.append(d)
rov.append(v0)
rov.append(k)
rov.append(vt)
rov.append(vpeak)

BOUND_LOW=[ np.min(i) for i in rov ]
BOUND_UP=[ np.max(i) for i in rov ]

NDIM = len(param)

import functools

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("Individual", tools.initIterate, creator.Individual, toolbox.attr_float)
import deap as deap

toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
toolbox.register("select", tools.selNSGA2)

import utilities as outils
model = outils.model


def hypervolume_contrib(front, **kargs):
    """Returns the hypervolume contribution of each individual. The provided
    *front* should be a set of non-dominated individuals having each a
    :attr:`fitness` attribute.
    """
    import numpy
    # Must use wvalues * -1 since hypervolume use implicit minimization
    # And minimization in deap use max on -obj
    wobj = numpy.array([ind.fitness.wvalues for ind in front]) * -1
    ref = kargs.get("ref", None)
    if ref is None:
        ref = numpy.max(wobj, axis=0) + 1

    total_hv = hv.hypervolume(wobj, ref)

    def contribution(i):
        # The contribution of point p_i in point set P
        # is the hypervolume of P without p_i
        return total_hv - hv.hypervolume(numpy.concatenate((wobj[:i], wobj[i+1:])), ref)

    # Parallelization note: Cannot pickle local function
    return map(contribution, range(len(front)))

import quantities as pq
import neuronunit.capabilities as cap
import matplotlib.pyplot as plt

def evaluate(individual,tuple_params):#This method must be pickle-able for scoop to work.
    '''
    Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    value, that was found in a previous rheobase search.

    outputs: a tuple that is a compound error function that NSGA can act on.

    Assumes rheobase for each individual virtual model object (vms) has already been found
    there should be a check for vms.rheobase, and if not then error.
    Inputs a gene and a virtual model object.
    outputs are error components.
    '''
    gen,vms,rheobase=tuple_params
    assert vms.rheobase==rheobase
    params = utitilities.params
    model = utitilities.model

    uc = {'amplitude':rheobase}
    current = params.copy()['injected_square_current']
    current.update(uc)
    current = {'injected_square_current':current}
    #Its very important to reset the model here. Such that its vm is new, and does not carry charge from the last simulation
    #model.load_model()

    model.re_init(vms.attrs)#purge models stored charge. by reinitializing it
    model.inject_square_current(current)

    #model.update_run_params(vms.attrs)

    n_spikes = model.get_spike_count()
    assert n_spikes == 1 or n_spikes == 0  # Its possible that no rheobase was found

    inderr = getattr(individual, "error", None)
    if inderr!=None:
        if len(individual.error)!=0:
            #the average of 10 and the previous score is chosen as a nominally high distance from zero
            error = [ (abs(-10.0+i)/2.0) for i in individual.error ]
    else:
        #100 is chosen as a nominally high distance from zero
        error = [ 100.0 for i in range(0,8) ]

    sane = get_neab.suite.tests[0].sanity_check(vms.rheobase*pq.pA,model)
    if sane == True and n_spikes == 1:
        for i in [4,5,6]:
            get_neab.suite.tests[i].params['injected_square_current']['amplitude'] = vms.rheobase*pq.pA
        get_neab.suite.tests[0].prediction={}
        if vms.rheobase == 0:
            vms.rheobase = 1E-10
        assert vms.rheobase != None
        score = get_neab.suite.tests[0].prediction['value']=vms.rheobase*pq.pA
        score = get_neab.suite.judge(model)#passing in model, changes the model


        spikes_numbers=[]
        model.run_number+=1
        error = score.sort_key.values.tolist()[0]
        for x,y in enumerate(error):
            if y != None and x == 0 :
                error[x] = abs(vms.rheobase - get_neab.suite.tests[0].observation['value'].item())
            elif y != None and x!=0:
                error[x]= abs(y-0.0)+error[0]
            elif y == None:
                inderr = getattr(individual, "error", None)
                if inderr!=None:
                    error[x]= abs(inderr[x]-10)/2.0 + error[0]
                else:
                    error[x] = 10.0 + error[0]

    individual.error = error
    import copy
    individual.rheobase = copy.copy(vms.rheobase)
    return error[0],error[1],error[2],error[3],error[4],error[5],error[6],error[7],




toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)

toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

toolbox.register("select", tools.selNSGA2)
toolbox.register("map", futures.map)


def plotss(vmlist,gen):
    import matplotlib.pyplot as plt
    plt.clf()
    for ind,j in enumerate(vmlist):
        if hasattr(ind,'results'):
            plt.plot(ind.results['t'],ind.results['vm'],label=str(vmlist[j].attr) )
            #plt.xlabel(str(vmlist[j].attr))
    plt.savefig('snap_shot_at_gen_'+str(gen)+'.png')
    plt.clf()


def individual_to_vm(ind,vmpop=None):

    import copy
    if vmpop==None:
        for i, p in enumerate(param):
            value = str(ind[i])
            if i == 0:
                attrs={'//izhikevich2007Cell':{p:value }}
            else:
                attrs['//izhikevich2007Cell'][p] = value
        vm = outils.VirtualModel()
        vm.attrs = attrs

        inderr = getattr(ind, "error", None)
        if not inderr==None:
            vm.error = copy.copy(ind.error)
            assert not vm.error == None

        indrheobase = getattr(ind, "rheobase", None)
        if not indrheobase==None:
            vm.rheobase =  copy.copy(ind.rheobase)
            assert not vm.rheobase == None


    return vm

def replace_rh(pop,rh_value,vmpop):
    rheobase_checking=outils.evaluate
    from itertools import repeat
    import copy
    assert len(pop) == len(vmpop)


    for i,ind in enumerate(pop):

         #sense if Rheobase is less than or equal to zero.
         #discard models that cause such results,
         # and create new models by mutating these parameters.

         while type(vmpop[i].rheobase) is type(None):
              print('this loop appropriately exits none mutate away from ')

              toolbox.mutate(ind)
              toolbox.mutate(ind)
              toolbox.mutate(ind)
              print('tryingmutations', ind)
              temp = individual_to_vm(ind)
              pop_rh = rheobase_checking(temp)
              print('trying ', pop_rh.rheobase)
              if type(pop_rh.rheobase) is not type(None):
                  if not (float(pop_rh.rheobase) <=0.0) :

                      #make sure that the old individual, and virtual model object are
                      #over written so do not use list append pattern, as this will not
                      #over write the objects in place, but instead just grow the lists inappropriately
                      #also some of the lines below may be superflous in terms of machine instructions, but
                      #they function to make the code more explicit and human readable.
                      #ind = pop_rh
                      ind.rheobase = pop_rh.rheobase

                      pop[i] = copy.copy(ind)
                      vmpop[i] = copy.copy(pop_rh)
                      print(pop_rh.rheobase, 'rheobase value is updating')
                      assert ind.rheobase == pop_rh.rheobase == vmpop[i].rheobase
                      assert vmpop[i].rheobase is not type(None)
         assert vmpop[i].rheobase is not type(None)

         while float(vmpop[i].rheobase) <=0.0 :
             print('this loop appropriately exits less than or equal to zero rheobase mutate away from')

             toolbox.mutate(ind)
             toolbox.mutate(ind)
             toolbox.mutate(ind)
             print('tryingmutations', ind)
             temp = individual_to_vm(ind)
             pop_rh = rheobase_checking(temp)
             print('trying ', pop_rh.rheobase)
             if type(pop_rh.rheobase) is not type(None):
                 if not (float(pop_rh.rheobase) <=0.0) :
                      #make sure that the old individual, and virtual model object are
                      #over written so do not use list append pattern, as this will not
                      #over write the objects in place, but instead just grow the lists inappropriately
                      #also some of the lines below may be superflous in terms of machine instructions, but
                      #they function to make the code more explicit and human readable.
                      #ind = pop_rh
                     ind.rheobase = pop_rh.rheobase
                     pop[i] = ind
                     vmpop[i] = pop_rh
                     print(pop_rh.rheobase, 'rheobase value is updating')
                     assert ind.rheobase == pop_rh.rheobase == vmpop[i].rheobase



    assert len(pop)!=0
    assert len(vmpop)==len(pop)
    return pop, vmpop




def test_to_model(vms,local_test_methods):
    import get_neab
    tests = get_neab.suite.tests
    import matplotlib.pyplot as plt
    import copy
    global model
    global nsga_matrix
    model.local_run()
    #model.update_run_params(vms.attrs)
    model.re_init(vms.attrs)
    tests = None
    tests = get_neab.suite.tests
    tests[0].prediction={}
    tests[0].prediction['value']=vms.rheobase*qt.pA
    tests[0].params['injected_square_current']['amplitude']=vms.rheobase*qt.pA
    #score = get_neab.suite.judge(model)#pass
    if local_test_methods in [4,5,6]:
        tests[local_test_methods].params['injected_square_current']['amplitude']=vms.rheobase*qt.pA
    #model.results['vm'] = [ 0 ]
    model.re_init(vms.attrs)
    tests[local_test_methods].generate_prediction(model)
    injection_trace = np.zeros(len(model.results['t']))

    trace_size = int(len(model.results['t']))
    injection_trace = np.zeros(trace_size)

    end = len(model.results['t'])#/delta
    delay = int((float(get_neab.suite.tests[0].params['injected_square_current']['delay'])/1600.0 ) * end )
    #delay = get_neab.suite.tests[0].params['injected_square_current']['delay']['value']/delta
    duration = int((float(1100.0)/1600.0) * end ) # delta
    injection_trace[0:int(delay)] = 0.0
    injection_trace[int(delay):int(duration)] = vms.rheobase
    injection_trace[int(duration):int(end)] = 0.0
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(model.results['t'],model.results['vm'],label='$V_{m}$ (mV)')
    axarr[0].set_xlabel(r'$V_{m} (mV)$')
    axarr[0].set_xlabel(r'$time (ms)$')

    axarr[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    axarr[1].plot(model.results['t'],injection_trace,label='$I_{i}$(pA)')
    if vms.rheobase > 0:
        axarr[1].set_ylim(0, 2*vms.rheobase)
    if vms.rheobase < 0:
        axarr[1].set_ylim(2*vms.rheobase,0)
    axarr[1].set_xlabel(r'$current injection (pA)$')
    axarr[1].set_xlabel(r'$time (ms)$')

    axarr[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    model.re_init(vms.attrs)
    tests = None
    tests = get_neab.suite.tests
    #tests[0].prediction={}
    tests[0].prediction={}
    tests[0].prediction['value']=vms.attrs*qt.pA
    tests[0].params['injected_square_current']['amplitude']=vms.rheobase*qt.pA


    if local_test_methods in [4,5,6]:
        tests[local_test_methods].params['injected_square_current']['amplitude']=vms.rheobase*qt.pA

    #model.results['vm'] = [ 0 ]
    model.re_init(vms.attrs)
    #tests[local_test_methods].judge(model)
    tests[local_test_methods].generate_prediction(model)
    injection_trace = np.zeros(len(model.results['t']))
    delta = model.results['t'][1]-model.results['t'][0]

    trace_size = int(len(model.results['t']))
    injection_trace = np.zeros(trace_size)

    end = len(model.results['t'])#/delta
    delay = int((float(get_neab.suite.tests[0].params['injected_square_current']['delay'])/1600.0 ) * end )
    #delay = get_neab.suite.tests[0].params['injected_square_current']['delay']['value']/delta
    duration = int((float(1100.0)/1600.0) * end ) # delta
    injection_trace[0:int(delay)] = 0.0
    injection_trace[int(delay):int(duration)] = vms.rheobase
    injection_trace[int(duration):int(end)] = 0.0
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(model.results['t'],model.results['vm'],label='$V_{m}$ (mV)')
    axarr[0].set_xlabel(r'$V_{m} (mV)$')
    axarr[0].set_xlabel(r'$time (ms)$')

    axarr[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    axarr[1].plot(model.results['t'],injection_trace,label='$I_{i}$(pA)')
    if vms.rheobase > 0:
        axarr[1].set_ylim(0, 2*vms.rheobase)
    if vms.rheobase < 0:
        axarr[1].set_ylim(2*vms.rheobase,0)
    axarr[1].set_xlabel(r'$current injection (pA)$')
    axarr[1].set_xlabel(r'$time (ms)$')

    axarr[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    plt.title(str(tests[local_test_methods]))
    plt.savefig(str('best_solution')+str('.png'))
    #plt.clf()
    model.results['vm']=None
    model.results['t']=None
    tests[local_test_methods].related_data=None
    local_test_methods=None
    return 0

def updatevmpop(pop,rh_value=None):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
    from itertools import repeat
    import numpy as np
    import copy
    #global MU
    #pop=copy.copy(pop)
    assert len(pop)!=0
    rheobase_checking=outils.evaluate
    vmpop = list(futures.map(individual_to_vm,[toolbox.clone(i) for i in pop] ))

    assert len(pop)!=0
    if len(pop)!=0 and len(vmpop)!= 0:
        if rh_value == None:
            rh_value = np.mean(np.array([i.rheobase for i in vmpop]))
            assert type(rh_value) is not type(None)
        rh_value_array = [rh_value for i in vmpop ]
        vmpop = list(futures.map(rheobase_checking,vmpop,repeat(rh_value)))

        pop,vmpop = replace_rh(pop,rh_value,vmpop)
        assert len(pop)!=0
    return pop,vmpop




def main():
    fbest =[]
    global NGEN
    NGEN=6
    global MU
    import numpy as np
    numpy = np
    MU=12#Mu must be some multiple of 8, such that it can be split into even numbers over 8 CPUs

    best = [ 0 for i in range(0,NGEN) ]

    CXPB = 0.9
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    pf = tools.ParetoFront()


    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)
    pop = [toolbox.clone(i) for i in pop]


    #create the first population of individuals
    #guess_attrs=[]

    #find rheobase on a model constructed out of the mean parameter values.
    #for x,y in enumerate(param):
    #    guess_attrs.append(np.mean( [ i[x] for i in pop ]))

    #from itertools import repeat
    '''
    mean_vm = outils.VirtualModel()
    #def assign_param(param):
    for i, p in enumerate(param):
        value=str(guess_attrs[i])
        model.name=str(model.name)+' '+str(p)+str(value)
        if i==0:
            attrs={'//izhikevich2007Cell':{p:value }}
        else:
            attrs['//izhikevich2007Cell'][p]=value
    mean_vm.attrs=attrs
    import copy

    steps = np.linspace(50,150,7.0)
    steps_current = [ i*pq.pA for i in steps ]
    rh_param=(False,steps_current)
    searcher=outils.searcher
    check_current=outils.check_current
    pre_rh_value=searcher(check_current,rh_param)
    rh_value=pre_rh_value.rheobase
    global vmpop
    if rh_value == None:
        rh_value = 0.0
    '''

    #Now attempt to get the rheobase values by first trying the mean rheobase value.
    #This is not an exhaustive search that results in found all rheobase values
    #It is just a trying out an educated guess on each individual in the whole population as a first pass.
    #invalid_ind = [ ind for ind in pop if not ind.fitness.valid ]

    rheobase_checking=outils.evaluate
    pop,vmpop = updatevmpop(pop)
    history.update(pop)
    pf.update(pop)

    print(pf)

    assert len(pop)==len(vmpop)
    rhstorage = [i.rheobase for i in vmpop]
    from itertools import repeat
    tuple_storage = zip(repeat(0),vmpop,rhstorage)


    #Now get the fitness of genes:
    #Note the evaluate function called is different
    fitnesses = list(toolbox.map(toolbox.evaluate, pop, tuple_storage))

    invalid_ind = [ ind for ind in pop if not ind.fitness.valid ]

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    #purge individuals for which rheobase was not found
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = tools.selNSGA2(pop, MU)

    assert len(pop)!=0
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    # Begin the generational process
    rhstorage2 = 0.0
    for gen in range(1, NGEN):
        # Vary the population
        for i in vmpop:
            if type(i.rheobase) is not type(None):
                #rhstorage.append(i.rheobase)
                rhstorage2 += i.rheobase
        #rhstorage = [i.rheobase for i in vmpop]
        rhmean = rhstorage2/len(vmpop)
        pop,vmpop = updatevmpop(pop,rhmean)
        assert len(pop)!=0
        assert len(vmpop)!=0


        invalid_ind = [ ind for ind in pop if ind.fitness.valid ]
        assert len(invalid_ind)!=0
        #offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = tools.selNSGA2(pop, len(pop))
        assert len(offspring)!=0

        offspring = [toolbox.clone(ind) for ind in offspring]
        assert len(offspring)!=0
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        vmpop = list(futures.map(individual_to_vm,[toolbox.clone(i) for i in offspring] ))
        vmpop = list(futures.map(rheobase_checking,vmpop,repeat(rh_value)))
        rhstorage = [ i.rheobase for i in vmpop ]
        tuple_storage = zip(repeat(gen),vmpop,rhstorage)
        assert len(vmpop)==len(pop)
        fitnesses = list(toolbox.map(toolbox.evaluate, offspring , tuple_storage))

        attr_dict = [p.attrs for p in vmpop ]
        #attr_keys = [ i.keys() for d in attr_dict for i in d.values() ][0]
        attr_list = [ i.values() for d in attr_dict for i in d.values() ][0]
        #std_dev=np.std(attr_list)
        #print('the standard deviation is',std_dev)

        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        size_delta = MU-len(invalid_ind)

        pop = toolbox.select(offspring, MU)
        print('the pareto front is',pf)

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

        #logbook.header = "gen", "evals", "std", "min", "avg", "max"
        #x = list(range(0, NGEN))
        #x = list(range(0, strategy.lambda_ * NGEN, strategy.lambda_))
        fbest.append( pf[0].fitness.values )
        best[gen].append(pf[0])


    pop.sort(key=lambda x: x.fitness.values)
    #logbook select is like a database selection, as opposed to logbook record which initializes and populates the data
    evals, gen ,std, avg, max_, min_ = logbook.select("evals","gen","avg", "max", "min", "std")
    x = list(range(0,len(avg)))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.semilogy(x, avg, "--b")
    plt.semilogy(x, max_, "--b")
    plt.semilogy(x, min_, "-b")
    plt.semilogy(x, fbest, "-c")
    #plt.semilogy(x, sigma, "-g")
    #plt.semilogy(x, axis_ratio, "-r")
    plt.grid(True)
    plt.title("blue: f-values, green: sigma, red: axis ratio")

    plt.subplot(2, 2, 2)
    plt.plot(x, best)
    plt.grid(True)
    plt.title("Object Variables")

    plt.subplot(2, 2, 3)
    plt.semilogy(x, std)
    plt.grid(True)
    plt.title("Standard Deviations in All Coordinates")
    plt.savefig('GA_stats_vs_generation.png')
    f=open('worst_candidate.txt','w')
    if len(vmpop)!=0:
        f.write(str(vmpop[-1].attrs))
        f.write(str(vmpop[-1].rheobase))

    f.write(logbook.stream)
    f.close()
    score_matrixt=[]
    if len(vmpop)!=0:

        score_matrixt.append((vmpop[0].error,vmpop[0].attrs,vmpop[0].rheobase))
        score_matrixt.append((vmpop[1].error,vmpop[1].attrs,vmpop[1].rheobase))

        score_matrixt.append((vmpop[-1].error,vmpop[-1].attrs,vmpop[-1].rheobase))
    import pickle
    import pickle
    with open('score_matrixt.pickle', 'wb') as handle:
        pickle.dump(score_matrixt, handle)

    with open('vmpop.pickle', 'wb') as handle:
        pickle.dump(vmpop, handle)




    return vmpop, pop, stats, invalid_ind




if __name__ == "__main__":

    import time
    start_time=time.time()
    whole_initialisation = start_time-init_start
    model=outils.model
    vmpop, pop, stats, invalid_ind = main()



    try:
        ground_error = pickle.load(open('big_model_evaulated.pickle','rb'))
    except:
        '{} it seems the error truth data does not yet exist, lets create it now '.format(str(False))
        ground_error = list(futures.map(util.func2map, ground_truth))
        pickle.dump(ground_error,open('big_model_evaulated.pickle','wb'))

    _ = plot_results(ground_error)


    from sklearn.decomposition import PCA as sklearnPCA
    from sklearn.preprocessing import StandardScaler

    attr_dict = [p.attrs for p in vmpop ]
    attr_keys = [ i.keys() for d in attr_dict for i in d.values() ][0]
    print(attr_keys)
    X =  [ ind for ind in pop ]
    X_std = StandardScaler().fit_transform(X)
    sklearn_pca = sklearnPCA(n_components=3)

    Y_sklearn = sklearn_pca.fit_transform(X_std)



    #print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    print(stats)
    #print("Convergence: ", convergence(pop, optimal_front))
    #print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))
    import numpy
    #
    front = numpy.array([ind.fitness.values for ind in pop])
    #optimal_front = numpy.array(optimal_front)
    #plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")

    import networkx
    import pickle
    #import networkx
    with open('pca_transform.pickle', 'wb') as handle:
        pickle.dump(Y_sklearn, handle)

    graph = networkx.DiGraph(history.genealogy_tree)
    graph = graph.reverse()     # Make the grah top-down

    assert len(vmpop)==len(pop)
    print(len(graph),'length of graph')
    for i in graph:
        print(history.genealogy_history[i],'i in graph')
    gpop = [ history.genealogy_history[i] for i in graph ]
    print(gpop)
    for j in gpop:
        print(j,'jpop')
    vmpop = list(futures.map(individual_to_vm,[toolbox.clone(i) for i in gpop] ))

    #colors = list([ i.errors for i in gpop ])
    pgop,vmpop = updatevmpop(vmpop)
    rhstorage = [i.rheobase for i in vmpop]
    iter_ = zip(repeat(NGEN),vmpop,rhstorage)

    colors = list(toolbox.map(toolbox.evaluate, gpop , iter_))

    #colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    networkx.draw(graph, node_color=colors)
    plt.savefig('genealogy_history.png')

    print(len(Y_sklearn))
    traces = []
    import plotly.plotly as py
    import plotly.graph_objs as go
    for name in ['componen1', 'component2', 'component3']:

        trace = go.Scatter(
            x=Y_sklearn[y==name,0],
            y=Y_sklearn[y==name,1],
            mode='markers',
            name=name,
            marker=Marker(
                size=12,
                line=Line(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5),
                opacity=0.8))
        traces.append(trace)

    attr_keys[Y_sklearn]
    data = Data(traces)
    layout = Layout(xaxis=XAxis(title='PC1', showline=False),
                    yaxis=YAxis(title='PC2', showline=False))
    fig = go.Figure(data=data, layout=layout)

    py.image.save_as(fig, filename='principle_components.png')




    steps = np.linspace(50,150,7.0)
    steps_current = [ i*pq.pA for i in steps ]
    rh_param = (False,steps_current)
    searcher = outils.searcher
    check_current = outils.check_current
    print(vmpop)

    score_matrixt=[]
    vmpop = list(map(individual_to_vm,pop))



'''
trace_size = int(len(model.results['t']))

injection_trace = np.zeros(trace_size)

end = len(model.results['t'])#/delta
delay = int((float(get_neab.suite.tests[0].params['injected_square_current']['delay'])/1600.0 ) * end )
#delay = get_neab.suite.tests[0].params['injected_square_current']['delay']['value']/delta
duration = int(float(1100.0/1600.0) * end ) # delta
#print(len(delay),len(duration),len(end),len(model.results['t']),' len(delay),len(duration),len(end),len(model.results["t"]) ' )
injection_trace[0:int(delay)] = 0.0
injection_trace[int(delay):int(duration)] = rheobase
injection_trace[int(duration):int(end)] = 0.0
'''

#plt.plot(model.results['t'],injection_trace,label='$I_{i}$(pA)')
#if vms.rheobase > 0:
#        axarr[1].set_ylim(0, 2*rheobase)
#if vms.rheobase < 0:
#    axarr[1].set_ylim(2*rheobase,0)
#axarr[1].set_xlabel(r'$current injection (pA)$')
#axarr[1].set_xlabel(r'$time (ms)$')
#pdb.set_trace()
#print(get_neab.suite.tests[i].params['injected_square_current'].keys())


#plt.clf()
