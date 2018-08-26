import pickle
import copy
import os

from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
from neuronunit.optimization import get_neab
from neuronunit.optimization.optimization_management import run_ga
from neuronunit.models.NeuroML2 import model_parameters as modelp
from neuronunit.models.NeuroML2 .model_parameters import path_params

from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION

import matplotlib as mpl
mpl.use('Agg')
from matplotlib.colors import LogNorm
from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid
import matplotlib.pyplot as plt

from neuronunit import plottools
plot_surface = plottools.plot_surface
scatter_surface = plottools.plot_surface

electro_path = str(os.getcwd())+'/pipe_tests.p'
print(os.getcwd())
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)

from neuronunit.optimization import exhaustive_search as es
import quantities as pq

from itertools import product
import matplotlib.pyplot as plt

electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
test, observation = electro_tests[0]

tests = copy.copy(electro_tests[0][0])
tests_ = []
tests_ += [tests[0]]
tests_ += tests[4:7]

ax = None






def check_line(line,gr,newrange):
    range_adj = False
    key = list(newrange.keys())[0]
    min_ = np.min(line)
    cl = [ g.dtc.attrs[key] for g in gr ]
    
    if line[0] == min_:
        attrs = gr[0].dtc.attrs[key]
        remin = - 2*np.abs(attrs)*2
        #same_max = gr[-1].dtc.attrs[key]
        cl.insert(0,remin)
        newrange[key] = cl
        range_adj = True
        new = remin 
    if line[-1] == min_:
        attrs = gr[-1].dtc.attrs[key]
        #same_min = gr[0].dtc.attrs[key]
        remax = np.abs(attrs)*2
        cl.append(remax)
        newrange[key] = cl
        range_adj = True
        new = remax
    return (newrange, range_adj, new)

def mp_process(newrange):
    from neuronunit.models.NeuroML2 import model_parameters as modelp
    mp = copy.copy(modelp.model_params)
    for k,v in newrange.items():
        if type(v) is not type(None):
            mp[k] = v
    return mp

from neuronunit.models.NeuroML2 import model_parameters as modelp

def pre_run(tests,opt_keys):
    nparams = len(opt_keys)
    from neuronunit.models.NeuroML2 import model_parameters as modelp
    mp = copy.copy(modelp.model_params)
    package = run_ga(mp,12,12,tests_,provided_keys = opt_keys)
    pop = package[0]
    history = package[4]
    gen_vs_pop =  package[6]
    hof = package[1]

    dim = len(hof[0].dtc.attrs.keys())
    flat_iter = [ (i,ki,j,kj) for i,ki in enumerate(hof[0].dtc.attrs.keys()) for j,kj in enumerate(hof[0].dtc.attrs.keys()) ]
    cnt = 0
    fc = {} # final container
    for i,ki,j,kj in flat_iter:
        free_param = set([ki,kj]) # construct a small-set out of the indexed keys 2. If both keys are
        # are the same, this set will only contain one index
        bs = set(hof[0].dtc.attrs.keys()) # construct a full set out of all of the keys available, including ones not indexed here.
        diff = bs.difference(free_param) # diff is simply the key that is not indexed.
        # BD is the dictionary of parameters to be held constant
        # if the plot is 1D then two parameters should be held constant.
        hc =  {}
        for d in diff:
            hc[d] = hof[0].dtc.attrs[d]

        mp = copy.copy(modelp.model_params)
        if i == j:
            assert len(free_param) == len(hc) - 1
            assert len(hc) == len(free_param) + 1
            # from neuronunit.models.NeuroML2 import model_parameters as modelp
            gr = run_grid(3,tests,provided_keys = free_param ,hold_constant = hc, mp_in = mp)
            # make a psuedo test, that still depends on input Parametersself.
            # each test evaluates a normal PDP.
            line = [ np.sum(list(g.dtc.scores.values())) for g in gr]
            nr = {str(list(free_param)[0]):None}
            newrange, range_adj, new = check_line(line,gr,nr)
            cnt = 0
            while range_adj == True:

                mp = mp_process(newrange)
                gr_ = run_grid(1,tests,provided_keys = new,hold_constant = hc, mp_in = mp)

                if new < 0.0:
                    gr.insert(0,gr_)
                else:
                    gr.append(gr_)                
                # make a psuedo test, that still depends on input Parametersself.
                # each test evaluates a normal PDP.
                nr = {}
                line = [ np.sum(list(g.dtc.scores.values())) for g in gr]
                newrange, range_adj, new = check_line(line,gr,newrange)
                mp = mp_process(newrange)
                cnt+=1

            fc[ki] = {}    
            fc[ki]['line'] = line    
            fc[ki]['range'] = newrange
            fc[ki]['cnt'] = cnt
    return fc, mp

# fc, mp = pre_run(tests=tests_)


