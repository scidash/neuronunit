import pickle
import copy
import os

from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
from neuronunit.optimization import get_neab
from neuronunit.optimization.optimization_management import run_ga
from neuronunit.models.NeuroML2 import model_parameters as modelp
from neuronunit.models.NeuroML2 .model_parameters import path_params
from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION

#import matplotlib as mpl#
#mpl.use('Agg')
#from matplotlib.colors import LogNorm
from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid, WSListIndividual
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


def get_tests():
    from neuronunit.optimization import get_neab
    electro_path = str(os.getcwd())+'/pipe_tests.p'
    assert os.path.isfile(electro_path) == True
    with open(electro_path,'rb') as f:
        electro_tests = pickle.load(f)
    test, observation = electro_tests[0]
    tests = copy.copy(electro_tests[0][0])
    tests_ = []
    tests_ += [tests[0]]
    tests_ += tests[4:7]
    return tests_, test, observation

tests_,test, observation = get_tests()

ax = None

#@jit
def check_line(line,gr,newrange,key):
    # Is this a concave down shape (optima in the middle)
    # Or is the most negative value on an edge?
    # if it's on the edge calculate a new parameter value to explore
    range_adj = False
    min_ = np.min(line)
    cl = [ g.dtc.attrs[key] for g in gr ]
    new = None
    index = None
    new_param_val = None
    if line[0] == min_:
        attrs = gr[0].dtc.attrs[key]
        # quantity may not be negative yet
        # don't force it to be straight away
        remin =  attrs - 10*np.abs(attrs)
        if remax == 0.0:
            remax = -1.0

        cl.insert(0,remin)
        newrange[key] = cl
        range_adj = True
        new_param_val = remin
        index = 0
    if line[-1] == min_:
        attrs = gr[-1].dtc.attrs[key]
        # quantity might not be positve yet
        # don't force it to be straight away
        remax = attrs + np.abs(attrs)*10
        if remax == 0.0:
            remax = 1.0

        cl.append(remax)
        newrange[key] = cl
        range_adj = True
        new_param_val = remax
        index = -1
    return (newrange, range_adj, new_param_val, index)

from neuronunit.models.NeuroML2 import model_parameters as modelp
from neuronunit.optimization.optimization_management import nunit_evaluation, update_deap_pop
from collections import OrderedDict

# https://stackoverflow.com/questions/33467738/numba-cell-vars-are-not-supported
# numba jit does not work on nested list iteration
#@jit
def pre_run(tests,opt_keys):
    # algorithmically find the the edges of parameter ranges, via a course grained
    # sampling of extreme parameter values
    # to find solvable instances of Izhi-model, (models with a rheobase value).
    nparams = len(opt_keys)
    from neuronunit.models.NeuroML2 import model_parameters as modelp
    mp = copy.copy(modelp.model_params)
    mp['b'] = [ -0.002, 50.0 ]
    mp['vr'] = [ -200.0, 0.0 ]

    cnt = 0
    fc = {} # final container
    
    for key in opt_keys:
        cnt = 0
        print(key,mp)
        gr = run_grid(3,tests,provided_keys = key, mp_in = mp)
        line = [ g.dtc.get_ss() for g in gr]
        nr = {key:None}
        _, range_adj, new, index = check_line(line,gr,nr,key)
        while range_adj == True:
            # while the sampled line is not concave (when minimas are at the edges)
            # sample a point to a greater extreme
            gr_ = update_deap_pop(new, tests, key)
            param_line = [ g.dtc.attrs[key] for g in gr ]
            temp = list(mp[key])
                
            if index == 0:
                gr.insert(index,gr_)
                temp.insert(index,new)
            else:
                gr.append(gr_)
                temp.append(gr_)
                
            mp[key] = np.array(temp)
            line = [ g.dtc.get_ss() for g in gr]
            _, range_adj, new,index = check_line(line,gr,mp,key)
            cnt += 1
            with open('temp.p','wb') as f:
                pickle.dump(mp,f)
        fc[key] = {}
        fc[key]['line'] = line
        fc[key]['range'] = mp
        fc[key]['cnt'] = cnt
    return fc, mp
