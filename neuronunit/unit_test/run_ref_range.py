import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# verify that backend is appropriate before compute job:
plt.clf()

import copy
import os
import pandas as pd

import pickle
from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid
from neuronunit.optimization import model_parameters as modelp
from neuronunit.optimization import exhaustive_search as es
from neuronunit.models.NeuroML2 import model_parameters as modelp
from neuronunit.models.NeuroML2 .model_parameters import path_params

from neuronunit.optimization.optimization_management import run_ga
from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
import matplotlib.pyplot as plt
from neuronunit.models.reduced import ReducedModel
from itertools import product

import quantities as pq
from numba import jit
from neuronunit import plottools
ax = None

#plot_surface = plottools.plot_surface
#scatter_surface = plottools.plot_surface


from neuronunit.optimization import get_neab

def get_tests():
    # get neuronunit tests
    # and select out the tests that are more about waveform shape
    # and less about electrophysiology of the membrane.
    # We are more interested in phenomonogical properties.
    electro_path = str(os.getcwd())+'/pipe_tests.p'
    assert os.path.isfile(electro_path) == True
    with open(electro_path,'rb') as f:
        electro_tests = pickle.load(f)

    electro_tests = get_neab.replace_zero_std(electro_tests)
    electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
    test, observation = electro_tests[0]
    tests = copy.copy(electro_tests[0][0])
    tests_ = []
    tests_ += [tests[0]]
    tests_ += tests[4:7]
    return tests_, test, observation

tests_,test, observation = get_tests()

grid_results = {}

def plot_scatter(hof,ax,keys):
    z = np.array([ np.sum(list(p.dtc.scores.values())) for p in hof ])
    x = np.array([ p.dtc.attrs[str(keys[0])] for p in hof ])
    if len(keys) != 1:
        y = np.array([ p.dtc.attrs[str(keys[1])] for p in hof ])
        ax.cla()
        ax.set_title(' {0} vs {1} '.format(keys[0],keys[1]))
        ax.scatter(x, y, c=y, s=125)#, cmap='gray')
    return ax

def plot_surface(gr,ax,keys,imshow=False):
    # from https://github.com/russelljjarvis/neuronunit/blob/dev/neuronunit/unit_test/progress_report_4thJuly.ipynb
    # Not rendered https://github.com/russelljjarvis/neuronunit/blob/dev/neuronunit/unit_test/progress_report_.ipynb
    gr = [ g for g in gr if type(g.dtc) is not type(None) ]
    gr = [ g for g in gr if type(g.dtc.scores) is not type(None) ]
    ax.cla()
    gr_ = []
    index = 0
    for i,g in enumerate(gr):
       if type(g.dtc) is not type(None):
           gr_.append(g)
       else:
           index = i

    xx = np.array([ p.dtc.attrs[str(keys[0])] for p in gr ])
    yy = np.array([ p.dtc.attrs[str(keys[1])] for p in gr ])
    zz = np.array([ np.sum(list(p.dtc.scores.values())) for p in gr ])
    dim = len(xx)
    N = int(np.sqrt(len(xx)))
    X = xx.reshape((N, N))
    Y = yy.reshape((N, N))
    Z = zz.reshape((N, N))
    if imshow==False:
        ax.pcolormesh(X, Y, Z, edgecolors='black')
    else:
        import seaborn as sns; sns.set()
        ax = sns.heatmap(Z)

    ax.set_title(' {0} vs {1} '.format(keys[0],keys[1]))
    return ax

def plot_line(gr,ax,key):
    ax.cla()
    ax.set_title(' {0} vs  score'.format(key[0]))
    z = np.array([ np.sum(list(p.dtc.scores.values())) for p in gr ])
    x = np.array([ p.dtc.attrs[key[0]] for p in gr ])

    ax.plot(x,z)
    ax.set_xlim(np.min(x),np.max(x))
    ax.set_ylim(np.min(z),np.max(z))
    return ax



def mock_grids(params):
    with open('package.p','rb') as f:
        package = pickle.load(f)
    hof = package[1]
    with open('ranges.p','rb') as f:
        _,params = pickle.load(f)
    dim = len(hof[0].dtc.attrs.keys())
    flat_iter = iter([(i,ki,j,kj) for i,ki in enumerate(hof[0].dtc.attrs.keys()) for j,kj in enumerate(hof[0].dtc.attrs.keys())])
    matrix = [[0 for x in range(dim)] for y in range(dim)]
    plt.clf()

    fig,ax = plt.subplots(dim,dim,figsize=(10,10))
    cnt = 0
    mat = np.zeros((dim,dim))
    df = pd.DataFrame(mat)

    for i,freei,j,freej in flat_iter:
        free_param = set([freei,freej]) # construct a small-set out of the indexed keys 2. If both keys are
        # are the same, this set will only contain one index
        bs = set(hof[0].dtc.attrs.keys()) # construct a full set out of all of the keys available, including ones not indexed here.
        diff = bs.difference(free_param) # diff is simply the key that is not indexed.
        # hc is the dictionary of parameters to be held constant
        # if the plot is 1D then two parameters should be held constant.
        hc =  {}
        for d in diff:
            hc[d] = hof[0].dtc.attrs[d]

        cpparams = {}
        if i == j:
            assert len(free_param) == len(hc) - 1
            assert len(hc) == len(free_param) + 1
            # zoom in on optima
            cpparams['freei'] = (np.min(params[freei]), np.max(params[freei]))
            
            ##
            # gr = run_grid(10,tests,provided_keys = freei, hold_constant = hc,mp_in = params)
            ##
            gr = np.random(10)
            #W = np.random.random((dim, dim))
            # make a psuedo test, that still depends on input Parametersself.
            # each test evaluates a normal PDP.
            fp = list(copy.copy(free_param))
            ax[i,j] = plot_line(gr,ax[i,j],fp)
        if i >j:
            assert len(free_param) == len(hc) + 1
            assert len(hc) == len(free_param) - 1
            cpparams['freei'] = (np.min(params[freei]), np.max(params[freei]))
            cpparams['freej'] = (np.min(params[freej]), np.max(params[freej]))
            gr = np.random(100)

            #gr = run_grid(10,tests,provided_keys = list((freei,freej)), hold_constant = hc, mp_in = params)
            fp = list(copy.copy(free_param))
            ax[i,j] = plot_surface(gr,ax[i,j],fp,imshow=False)

        if i < j:
            free_param = list(copy.copy(list(free_param)))
            if len(free_param) == 2:
                ax[i,j] = plot_scatter(hof,ax[i,j],free_param)
        # To Pandas:
        k = 0
        df.insert(i, j, k, free_param)
        k = 1
        df.insert(i, j, k, hc)
        k = 2
        df.insert(i, j, k, cpparams)
        k = 3
        df.insert(i, j, k, gr)
        '''
        where, i and j are indexs to the 3 by 3 (9 element) subplot matrix,
        and `k`-dim-0 is the parameter(s) that were free to vary (this can be two free in the case for i<j,
        or one free to vary for i==j).
        `k`-dim-1, is the parameter(s) that were held constant.
        `k`-dim-2 `cpparams` is a per parameter dictionary, whose values are tuples that mark the edges of (free)
        parameter ranges. `k`-dim-3 is the the grid that results from varying those parameters
        (the grid results can either be square (if len(free_param)==2), or a line (if len(free_param==1)).
        '''
    plt.savefig(str('cross_section_and_surfaces.png'))
    return matrix, df


def grids(hof,tests,params):
    '''                                                                                 
    Obtain using the best candidate Gene (HOF, NU-tests, and expanded parameter ranges found via                                                                               
    exploring extreme edge cases of parameters                                          
    plot a error surfaces, and cross sections, about the optima.                                                                                                                
    where, i and j are indexs to the 3 by 3 (9 element) subplot matrix,                 
    and `k`-dim-0 is the parameter(s) that were free to vary (this can be two free in the case for i<j,                                                                        
    or one free to vary for i==j).                                                      
    `k`-dim-1, is the parameter(s) that were held constant.                             
    `k`-dim-2 `cpparams` is a per parameter dictionary, whose values are tuples that mark the edges of (free)                                                                  
    parameter ranges. `k`-dim-3 is the the grid that results from varying those parameters                                                                                     
    (the grid results can either be square (if len(free_param)==2), or a line (if len(free_param==1)).                                                                         
    '''               
    dim = len(hof[0].dtc.attrs.keys())
    flat_iter = iter([(i,ki,j,kj) for i,ki in enumerate(hof[0].dtc.attrs.keys()) for j,kj in enumerate(hof[0].dtc.attrs.keys())])
    matrix = [[0 for x in range(dim)] for y in range(dim)]
    plt.clf()

    fig,ax = plt.subplots(dim,dim,figsize=(10,10))
    cnt = 0
    mat = np.zeros((dim,dim))
    df = pd.DataFrame(mat)

    for i,freei,j,freej in flat_iter:
        free_param = set([freei,freej]) # construct a small-set out of the indexed keys 2. If both keys are
        # are the same, this set will only contain one index
        bs = set(hof[0].dtc.attrs.keys()) # construct a full set out of all of the keys available, including ones not indexed here.
        diff = bs.difference(free_param) # diff is simply the key that is not indexed.
        # hc is the dictionary of parameters to be held constant
        # if the plot is 1D then two parameters should be held constant.
        hc =  {}
        for d in diff:
            hc[d] = hof[0].dtc.attrs[d]

        cpparams = {}
        if i == j:
            assert len(free_param) == len(hc) - 1
            assert len(hc) == len(free_param) + 1
            # zoom in on optima
            cpparams['freei'] = (np.min(params[freei]), np.max(params[freei]))

            gr = run_grid(10,tests,provided_keys = freei, hold_constant = hc,mp_in = params)
            # make a psuedo test, that still depends on input Parametersself.
            # each test evaluates a normal PDP.
            fp = list(copy.copy(free_param))
            ax[i,j] = plot_line(gr,ax[i,j],fp)
        if i >j:
            assert len(free_param) == len(hc) + 1
            assert len(hc) == len(free_param) - 1
            cpparams['freei'] = (np.min(params[freei]), np.max(params[freei]))
            cpparams['freej'] = (np.min(params[freej]), np.max(params[freej]))

            gr = run_grid(10,tests,provided_keys = list((freei,freej)), hold_constant = hc, mp_in = params)
            fp = list(copy.copy(free_param))
            ax[i,j] = plot_surface(gr,ax[i,j],fp,imshow=False)

        if i < j:
            free_param = list(copy.copy(list(free_param)))
            if len(free_param) == 2:
                ax[i,j] = plot_scatter(hof,ax[i,j],free_param)
        # To Pandas:
        k = 0
        df.insert(i, j, k, free_param)
        k = 1
        df.insert(i, j, k, hc)
        k = 2
        df.insert(i, j, k, cpparams)
        k = 3
        df.insert(i, j, k, gr)
        '''
        where, i and j are indexs to the 3 by 3 (9 element) subplot matrix,
        and `k`-dim-0 is the parameter(s) that were free to vary (this can be two free in the case for i<j,
        or one free to vary for i==j).
        `k`-dim-1, is the parameter(s) that were held constant.
        `k`-dim-2 `cpparams` is a per parameter dictionary, whose values are tuples that mark the edges of (free)
        parameter ranges. `k`-dim-3 is the the grid that results from varying those parameters
        (the grid results can either be square (if len(free_param)==2), or a line (if len(free_param==1)).
        '''
    plt.savefig(str('cross_section_and_surfaces.png'))
    return matrix, df


opt_keys = [str('vr'),str('a'),str('b')]
nparams = len(opt_keys)

try:
    assert 1==2
    with open('ranges.p','rb') as f:
        [fc,mp] = pickle.load(f)

except:
    # algorithmically find the the edges of parameter ranges, via a course grained
    # sampling of extreme parameter values
    # to find solvable instances of Izhi-model, (models with a rheobase value).
    import explore_ranges
    fc, mp = explore_ranges.pre_run(tests_,opt_keys)
    with open('ranges.p','wb') as f:
        pickle.dump([fc,mp],f)



# update and simplify model parameter dictionary.
# this is probably unnecessary
for k,v in mp.items():
    if type(v) is type(tuple((0,0))):
        mp[k] = np.linspace(v[0],v[1],7)


# get a genetic algorithm that operates on this new parameter range.
try:
    assert 1==2
    with open('package.p','rb') as f:
        package = pickle.load(f)

except:

    package = run_ga(mp,6,tests_,provided_keys = opt_keys)
    with open('package.p','wb') as f:
        pickle.dump(package,f)

pop = package[0]
history = package[4]
gen_vs_pop =  package[6]
hof = package[1]


df, matrix = grids(hof,tests_,mp)
with open('surfaces.p','wb') as f:
    pickle.dump([df,matrix],f)
