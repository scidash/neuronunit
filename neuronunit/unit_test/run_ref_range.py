
from neuronunit.optimization import get_neab
import copy
import os

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
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import quantities as pq

from neuronunit import plottools
ax = None

plot_surface = plottools.plot_surface
scatter_surface = plottools.plot_surface



def get_tests():
    from neuronunit.optimization import get_neab
    electro_path = str(os.getcwd())+'/pipe_tests.p'
    assert os.path.isfile(electro_path) == True
    with open(electro_path,'rb') as f:
        electro_tests = pickle.load(f)
    #import pdb; pdb.set_trace()
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
    # from
    # https://github.com/russelljjarvis/neuronunit/blob/dev/neuronunit/unit_test/progress_report_4thJuly.ipynb
    # Not rendered
    # https://github.com/russelljjarvis/neuronunit/blob/dev/neuronunit/unit_test/progress_report_.ipynb
    gr = [ g for g in gr if type(g.dtc) is not type(None) ]
    gr = [ g for g in gr if type(g.dtc.scores) is not type(None) ]
    ax.cla()
    #gr = [ g
    gr_ = []
    index = 0
    for i,g in enumerate(gr):
       if type(g.dtc) is not type(None):
           gr_.append(g)
       else:
           index = i

    z = [ np.sum(list(p.dtc.scores.values())) for p in gr ]
    x = [ p.dtc.attrs[str(keys[0])] for p in gr ]
    y = [ p.dtc.attrs[str(keys[1])] for p in gr ]

    if len(x) != 100:
        delta = 100-len(x)
        for i in range(0,delta):
            x.append(np.mean(x))
            y.append(np.mean(y))
            z.append(np.mean(z))

    xx = np.array(x)
    yy = np.array(y)
    zz = np.array(z)
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

import pandas as pd


def grids(hof,tests,params):
    dim = len(hof[0].dtc.attrs.keys())
    flat_iter = [ (i,ki,j,kj) for i,ki in enumerate(hof[0].dtc.attrs.keys()) for j,kj in enumerate(hof[0].dtc.attrs.keys()) ]
    matrix = [[0 for x in range(dim)] for y in range(dim)]
    plt.clf()
    fig,ax = plt.subplots(dim,dim,figsize=(10,10))
    mat = np.array((dim,dim))
    cnt = 0
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

        if i == j:
            assert len(free_param) == len(hc) - 1
            assert len(hc) == len(free_param) + 1
            min_ = np.min(params[free_param])
            max_ = np.max(params[free_param])
            limited = params[free_param]
            gr = run_grid(10,tests,provided_keys = free_param ,hold_constant = hc,mp_in = limited)
            # make a psuedo test, that still depends on input Parametersself.
            # each test evaluates a normal PDP.
            matrix[i][j] = ( free_param,gr )
            fp = list(copy.copy(free_param))
            ax[i,j] = plot_line(gr,ax[i,j],fp)
        if i >j:
            assert len(free_param) == len(hc) + 1
            assert len(hc) == len(free_param) - 1
            # what I want to do, I want to plot grid lines not a heat map.
            # I want to plot bd.attrs is cross hairs,
            # I want the range of the plot shown to be bigger than than the grid lines.
            limited = {i:params[i] for i in list(free_param) }
            gr = run_grid(10,tests,provided_keys = free_param ,hold_constant = hc, mp_in = limited)
            fp = list(copy.copy(free_param))
            ax[i,j] = plot_surface(gr,ax[i,j],fp,imshow=False)
            matrix[i][j] = ( free_param,gr )

        if i < j:
            matrix[i][j] = ( free_param,gr )
            fp = list(copy.copy(free_param))
            if len(fp) == 2:
                ax[i,j] = plot_scatter(hof,ax[i,j],fp)

        mat[i,j] = ( free_param,gr )
    plt.savefig(str('cross_section_and_surfaces.png'))
    return matrix, mat


opt_keys = ['a','b','vr']
nparams = len(opt_keys)

try:
    assert 1==2
    with open('ranges.p','rb') as f:
        [fc,mp] = pickle.load(f)

except:
    import explore_ranges
    fc, mp = explore_ranges.pre_run_two(tests_,opt_keys)
    with open('ranges.p','wb') as f:
        pickle.dump([fc,mp],f)
    lines = [ v['line'] for v in fc.values() ]
    cnts = [ v['cnt'] for v in fc.values() ]
    print('cnts ',cnts)
    print('lines ', lines)


#import pdb; pdb.set_trace()
#
# Rick wants this to be a data frame
#
# def run_ga(model_params,nparams,npoints,test, provided_keys = None, nr = None):
#import pdb; pdb.set_trace()
#def run_ga(model_params,npoints,test, provided_keys = None, nr = None):

package = run_ga(mp,6,tests_,provided_keys = opt_keys)
pop = package[0]
history = package[4]
gen_vs_pop =  package[6]
hof = package[1]

#import pdb; pdb.set_trace()
mat, matrix = grids(hof,tests_,param_ranges)
with open('surfaces.p','wb') as f:
    pickle.dump([mat,matrix],f)


print('cnts ',cnts)
print('param_ranges ',param_ranges)
print('lines ', lines)
import pdb; pdb.set_trace()

#with open('ga_run.p','wb') as f:
#    pickle.dump(package,f)


def plot_vm(hof,ax,key):
    ax.cla()
    ax.set_title(' {0} vs  $V_{M}$'.format(key[0]))
    best_dtc = hof[0].dtc
    best_rh = hof[0].dtc.rheobase
    neuron = None
    model = ReducedModel(path_params['model_path'],name = str('regular_spiking'),backend =('NEURON',{'DTC':best_dtc}))
    params = {'injected_square_current':
            {'amplitude': best_rh, 'delay':DELAY, 'duration':DURATION}}
    results = modelrs.inject_square_current(params)
    vm = model.get_membrane_potential()
    times = vm.times
    ax.plot(times,vm)
    return ax


def plotss(matrix,hof):
    dim = np.shape(matrix)[0]
    print(dim)
    cnt = 0
    fig,ax = plt.subplots(dim,dim,figsize=(10,10))
    flat_iter = []
    for i,k in enumerate(matrix):
        for j,r in enumerate(k):
            keys = list(r[0])
            gr = r[1]
            if i==j:
                ax[i,j] = plot_vm(hof,ax[i,j],keys)
            if i>j:
                ax[i,j] = plot_surface(gr,ax[i,j],keys,imshow=False)
            if i < j:
                ax[i,j] = plot_scatter(hof,ax[i,j],keys)
        print(i,j)
    plt.savefig(str('surface_and_vm.png'))
    return None

with open('surfaces.p','rb') as f:
    matrix = pickle.load(f)
