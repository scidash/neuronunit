import pickle
from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION


from neuronunit.optimization import get_neab
import copy
import os
from neuronunit.optimization.optimization_management import run_ga

import matplotlib as mpl
mpl.use('agg')


electro_path = str(os.getcwd())+'/pipe_tests.p'
print(os.getcwd())
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)

electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
test, observation = electro_tests[0]


import matplotlib.pyplot as plt



from neuronunit.optimization import get_neab
import copy
import os
import pickle
electro_path = str(os.getcwd())+'/pipe_tests.p'
from neuronunit import plottools
import numpy as np
ax = None
from neuronunit.optimization import exhaustive_search as es

with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)
from matplotlib.colors import LogNorm
from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid#, mock_grid
from neuronunit.optimization import model_parameters as modelp
from neuronunit.optimization.model_parameters import path_params
mp = modelp.model_params



electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
test, observation = electro_tests[0]
import quantities as pq

from neuronunit.optimization import model_parameters as modelp
mp = modelp.model_params



opt_keys = ['a','b','vr']
nparams = len(opt_keys)

observation = {'a':[np.median(mp['a']),np.std(mp['a'])], 'b':[np.median(mp['b']),np.std(mp['b'])], 'vr':[np.median(mp['vr']),np.std(mp['vr'])]}

tests = copy.copy(electro_tests[0][0])
tests_ = []
tests_ += [tests[0]]
tests_ += tests[4:7]
#import pdb
#pdb.set_trace()
try:
    #assert 1 == 2
    with open('ga_run.p','rb') as f:
        package = pickle.load(f)
    pop = package[0]
    print(pop[0].dtc.attrs.items())
    history = package[4]
    gen_vs_pop =  package[6]
    hof = package[1]
except:
    print(mp)
    nparams = len(opt_keys)
    package = run_ga(mp,nparams*2,12,tests_,provided_keys = opt_keys)#, use_cache = True, cache_name='simple')
    pop = package[0]
    history = package[4]
    gen_vs_pop =  package[6]
    hof = package[1]

    with open('ga_run.p','wb') as f:
        pickle.dump(package,f)

    grid_results = {}

#import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt



def plot_scatter(hof,ax,keys):
    z = np.array([ np.sum(list(p.dtc.scores.values())) for p in hof ])
    x = np.array([ p.dtc.attrs[str(keys[0])] for p in hof ])
    if len(keys) != 1:
        y = np.array([ p.dtc.attrs[str(keys[1])] for p in hof ])

        ax.cla()
        ax.set_title(' {0} vs {1} '.format(keys[0],keys[1]))
        ax.scatter(x, y, c=y, s=125)#, cmap='gray')

        #ax.scatter(x, y, z, [3 for i in x] )
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
    
        #ax.imshow(Z)
    #ax.pcolormesh(xi, yi, zi, edgecolors='black')
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

def grids(hof,tests):
    dim = len(hof[0].dtc.attrs.keys())
    flat_iter = [ (i,ki,j,kj) for i,ki in enumerate(hof[0].dtc.attrs.keys()) for j,kj in enumerate(hof[0].dtc.attrs.keys()) ]
    matrix = [[0 for x in range(dim)] for y in range(dim)]
    plt.clf()
    fig,ax = plt.subplots(dim,dim,figsize=(10,10))
    #fig1,ax1 = plt.subplots(dim,dim,figsize=(10,10))

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
            gr = run_grid(10,tests,provided_keys = free_param ,hold_constant = hc)
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

            gr = run_grid(10,tests,provided_keys = free_param ,hold_constant = hc)
            
            fp = list(copy.copy(free_param))
            #if len(gr) == 100:
            ax[i,j] = plot_surface(gr,ax[i,j],fp,imshow=False)

            matrix[i][j] = ( free_param,gr )

        if i < j:
            matrix[i][j] = ( free_param,gr )

            fp = list(copy.copy(free_param))
            if len(fp) == 2:
                ax[i,j] = plot_scatter(hof,ax[i,j],fp)

    plt.savefig(str('first_surfaces.png'))
    return matrix

import matplotlib.pyplot as plt
from neuronunit.models.reduced import ReducedModel

def plot_vm(hof,ax,key):
    ax.cla()
    best_dtc = hof[1].dtc
    best_rh = hof[1].dtc.rheobase
    print( [ h.dtc.rheobase for h in hof ])
    
    neuron = None
    model = ReducedModel(path_params['model_path'],name = str('regular_spiking'),backend =('NEURON',{'DTC':best_dtc}))
    params = {'injected_square_current':
            {'amplitude': best_rh['value'], 'delay':DELAY, 'duration':DURATION}}
    model.set_attrs(hof[0].dtc.attrs)
    results = model.inject_square_current(params)
    print(model.attrs,best_dtc.attrs,best_rh)
    vm = model.get_membrane_potential()
    times = vm.times
    ax.plot(times,vm)
    #ax.xlabel('ms')
    #ax.ylabel('mV')
    #ax.set_xlim(np.min(x),np.max(x))
    #ax.set_ylim(np.min(z),np.max(z))
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
    plt.savefig(str('other_surfaces.png'))
    return None    


def evidence(matrix,hof):
    dim = np.shape(matrix)[0]
    print(dim)
    cnt = 0
    fig,ax = plt.subplots(dim,dim,figsize=(10,10))
    flat_iter = []
    newrange = {}
    for i,k in enumerate(matrix):
        for j,r in enumerate(k):

            keys = list(r[0])
            gr = r[1]
            line = [ np.sum(list(g.dtc.scores.values())) for g in gr]
            print(line)
            if i==j:
                keys = keys[0]
                min_ = np.min(line)
                print(min_,line[0],line[1],'diff?')
                if line[0] == min_:
                    print('hit')
                    attrs = gr[0].dtc.attrs[keys]
                    remin = - np.abs(attrs)*10
                    remax = np.abs(gr[-1].dtc.attrs[keys])*10
                    nr = np.linspace(remin,remax,10)
                    newrange[keys] = nr

                if line[-1] == min_:
                    print('hit')
                    attrs = gr[-1].dtc.attrs[keys]
                    remin = - np.abs(attrs)*10
                    remax = np.abs(gr[-1].dtc.attrs[keys])*10
                    nr = np.linspace(remin,remax,10)
                    newrange[keys] = nr

                print(newrange,'newrange')
    return newrange
#matrix = grids(hof,tests=tests_)
#surfaces.p
#with open('surfaces.p','wb') as f:
#    pickle.dump(matrix,f)
with open('surfaces.p','rb') as f:
    matrix = pickle.load(f)
import pdb

newrange = evidence(matrix,hof)
#_ = plotss(matrix,hof)
#except:
print('finished')
print(newrange)
exit
