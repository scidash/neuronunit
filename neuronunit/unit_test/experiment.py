import pickle

from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION

from neuronunit.optimization import get_neab
import copy
import os
from neuronunit.optimization.optimization_management import run_ga
from neuronunit.optimization.model_parameters import model_params, path_params
from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION

import matplotlib as mpl
mpl.use('Agg')
#mpl.switch_backend('Agg')


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

plot_surface = plottools.plot_surface
scatter_surface = plottools.plot_surface

with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)
from matplotlib.colors import LogNorm
from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid#, mock_grid
from neuronunit.models.NeuroML2 import model_parameters as modelp

from neuronunit.models.NeuroML2 .model_parameters import path_params



electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
test, observation = electro_tests[0]

import quantities as pq



opt_keys = ['a','b','vr']
nparams = len(opt_keys)
mp = modelp.model_params
observation = {'a':[np.median(mp['a']),np.std(mp['a'])], 'b':[np.median(mp['b']),np.std(mp['b'])], 'vr':[np.median(mp['vr']),np.std(mp['vr'])]}


tests = copy.copy(electro_tests[0][0])
tests_ = []
tests_ += [tests[0]]
tests_ += tests[4:7]


with open('ga_run.p','rb') as f:
    package = pickle.load(f)
pop = package[0]
print(pop[0].dtc.attrs.items())
history = package[4]
gen_vs_pop =  package[6]
hof = package[1]


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

    # impute missings
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
'''
Depricated
def check_range(matrix,hof):
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
            print(line)
            if i==j:
                line = [ np.sum(list(g.dtc.scores.values())) for g in gr]
                (newrange, range_adj) = check_line(line,newrange)
                print(newrange,'newrange')
    return (newrange, range_adj)
'''
def check_line(line,gr,newrange):
    range_adj = False
    key = list(newrange.keys())[0]
    #keys = keys[0]
    min_ = np.min(line)
    print(min_,line[0],line[1],'diff?')
    if line[0] == min_:
        #print('hit')
        attrs = gr[0].dtc.attrs[key]
        remin = - np.abs(attrs)*10
        remax = np.abs(gr[-1].dtc.attrs[key])*10
        nr = np.linspace(remin,remax,3)
        newrange[key] = nr
        range_adj = True

    if line[-1] == min_:
        #print('hit')
        attrs = gr[-1].dtc.attrs[key]
        remin = - np.abs(attrs)*10
        remax = np.abs(gr[-1].dtc.attrs[key])*10
        nr = np.linspace(remin,remax,3)
        newrange[key] = nr
        range_adj = True
    return (newrange, range_adj)

def mp_process(newrange):
    from neuronunit.models.NeuroML2 import model_parameters as modelp
    mp = copy.copy(modelp.model_params)
    for k,v in newrange.items():
        if type(v) is not type(None):
            mp[k] = v
    return mp

def pre_run(tests):
    dim = len(hof[0].dtc.attrs.keys())
    flat_iter = [ (i,ki,j,kj) for i,ki in enumerate(hof[0].dtc.attrs.keys()) for j,kj in enumerate(hof[0].dtc.attrs.keys()) ]
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
            from neuronunit.models.NeuroML2 import model_parameters as modelp
            mp = copy.copy(modelp.model_params)
            gr = run_grid(3,tests,provided_keys = free_param ,hold_constant = hc, mp_in = mp)
            # make a psuedo test, that still depends on input Parametersself.
            # each test evaluates a normal PDP.
            line = [ np.sum(list(g.dtc.scores.values())) for g in gr]
            nr = {str(list(free_param)[0]):None}
            newrange, range_adj = check_line(line,gr,nr)
            while range_adj == True:

                mp = mp_process(newrange)
                gr = run_grid(3,tests,provided_keys = newrange ,hold_constant = hc, mp_in = mp)
                # make a psuedo test, that still depends on input Parametersself.
                # each test evaluates a normal PDP.
                nr = {}
                line = [ np.sum(list(g.dtc.scores.values())) for g in gr]
                newrange, range_adj = check_line(line,gr,newrange)

                mp = mp_process(newrange)

    with open('parameter_bf_ranges.p','wb') as f:
        pickle.dump(mp,f)
    import pdb; pdb.set_trace()
    package = run_ga(mp,nparams*2,12,tests_,provided_keys = opt_keys)#, use_cache = True, cache_name='simple')
    return package

matrix = pre_run(tests=tests_)



#plotss(matrix)
#except:
