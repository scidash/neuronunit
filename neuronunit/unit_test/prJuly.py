import pickle
from neuronunit.optimization import get_neab
import copy
import os
from neuronunit.optimization.optimization_management import run_ga
from neuronunit.optimization.model_parameters import model_params, path_params
from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION

import matplotlib as mpl
mpl.use('Agg')


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
from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid, mock_grid
from neuronunit.optimization import model_parameters as modelp
mp = modelp.model_params



electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
test, observation = electro_tests[0]


from neuronunit.tests import FakeTest
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

try:
    assert 1==2
    with open('ga_run.p','rb') as f:
        package = pickle.load(f)
    pop = package[0]
    print(pop[0].dtc.attrs.items())
    history = package[4]
    gen_vs_pop =  package[6]
    hof = package[1]
except:
    print(mp)
    #import pdb; pdb.set_trace()
    #MU = 6
    nparams = len(opt_keys)
    package = run_ga(mp,nparams,10,tests_,provided_keys = opt_keys)#, use_cache = True, cache_name='simple')
    #import pdb; pdb.set_trace()
    pop = package[0]
    history = package[4]
    gen_vs_pop =  package[6]
    hof = package[1]

    with open('ga_run.p','wb') as f:
        pickle.dump(package,f)

    grid_results = {}


visited = []
#cnt = 0
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

def plot_seperate(gr,keys,ax):
    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm

    z = np.array([ np.sum(list(p.dtc.scores.values())) for p in gr ])
    x = np.array([ p.dtc.attrs[str(keys[0])] for p in gr ])
    if len(keys) != 1:
        y = np.array([ p.dtc.attrs[str(keys[1])] for p in gr ])
    N = int(np.sqrt(len(x)))
    X = x.reshape((N, N))
    Y = y.reshape((N, N))
    Z = z.reshape((N, N))


    #f = figure(figsize=(6.2,5.6))
    #ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
    #axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])
    import matplotlib.cm as cm
    im = plt.imshow(Z, cmap=cm.gray_r, norm=LogNorm(vmin=np.min(z), vmax=np.max(z)))

    ax.colorbar(im)#), cax=axcolor, format='$%.2f$')
    #plt.savefig(str(keys[0])+str(keys[1])+'imshow_log.png')
    return ax

def plot_surface(gr,ax,keys):
    # from
    # https://github.com/russelljjarvis/neuronunit/blob/dev/neuronunit/unit_test/progress_report_4thJuly.ipynb
    # Not rendered
    # https://github.com/russelljjarvis/neuronunit/blob/dev/neuronunit/unit_test/progress_report_.ipynb
    zz = np.array([ np.sum(list(p.dtc.scores.values())) for p in gr ])
    xx = np.array([ p.dtc.attrs[str(keys[0])] for p in gr ])
    if len(keys) != 1:
        yy = np.array([ p.dtc.attrs[str(keys[1])] for p in gr ])
    dim = len(xx)
    '''

    zi, yi, xi = np.histogram2d(yy, xx, bins=(6,6), weights=zz, normed=True)
    counts, _, _ = np.histogram2d(yy, xx, bins=(6,6))
    zi = zi / counts
    zi = np.ma.masked_invalid(zi)
    print(zi)
    '''
    N = int(np.sqrt(len(x)))
    X = xx.reshape((N, N))
    Y = yy.reshape((N, N))
    Z = zz.reshape((N, N))

    ax.pcolormesh(X, Y, Z, edgecolors='black')
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


def plot_planes(package,matrix):

    axis = [ [ str('vr'), str('a'), str('b') ], [ str('vr'), str('b'), str('a')], [ str('b'), str('a'), str('vr') ] ]
    matrix[i][j]
    gen_vs_pop = package[6]
    pop = gen_vs_pop[-1]

    for k in axis:

        zz = [ np.sum(list(i.dtc.scores.values())) for i in grid_results ]
        yy = [ i.dtc.attrs[k[1]] for i in grid_results ]
        xx = [ i.dtc.attrs[k[0]] for i in grid_results ]

        last_frame = len(gen_vs_pop)
        other_points = []
        pf_points = []
        hof_points = []
        labels = []

        pf = package[2]
        hof = package[1]

        for p in pop:
            xy = []
            for key in k:
                v = p.dtc.attrs[key]
                xy.append(v)
                labels.append(key)
                other_points.append(xy)


        for p in pf:
            xy = []
            for key in k:
                v = p.dtc.attrs[key]
                xy.append(v)
                labels.append(key)
                pf_points.append(xy)

        for p in hof:
            xy = []
            for key in k:
                v = p.dtc.attrs[key]
                xy.append(v)
                labels.append(key)
                hof_points.append(xy)


        zi, yi, xi = np.histogram2d(yy, xx, bins=(6,6), weights=zz, normed=False)
        counts, _, _ = np.histogram2d(yy, xx, bins=(6,6))

        zi = zi / counts
        zi = np.ma.masked_invalid(zi)
        fig, ax = plt.subplots()
        scat = ax.pcolormesh(xi, yi, zi, edgecolors='black', cmap=green_purple)
        fig.colorbar(scat)
        ax.margins(0.05)

        #if i == last_frame-1:
        for xy in hof_points:
            ax.plot(xy[0], xy[1],'y*',label ='hall of fame')
        for xy in pf_points:
            ax.plot(xy[0], xy[1],'b*',label ='pareto front')
            #legend = ax.legend([rect("r"), rect("g"), rect("b")], ["gene population","pareto front","hall of fame"])


        for xy in other_points:
            ax.plot(xy[0], xy[1],'ro',label ='gene population')
        ax.margins(0.05)

        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.savefig(str(k)+str('grid_frag_surfaces.png'))

def grids(hof,tests):
    dim = len(hof[0].dtc.attrs.keys())
    flat_iter = [ (i,ki,j,kj) for i,ki in enumerate(hof[0].dtc.attrs.keys()) for j,kj in enumerate(hof[0].dtc.attrs.keys()) ]
    matrix = [[0 for x in range(dim)] for y in range(dim)]
    plt.clf()
    fig,ax = plt.subplots(dim,dim,figsize=(10,10))
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
            print('hc',hc)
            #import pdb; pdb.set_trace()
            #new_hc = {k:float(v)+np.random.uniform(-float(v),float(v)) for k,v in hc.items() }
            gr = run_grid(10,tests,provided_keys = free_param ,hold_constant = hc)
            #print(gr[0].dtc.scores())
            print([g.dtc.scores for g in gr])
            print(gr[0].dtc.scores)
            #import pdb; pdb.set_trace()
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

            #new_hc = {k:float(v)+np.random.uniform(-float(v),float(v)) for k,v in hc.items() }cx
            gr = run_grid(10,tests,provided_keys = free_param ,hold_constant = hc)
            fp = list(copy.copy(free_param))
            #print('free param: ',fp,'hold constant: ',hc)
            ax[i,j] = plot_surface(gr,ax[i,j],fp)
            #_ = plot_seperate(gr,fp)

            matrix[i][j] = ( free_param,gr )

        if i < j:
            fp = list(copy.copy(free_param))
            if len(fp) == 2:
                ax[i,j] = plot_scatter(hof,ax[i,j],fp)

    plt.savefig(str('first_surfaces.png'))
    return matrix

import matplotlib.pyplot as plt

from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization.model_parameters import model_params, path_params
from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION

def plot_vm(hof,ax,key):
    ax.cla()
    ax.set_title(' {0} vs  $V_{M}$'.format(key[0]))
    best_dtc = hof[0].dtc
    best_rh = hof[0].dtc.rheobase

    dtc.attrs = attrs

    neuron = None

    modelrs = ReducedModel(path_params['model_path'],name = str('regular_spiking'),backend =('NEURON',{'DTC':best_dtc}))

    params = {'injected_square_current':
            {'amplitude': best_rh, 'delay':DELAY, 'duration':DURATION}}

    modelrs.inject_square_current(params)

    #z = np.array([ np.sum(list(p.dtc.scores.values())) for p in gr ])
    #x = np.array([ p.dtc.attrs[key[0]] for p in gr ])
    times = vm.times

    ax.plot(times,vm)
    ax.xlabel('ms')
    ax.ylabel('mV')
    ax.set_xlim(np.min(x),np.max(x))
    ax.set_ylim(np.min(z),np.max(z))
    return ax


def plotss(matrix):
    #dim = len(two_d.keys())
    dim = np.shape(matrix)[0]
    print(dim)
    cnt = 0
    for i,k in enumerate(matrix):
        for j,r in enumerate(k):
            keys = list(r[0])
            gr = r[1]
            cnt += 1
            z = np.array([ np.sum(list(p.dtc.scores.values())) for p in gr ])

    fig,ax = plt.subplots(dim,dim,figsize=(10,10))
    flat_iter = []
    for i,k in enumerate(matrix):
        for j,r in enumerate(k):

            keys = list(r[0])
            gr = r[1]

            if i==j:
                ax[i,j] = plot_line(gr,ax[i,j])
            if i>j:
                ax[i,j] = plot_surface(gr,ax[i,j])

        print(i,j)
    plt.savefig(str('surfaces.png'))
    return None
matrix = grids(hof,tests=tests_)
with open('surfaces.p','wb') as f:
    pickle.dump(matrix,f)
with open('surfaces.p','rb') as f:
    matrix = pickle.load(f)





#plotss(matrix)
#except:
