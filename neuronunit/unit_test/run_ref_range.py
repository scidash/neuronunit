import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# verify that backend is appropriate before compute job:
plt.clf()

import copy
import os

import pickle
from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid, run_simple_grid
#from neuronunit.optimization import exhaustive_search as es

from neuronunit.optimization.optimization_management import run_ga
from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
import matplotlib.pyplot as plt
from neuronunit.models.reduced import ReducedModel
from itertools import product

import quantities as pq
from numba import jit
from neuronunit import plottools
ax = None
import sys
import pickle



from neuronunit.optimization import get_neab

grid_results = {}

def plot_scatter(history,ax,keys,constant):
    pop = [ v for v in history.genealogy_history.values() ]
    z = np.array([ p.dtc.get_ss() for p in pop ])
    x = np.array([ p.dtc.attrs[str(keys[0])] for p in pop ])
    y = np.array([ p.dtc.attrs[str(keys[1])] for p in pop ])
    ax.cla()
    ax.set_title('held constant: '+str(constant))
    ax.scatter(x, y, c=y, s=125)#, cmap='gray')
    ax.set_xlabel('free: '+str(keys[0]))
    ax.set_ylabel('free: '+str(keys[1]))
    return ax

def plot_surface(gr,ax,keys,constant,imshow=True):
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
    zz = np.array([ p.dtc.get_ss() for p in gr ])
    dim = len(xx)
    N = int(np.sqrt(len(xx)))
    X = xx.reshape((N, N))
    Y = yy.reshape((N, N))
    Z = zz.reshape((N, N))
    if imshow==True:
        img = ax.pcolormesh(X, Y, Z, edgecolors='black')
        #ax.colorbar()

    else:


        import seaborn as sns;
        sns.set()

        current_palette = sns.color_palette()
        sns.palplot(current_palette)

        #df = pd.DataFrame(Z, columns=xx)

        img = sns.heatmap(Z)#,cm=current_palette)
        #ax.colorbar()

    ax.set_title(' {0} vs {1} '.format(keys[0],keys[1]))
    return ax, img

def plot_line_ss(gr,ax,free,hof,constant):
    ax.cla()

    ax.set_title(' {0} vs  score'.format(free))
    z = np.array([ p.dtc.get_ss() for p in gr ])
    print(str(free))
    print(free)
    x = np.array([ p.dtc.attrs[str(free)] for p in gr ])

    y = hof[0].dtc.attrs[free]
    i = hof[0].dtc.get_ss()
    #ax.hold(True)
    ax.scatter(x,z)
    ax.scatter(y,i)
    ax.plot(x,z)

    ax.set_xlabel(str(key[0]))
    ax.set_ylabel(str('Sum of Errors'))
    return ax

def plot_agreement(ax,gr,key,hof):
    dtcpop = [ g.dtc for g in gr ]
    for dtc in dtcpop:
        if hasattr(score,'prediction'):
            if type(score.prediction) is not type(None):
                dtc.score[str(t)][str('prediction')] = score.prediction
                dtc.score[str(t)][str('observation')] = score.observation
                boolean_means = bool('mean' in score.observation.keys() and 'mean' in score.prediction.keys())
                boolean_value = bool('value' in score.observation.keys() and 'value' in score.prediction.keys())

            if boolean_means:
                dtc.score[str(t)][str('agreement')] = np.abs(score.observation['mean'] - score.prediction['mean'])

            if boolean_value:
                dtc.score[str(t)][str('agreement')] = np.abs(score.observation['value'] - score.prediction['value'])

    ss = hof[0].dtc.score
    #for v in ss:
    if str('agreement') in ss.keys():
        ax.plot( [ v['agreement'] for v in list(ss.values()) ], [ i for i in range(0,len(ss.values())) ] )
        ax.plot( [ v['prediction'] for v in list(ss.values()) ], [ i for i in range(0,len(ss.values())) ] )
        ax.plot( [ v['observation'] for v in list(ss.values()) ], [ i for i in range(0,len(ss.values())) ] )
    return ax

from collections import OrderedDict

def grids(hof,tests,ranges,us,history):
    '''
    Obtain using the best candidate Gene (HOF, NU-tests, and expanded parameter ranges found via
    exploring extreme edge cases of parameters

    plot a error surfaces, and cross sections, about the optima in a 3by3 subplot matrix.

    where, i and j are indexs to the 3 by 3 (9 element) subplot matrix,
    and `k`-dim-0 is the parameter(s) that were free to vary (this can be two free in the case for i<j,
    or one free to vary for i==j).
    `k`-dim-1, is the parameter(s) that were held constant.
    `k`-dim-2 `cpparams` is a per parameter dictionary, whose values are tuples that mark the edges of (free)
    parameter ranges. `k`-dim-3 is the the grid that results from varying those parameters
    (the grid results can either be square (if len(free_param)==2), or a line (if len(free_param==1)).
    '''
    #scores = hof[0].dtc.scores = OrderedDict(hof[0].dtc.scores)
    hof[0].dtc.attrs = OrderedDict(hof[0].dtc.attrs)
    attrs = hof[0].dtc.attrs
    dim = len(hof[0].dtc.attrs.keys())
    best_param_keys = hof[0].dtc.attrs.keys()
    flat_iter = iter([(i,freei,j,freej) for i,freei in enumerate(best_param_keys) for j,freej in enumerate(best_param_keys)])

    plt.clf()
    fig0,ax0 = plt.subplots(dim,dim,figsize=(10,10))
    #fig1,ax1 = plt.subplots(dim,dim,figsize=(10,10))

    cnt = 0
    temp = []
    loc_key = {}

    for k,v in attrs.items():
        loc_key[k] = attrs[k]
        ranges[k] = ( loc_key[k]- 1*np.abs(ranges[k]), loc_key[k]+1*np.abs(ranges[k]) )

    for i,freei,j,freej in flat_iter:
        if i == j:
            free_param = freei
        else:
            free_param = [freei,freej]
        free_param_set = set(free_param) # construct a small-set out of the indexed keys 2. If both keys are
        # are the same, this set will only contain one index
        bs = set(best_param_keys) # construct a full set out of all of the keys available, including ones not indexed here.
        diff = bs.difference(free_param_set) # diff is simply the key that is not indexed.
        # hc is the dictionary of parameters to be held constant
        # if the plot is 1D then two parameters should be held constant.
        hc =  {}
        for d in diff:
            hc[d] = hof[0].dtc.attrs[d]

        cpparams = {}


        cpparams['freei'] = (np.min(ranges[freei]), np.max(ranges[freei]))

                # make a psuedo test, that still depends on input Parametersself.
        # each test evaluates a normal PDP.
        fp = list(copy.copy(free_param))
                   #plot_line_ss(gr,ax,key,hof,free,constant)
        if i == j:
            gr = run_simple_grid(10, tests, ranges, freei, hold_constant = hc)

            ax0[i,j] = plot_line_ss(gr,ax0[i,j],freei,hof,hc)

        if i >j:
            #assert len(free_param) == len(hc) + 1
            #assert len(hc) == len(free_param) - 1
            cpparams['freei'] = (np.min(ranges[freei]), np.max(ranges[freei]))
            cpparams['freej'] = (np.min(ranges[freej]), np.max(ranges[freej]))
            free_params = [freei, freej]
            gr = run_simple_grid(10, tests, ranges, free_params, hold_constant = hc)
            #gr = run_grid(10,tests,, hold_constant = hc, mp_in = params)
            fp = list(copy.copy(free_param))
            #ax0[i,j],img = plot_surface(gr,ax0[i,j],fp,hc,imshow=True)
            ax0[i,j],img = plot_surface(gr,ax0[i,j],fp,hc,imshow=True)

            #plt.colorbar(img, ax = ax0[i,j])
            #ax1[i,j] = plot_surface(gr,ax1[i,j],fp,imshow=False)

        if i < j:
            free_param = list(copy.copy(list(free_param)))
            #if len(free_param) == 2:

            ax0[i,j] = plot_scatter(history,ax0[i,j],free_param,hc)
                #ax0[i,j] = ps(fig0,ax1[i,j],freei,freej,history)

            cpparams['freei'] = (np.min(ranges[freei]), np.max(ranges[freei]))
            cpparams['freej'] = (np.min(ranges[freej]), np.max(ranges[freej]))
            gr = hof

        limits_used = (us[str(freei)],us[str(freej)])
        scores = [ g.dtc.get_ss() for g in gr ]
        params_ = [ g.dtc.attrs for g in gr ]

        # To Pandas:
        # https://stackoverflow.com/questions/28056171/how-to-build-and-fill-pandas-dataframe-from-for-loop#28058264
        temp.append({'i':i,'j':j,'free_param':free_param,'hold_constant':hc,'param_boundaries':cpparams,'scores':scores,'params':params_,'ga_used':limits_used,'grid':gr})
        #print(temp)
        #intermediate = pd.DataFrame(temp)blah
        with open('intermediate.p','wb') as f:
            pickle.dump(temp,f)

    #df = pd.DataFrame(temp)
    plt.savefig(str('cross_section_and_surfaces.png'))
    return temp


# http://www.physics.usyd.edu.au/teach_res/mp/mscripts/
# ns_izh002.m
import collections
from collections import OrderedDict

# Fast spiking cannot be reproduced as it requires modifications to the standard Izhi equation,
# which are expressed in this mod file.
# https://github.com/OpenSourceBrain/IzhikevichModel/blob/master/NEURON/izhi2007b.mod

reduced2007 = collections.OrderedDict([
  #              C    k     vr  vt vpeak   a      b   c    d  celltype
  ('RS',        (100, 0.7,  -60, -40, 35, 0.03,   -2, -50,  100,  1)),
  ('IB',        (150, 1.2,  -75, -45, 50, 0.01,   5, -56,  130,   2)),
  ('LTS',       (100, 1.0,  -56, -42, 40, 0.03,   8, -53,   20,   4)),
  ('TC',        (200, 1.6,  -60, -50, 35, 0.01,  15, -60,   10,   6)),
  ('TC_burst',  (200, 1.6,  -60, -50, 35, 0.01,  15, -60,   10,   6)),
  ('RTN',       (40,  0.25, -65, -45,  0, 0.015, 10, -55,   50,   7)),
  ('RTN_burst', (40,  0.25, -65, -45,  0, 0.015, 10, -55,   50,   7))])

import numpy as np
reduced_dict = OrderedDict([(k,[]) for k in ['C','k','vr','vt','vPeak','a','b','c','d']])

#OrderedDict
for i,k in enumerate(reduced_dict.keys()):
    for v in reduced2007.values():
        reduced_dict[k].append(v[i])

explore_param = {k:(np.min(v),np.max(v)) for k,v in reduced_dict.items()}

opt_keys = [str('vr'),str('a'),str('b')]
nparams = len(opt_keys)


##
# find an optima copy and paste reverse search here.
##

#IB = mparams[param_dict['IB']]
RS = {}
IB = {}
TC = {}
CH = {}
RTN_burst = {}
cells = OrderedDict([(k,[]) for k in ['RS','IB','CH','LTS','FS','TC','TC_burst','RTN','RTN_busrt']])
reduced_cells = OrderedDict([(k,[]) for k in ['RS','IB','LTS','TC','TC_burst']])

for index,key in enumerate(reduced_cells.keys()):
    reduced_cells[key] = {}
    for k,v in reduced_dict.items():
        reduced_cells[key][k] = v[index]

print(reduced_cells)
cells = reduced_cells
from neuronunit.optimization import optimization_management as om
free_params = ['a','b']#,'vr','k'] # this can only be odd numbers.
#2**3
hc = {}
for k,v in cells['TC'].items():
    if k not in free_params:
        hc[k] = v
#print(hc)
#TC_tests = pickle.load(open('thalamo_cortical_tests.p','rb'))
                #run_ga(model_params, max_ngen, test, free_params = None, hc = None)

#ga_out, DO = om.run_ga(explore_param,10,TC_tests,free_params=free_params,hc = hc, NSGA = False, MU = 10)
'''
try:
    #assert 1==2
    ga_out_nsga = pickle.load(open('contents.p','rb'))
except:

    ga_out_nsga, _ = om.run_ga(explore_param,25,TC_tests,free_params=free_params,hc = hc, NSGA = True)
    pickle.dump(ga_out_nsga, open('contents.p','wb'))
'''

hof = ga_out_nsga['pf']
history = ga_out_nsga['history']


attr_keys = list(hof[0].dtc.attrs.keys())
us = {} # GA utilized_space
for key in attr_keys:
    temp = [ v.dtc.attrs[key] for k,v in history.genealogy_history.items() ]
    us[key] = ( np.min(temp), np.max(temp))




if 1==2:
    with open('surfaces.p','rb') as f:
        temp = pickle.load(f)
    from neuronunit.plottools import plot_surface as ps
    #get_justas_plot(history)

    plt.clf()
    dim = len(hof[0].dtc.attrs.keys())
    fig0,ax0 = plt.subplots(dim,dim,figsize=(10,10))
    list_axis = []
    for t in temp:
        fp = t['free_param']
        i = t['i']
        j = t['j']
        scores = t['scores']
        params = t['params']
        gr = t['grid']
        hc = t['hold_constant']
        #print(t.keys())
        if i==j:
            #ax0[i,j] = plot_surface(gr,ax0[i,j],fp,imshow=False)
            #ax0[i,j] = plot_line_ss(gr,ax0[i,j],fp,hof)
            ax0[i,j] = plot_line_ss(gr,ax0[i,j],fp,hof,hc)

        if i < j:
            #ax0[i,j] ,plot_axis = ps(fig0,ax0[i,j],fp[0],fp[1],history)
            ax0[i,j] = plot_scatter(history,ax0[i,j],fp,hc)

        if i > j:

            ax0[i,j],img = plot_surface(gr,ax0[i,j],fp,hc,imshow=True)
            list_axis.append(ax0[i,j])

            #plt.colorbar(img)#,ax = ax0[i,j])

    plt.savefig(str('cross_section_and_surfaces_new.png'))

else:

    temp = grids(hof,TC_tests,explore_param,us,history)
    with open('surfaces.p','wb') as f:
        pickle.dump(temp,f)


def get_justas_plot(history):

    # try:
    import plotly.plotly as py
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot#, iplot
    import plotly.graph_objs as go
    import cufflinks as cf
    cf.go_offline()
    gr = [ v for v in history.genealogy_history.values() ]
    gr = [ g for g in gr if type(g.dtc) is not type(None) ]
    gr = [ g for g in gr if type(g.dtc.scores) is not type(None) ]
    keys = list(gr[0].dtc.attrs.keys())
    xx = np.array([ p.dtc.attrs[str(keys[0])] for p in gr ])
    yy = np.array([ p.dtc.attrs[str(keys[1])] for p in gr ])
    zz = np.array([ p.dtc.attrs[str(keys[2])] for p in gr ])
    ee = np.array([ np.sum(list(p.dtc.scores.values())) for p in gr ])
    #pdb.set_trace()
    # z_data = np.array((xx,yy,zz,ee))
    list_of_dicts = []
    for x,y,z,e in zip(list(xx),list(yy),list(zz),list(ee)):
        list_of_dicts.append({ keys[0]:x,keys[1]:y,keys[2]:z,str('error'):e})

    z_data = pd.DataFrame(list_of_dicts)
    data = [
            go.Surface(
                        z=z_data.as_matrix()
                    )
        ]



    layout = go.Layout(
            width=1000,
            height=1000,
            autosize=False,
            title='Sciunit Errors',
            scene=dict(
                xaxis=dict(
                    title=str(keys[0]),

                    #gridcolor='rgb(255, 255, 255)',
                    #zerolinecolor='rgb(255, 255, 255)',
                    #showbackground=True,
                    #backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    title=str(keys[1]),

                    #gridcolor='rgb(255, 255, 255)',
                    #zerolinecolor='rgb(255, 255, 255)',
                    #showbackground=True,
                    #backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    title=str(keys[2]),

                    #gridcolor='rgb(255, 255, 255)',
                    #zerolinecolor='rgb(255, 255, 255)',
                    #showbackground=True,
                    #backgroundcolor='rgb(230, 230,230)'
                ),
                aspectratio = dict( x=1, y=1, z=0.7 ),
                aspectmode = 'manual'
            ),margin=dict(
                l=65,
                r=50,
                b=65,
                t=90
            )
        )

    fig = go.Figure(data=data, layout=layout)#,xTitle=str(keys[0]),yTitle=str(keys[1]),title='SciUnitOptimization')
    plot(fig, filename='sciunit-score-3d-surface.html')



sys.exit()
