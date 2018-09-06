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
import sys
        


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

def plot_scatter(history,ax,keys):
    pop = [ v for v in history.genealogy_history.values() ]   
    z = np.array([ np.sum(list(p.dtc.scores.values())) for p in pop ])
    x = np.array([ p.dtc.attrs[str(keys[0])] for p in pop ])
    if len(keys) != 1:
        y = np.array([ p.dtc.attrs[str(keys[1])] for p in pop ])
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

def plot_line_ss(gr,ax,key,hof):
    ax.cla()
    ax.set_title(' {0} vs  score'.format(key[0]))
    z = np.array([ np.sum(list(p.dtc.scores.values())) for p in gr ])
    x = np.array([ p.dtc.attrs[key[0]] for p in gr ])
    y = hof[0].dtc.attrs[key[0]]
    i = hof[0].dtc.get_ss()
    ax.scatter(x,z)
    ax.scatter(y,i)
    ax.plot(x,z)
    ax.set_xlim(np.min(x),np.max(x))
    ax.set_ylim(np.min(z),np.max(z))
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
                                                                                                                                                                                                                  
from neuronunit.plottools import plot_surface as ps
from collections import OrderedDict                                   

def grids(hof,tests,params,us,history):
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
    
    dim = len(hof[0].dtc.attrs.keys())
    flat_iter = iter([(i,freei,j,freej) for i,freei in enumerate(hof[0].dtc.attrs.keys()) for j,freej in enumerate(hof[0].dtc.attrs.keys())])
    #matrix = [[[0 for z in range(dim)] for x in range(dim)] for y in range(dim)]
    plt.clf()
    fig0,ax0 = plt.subplots(dim,dim,figsize=(10,10))
    fig1,ax1 = plt.subplots(dim,dim,figsize=(10,10))
    
    cnt = 0
    temp = []
    loc_key = {}
    
    #free_param = 
    for k,v in hof[0].dtc.attrs.items():
        loc_key[k] = hof[0].dtc.attrs[k]
        params[k] = ( loc_key[k]- 3*np.abs(loc_key[k]), loc_key[k]+2*np.abs(loc_key[k]) )   
    
    for i,freei,j,freej in flat_iter:
        free_param = [freei,freej]
        free_param_set = set(free_param) # construct a small-set out of the indexed keys 2. If both keys are
        # are the same, this set will only contain one index
        bs = set(hof[0].dtc.attrs.keys()) # construct a full set out of all of the keys available, including ones not indexed here.
        diff = bs.difference(free_param_set) # diff is simply the key that is not indexed.
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
            #ax0[i,j] = plot_line_ss(gr,ax[i,j],fp,hof)
            ax1[i,j] = plot_line_ss(gr,ax[i,j],fp,hof)
        if i >j:
            assert len(free_param) == len(hc) + 1
            assert len(hc) == len(free_param) - 1
            cpparams['freei'] = (np.min(params[freei]), np.max(params[freei]))
            cpparams['freej'] = (np.min(params[freej]), np.max(params[freej]))

            gr = run_grid(10,tests,provided_keys = list((freei,freej)), hold_constant = hc, mp_in = params)
            fp = list(copy.copy(free_param))
            ax0[i,j] = plot_surface(gr,ax[i,j],fp,imshow=False)
            ax1[i,j] = plot_surface(gr,ax[i,j],fp,imshow=False)

        if i < j:
            free_param = list(copy.copy(list(free_param)))
            if len(free_param) == 2:
                ax0[i,j] = plot_scatter(history,ax[i,j],free_param)
                ax1[i,j] = ps(fig1,ax[i,j],freei,freej,history)   

            cpparams['freei'] = (np.min(params[freei]), np.max(params[freei]))
            cpparams['freej'] = (np.min(params[freej]), np.max(params[freej]))
            gr = hof

        limits_used = (us[str(freei)],us[str(freej)])
        scores = [ g.dtc.get_ss() for g in gr ]
        params_ = [ g.dtc.attrs for g in gr ]
   
        # To Pandas:
        # https://stackoverflow.com/questions/28056171/how-to-build-and-fill-pandas-dataframe-from-for-loop#28058264
        temp.append({'i':i,'j':j,'free_param':free_param,'hold_constant':hc,'param_boundaries':cpparams,'scores':scores,'params':params_,'ga_used':limits_used})
        print(temp)
        #intermediate = pd.DataFrame(temp)
        with open('intermediate.p','wb') as f:
            pickle.dump(temp,f)
                
    #df = pd.DataFrame(temp)
    plt.savefig(str('cross_section_and_surfaces.png'))
    return temp


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
    mp['b'] = [ -10, 500.0 ]                                                                                      
    mp['vr'] = [ -100.0, -40.0 ]                                                                                     
    mp['a'] = [-10, 2]                                                                                             



# get a genetic algorithm that operates on this new parameter range.
try:
    assert 1==2 
    with open('package.p','rb') as f:
        package = pickle.load(f)

except:

    package, DO = run_ga(mp,12,tests_,provided_keys = opt_keys)
    with open('package.p','wb') as f:
        pickle.dump(package,f)

hof = package['hof']
history = package['history']
# import pdb
# pdb.set_trace()

attr_keys = list(hof[0].dtc.attrs.keys())
us = {} # GA utilized_space
for key in attr_keys:
    temp = [ v.dtc.attrs[key] for k,v in history.genealogy_history.items() ]
    us[key] = ( np.min(temp), np.max(temp))

        
try:
    assert 1==2 
    with open('surfaces.p','rb') as f:
        temp = pickle.load(f)

except:

    temp = grids(hof,tests_,mp,us,history)
    with open('surfaces.p','wb') as f:
        pickle.dump(temp,f)

sys.exit()
        
