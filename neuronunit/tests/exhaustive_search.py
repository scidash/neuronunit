

#import os
#os.system('ipcluster start -n 8 --engines=MPIEngineSetLauncher --profile=chase #--debug &')

#os.system('sleep 15 &')
#import ipyparallel as ipp
#rc = ipp.Client(profile='chase');
#rc[:].use_cloudpickle()

#print('hello from before cpu ');
#print(rc.ids)
#dview = rc[:]
#serial_result = list(map(lambda x:x**10, range(32)))
#parallel_result = list(dview.map_sync(lambda x: x**10, range(32)))
#print(serial_result)
#print(parallel_result, 'parallel_reult')
#assert serial_result == parallel_result

import time
import pdb
import array
import random


"""
Scoop can only operate on variables classes and methods at top level 0
This means something with no indentation, no nesting,
and no virtual nesting (like function decorators etc)
anything that starts at indentation level 0 is okay.

However the case may be different for functions. Functions may be imported from modules.
I am unsure if it is only the case that functions can be imported from a module, if they are not bound to
any particular class in that module.

Code from the DEAP framework, available at:
https://code.google.com/p/deap/source/browse/examples/ga/onemax_short.py
from scoop import futures
"""

import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from deap import algorithms
from deap import base
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
import get_neab

import quantities as qt
import os
import os.path
from scoop import utils, futures

import sciunit.scores as scores

global gs
import grid_search as gs

model=gs.model
'''
(ampl,attrs,host_name,host_number)=pickle.load(open('test_current_failed_attrs.pickle','rb'))
def build_single(attrs):
    #This method is only used to check singlular sets of hard coded parameters.]
    #This medthod is probably only useful for diagnostic purposes.
    import sciunit.scores as scores
    import quantities as qt
    vm = VirtuaModel()
    rh_value=searcher(rh_param,vms)
    get_neab.suite.tests[0].prediction={}
    get_neab.suite.tests[0].prediction['value']=rh_value*qt.pA
    score = get_neab.suite.judge(model)#passing in model, changes model

build_single(attrs)
'''
import scoop
from scoop import launcher
#Uncomment the code below to run an exhaustive search.
if __name__ == "__main__":
    import scoop
    import model_parameters as modelp
    iter_list=[ (i,j,k,l) for i in modelp.model_params['a'] for j in modelp.model_params['b'] for k in modelp.model_params['vr'] for l in modelp.model_params['vpeak'] ]
    list_of_models = list(futures.map(gs.model2map,iter_list))
    for li in list_of_models:
        print(li.rheobase, li.attrs)
    from itertools import repeat
    rhstorage=list(futures.map(gs.evaluate,list_of_models))#,repeat(rh_value)))
    iter_=[]
    rhstorage = list(filter(lambda item: type(item) is not type(None), rhstorage))

    #rhstorage = list(filter(lambda item: (type(item) is not 'NoneType') and (type(item) is not type(None)), rhstorage))
    rhstorage = list(filter(lambda item: type(item.rheobase) is not type(None), rhstorage))
    rhstorage = list(filter(lambda item: item.rheobase > 0.0, rhstorage))

    iter_=zip(rhstorage,list(futures.map( lambda item: item.rheobase,rhstorage )))
    #iter_=zip([ i.rheobase for i in rhstorage ],[ j for j in iter_ ])

    import pickle
    with open('big_model_list.pickle', 'wb') as handle:
        pickle.dump(iter_, handle)


    sm = list(futures.map(gs.func2map, iter_))

    with open('score_matrix.pickle', 'wb') as handle:
        pickle.dump(sm, handle)

    sum_error = [ i[0] for i in sm ]
    attrs = [ str(k)+str(v) for i in sm for k,v in i[1].items() ]
    vm = [ i[6] for i in sm ]
    time_vector = [ i[3] for i in sm ]
    component_error = [ i[5] for i in sm ]

    print(sum_error, 'error')
    min_value = min([float(s) for s in sum_error])
    assert min([float(s) for s in sum_error]) == np.min(np.array(sum_error))
    print(min([float(s) for s in sum_error]), np.min(np.array(sum_error)))
    min_ind = np.where( np.array(sum_error) == np.min(np.array(sum_error) ))[0][0]
    assert sum_error[np.where( np.array(sum_error) == np.min(np.array(sum_error) ))[0][0]] == min([float(s) for s in sum_error])
    print(min_ind,sum_error[min_ind],sum_error)
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.9*float(min_ind/len(sum_error)), 0.6, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])



    ax1.plot([i for i in range(0,len(sum_error))], sum_error, color='red')
    ax1.set_yscale('log')
    plt.xlabel('params'+str(attrs[min_ind]))
    plt.ylabel('error')

    ax2.plot(time_vector[min_ind], vm[min_ind],  color='green',label=str(attrs[min_ind]))
    legend = ax1.legend(loc='lower center', shadow=True)

    fig.savefig('inset_figure')
    '''
    for i in sm:
        for j in i:
            if type(j)==None:
                j=10.0
            if j==None:
                j=10.0


    with open('score_matrix.pickle', 'rb') as handle:
        matrix=pickle.load(handle)

    sm = np.array(sm)

    matrix3=[]
    for x,y, rheobase in sm:
        for i in x:
            matrix2=[]
            for j in i:
                if j==None:
                    j=10.0
                matrix2.append(j)
            matrix3.append(matrix2)
    storagei = [ np.sum(i) for i in matrix3 ]
    storagesmin=np.where(storagei==np.min(storagei))
    storagesmax=np.where(storagei==np.max(storagei))

    score0,attrs0,rheobase = matrix[storagesmin[0]]
    score1,attrs1,rheobase = matrix[storagesmin[1]]





    #import pdb; pdb.set_trace()
    import matplotlib as plt
    for i,s in enumerate(score_typev[np.shape(storagesmin)[0]]):
        #.related_data['vm']
        plt.plot(plot_vm())
        plt.savefig('s'+str(i)+'.png')
    #since there are non unique maximum and minimum values, just take the first ones of each.
    tuplepickle=(score_matrix[np.shape(storagesmin)[0]],score_matrix[np.shape(storagesmax)[0]],attrs[np.shape(storagesmax)[0]])
    with open('minumum_and_maximum_values.pickle', 'wb') as handle:
        pickle.dump(tuplepickle,handle)
    with open('score_matri.pickle', 'rb') as handle:
        opt_values=pickle.load(handle)
        print('minumum value')
        print(opt_values)
    '''
