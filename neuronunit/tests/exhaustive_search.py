

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




    #(ampl,attrs,host_name,host_number)=pickle.load(open('big_model_evaulated.pickle','rb'))
def run_single_model(attrs,rhvalue):
    import sciunit.scores as scores
    import quantities as qt
    print(attrs)
    print(rhvalue)
    model.rheobase = rhvalue#*qt.pA
    model.update_run_params(attrs)
    params = {}
    #params['injected_square_current']['delay'] = DELAY
    #params['injected_square_current']['duration'] = DURATION
    gs.get_neab.suite.tests[0].params['injected_square_current']['amplitude'] = rhvalue
    params = gs.get_neab.suite.tests[0].params
    #model.params['injected_square_current']['amplitude'] = rhvalue
    model.inject_square_current(params)
    print(model.results.keys())
    return (model.results['vm'],model.results['t'])


def plot_results(ground_error):

    sum_error = [ i[0] for i in ground_error  ]
    component_error = [ i[1] for i in ground_error  ]
    attrs = [ i[2] for i in ground_error ]
    rhvalue = [ i[3] for i in ground_error ]
    min_value = min([float(s) for s in sum_error])
    try:
        assert min([float(s) for s in sum_error]) == np.min(np.array(sum_error))
        assert sum_error[np.where( np.array(sum_error) == np.min(np.array(sum_error) ))[0][0]] == min([float(s) for s in sum_error])

    except:
        '{} no uique minimum error found'.format()
    min_ind = np.where( np.array(sum_error) == np.min(np.array(sum_error) ))[0][0]

    #get the rheobase current injection for this run
    (vm,time_vector) = run_single_model(attrs[min_ind],rhvalue[min_ind])

    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.9*float(min_ind/len(sum_error)), 0.6, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax1.plot([i for i in range(0,len(sum_error))], sum_error, color='red')
    ax1.plot(min_ind, np.min(np.array(sum_error)),'*',color='blue')
    ax1.set_yscale('log')
    plt.xlabel('params'+str(attrs[min_ind]))
    plt.ylabel('error')

    ax2.plot(time_vector, vm,  color='green',label=str(attrs[min_ind]))
    legend = ax1.legend(loc='lower center', shadow=False , prop={'size':6})

    fig.savefig('inset_figure.png')
    plt.clf()
    plot.plot(min_ind, np.min(np.array(sum_error)),'*',color='blue')
    plt.plot([i for i in range(0,len(sum_error))], sum_error,label='rectified sum of errors')#, color='red')
    plt.plot([i for i in range(0,len(sum_error))],[i[0] for i in component_error],label='RheobaseTest')#, color='red')
    plt.plot([i for i in range(0,len(sum_error))],[i[1] for i in component_error],label='InputResistanceTest')#, color='red')
    plt.plot([i for i in range(0,len(sum_error))],[i[2] for i in component_error],label='TimeConstantTest')#, color='red')
    plt.plot([i for i in range(0,len(sum_error))],[i[3] for i in component_error],label='CapacitanceTest')#, color='red')
    plt.plot([i for i in range(0,len(sum_error))],[i[4] for i in component_error],label='RestingPotentialTest')#, color='red')
    plt.plot([i for i in range(0,len(sum_error))],[i[5] for i in component_error],label='InjectedCurrentAPWidthTest')#, color='red')
    plt.plot([i for i in range(0,len(sum_error))],[i[6] for i in component_error],label='InjectedCurrentAPAmplitudeTest')#, color='red')
    for i in component_error:
        try:
            plt.plot([i for i in range(0,len(sum_error))],[i[7] for i in component_error],label='InjectedCurrentAPThresholdTest')#, color='red')
        except:
            '{} AP threshold test had to be abandoned'.format(str('InjectedCurrentAPThresholdTest'))

    legend = plt.legend(loc='upper right', shadow=False, prop={'size':6})
    fig.savefig('informative_error.png')
    return 0

'''
Move all of this to tests
'''
import scoop
from scoop import launcher
#Uncomment the code below to run an exhaustive search.
if __name__ == "__main__":
    model=gs.model
    import pickle
    try:
        ground_truth = pickle.load(open('big_model_list.pickle','rb'))
    except:
        '{} it seems the ground truth data does not yet exist, lets create it now '.format()
        import scoop
        import model_parameters as modelp
        iter_list=[ (i,j,k,l) for i in modelp.model_params['a'] for j in modelp.model_params['b'] for k in modelp.model_params['vr'] for l in modelp.model_params['vpeak'] ]
        list_of_models = list(futures.map(gs.model2map,iter_list))
        rhstorage=list(futures.map(gs.evaluate,list_of_models))
        rhstorage = list(filter(lambda item: type(item) is not type(None), rhstorage))
        rhstorage = list(filter(lambda item: type(item.rheobase) is not type(None), rhstorage))
        #rhstorage = list(filter(lambda item: item.rheobase > 0.0, rhstorage))
        data_for_iter = zip(rhstorage,list(futures.map( lambda item: item.rheobase,rhstorage )))
        import pickle
        with open('big_model_list.pickle', 'wb') as handle:
            pickle.dump(data_for_iter, handle)

    try:
        ground_error = pickle.load(open('big_model_evaulated.pickle','rb'))#rcm
    except:
        '{} it seems the error truth data does not yet exist, lets create it now '.format()
        ground_error = list(futures.map(gs.func2map, ground_truth))
        pickle.dump(ground_error,open('big_model_evaulated.pickle','wb'))

    _ = plot_results(ground_error)
    #print(ground_error)
    '''
    sum_error = [ i[0] for i in ground_error  ]
    component_error = [ i[1] for i in ground_error  ]
    attrs = [ i[2] for i in ground_error ]
    rhvalue = [ i[3] for i in ground_error ]
    print(attrs)
    print(len(attrs))
    print(np.shape(attrs))
    print(type(attrs))
    #vm = [ i[6] for i in ground_error  ]
    #time_vector = [ i[3] for i in ground_error  ]

    #print(sum_error, 'error')
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
