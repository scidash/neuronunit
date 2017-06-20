
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
    
