

import os
os.system('ipcluster start -n 8 --engines=MPIEngineSetLauncher --profile=chase --debug &')

#os.system('sleep 15 &')
import ipyparallel as ipp
rc = ipp.Client(profile='chase');
rc[:].use_cloudpickle()

print('hello from before cpu ');
print(rc.ids)
dview = rc[:]
serial_result = list(map(lambda x:x**10, range(32)))
parallel_result = list(dview.map_sync(lambda x: x**10, range(32)))
print(serial_result)
print(parallel_result, 'parallel_reult')
assert serial_result == parallel_result

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
from scoop import futures
import scoop

import get_neab

import quantities as qt
import os
import os.path
from scoop import utils

import sciunit.scores as scores


import grid_search as gs
model=gs.model



#Uncomment the code below to run an exhaustive search.
if __name__ == "__main__":

    import pdb
    import scoop
    import model_parameters as modelp
    #iter_list=[ (i,j,k,l) for i in modelp.model_params['a'] for j in modelp.model_params['b'] for k in modelp.model_params['vr'] for l in modelp.model_params['vpeak'] ]
    iter_list=[ i for i in modelp.model_params['a'] ]
    #iter_list=iter_list[0:1]
    import grid_search as gs

    mean_vm=gs.VirtualModel()
    guess_attrs = modelp.guess_attrs[0]
    #paramslist=['a','b','vr','vpeak']
    paramslist=['a']#,'b','vr','vpeak']

    #param
    value=guess_attrs
    print(value)
    x=paramslist[0]
    model.name = str(model.name)+' '+str(x)+str(value)
    #if i==0:
    attrs = {'//izhikevich2007Cell':{x:value }}
    #else:
    attrs['//izhikevich2007Cell'][x]=value
    mean_vm.attrs=attrs

    steps = np.linspace(50,150,7.0)
    steps_current = [ i*pq.pA for i in steps ]
    model.re_init(mean_vm.attrs)



    rh_param=(False,steps_current)

    pre_rh_value=gs.searcher(gs.check_current,rh_param,mean_vm)
    rh_value=pre_rh_value.rheobase
    list_of_models=list(futures.map(gs.model2map,iter_list))
    print('gets here c')

    for li in list_of_models:
        print(li.rheobase, li.attrs)
    from itertools import repeat
    rhstorage=list(futures.map(gs.evaluate,list_of_models,repeat(rh_value)))
    print('gets here b')

    for x in rhstorage:
        x=x.rheobase
        if x==False:
            vm_spot=VirtualModel()
            steps = np.linspace(40,250,7.0)
            steps_current = [ i*pq.pA for i in steps ]
            rh_param=(False,steps_current)
            rh_value=gs.searcher(gs.check_current,rh_param,vm_spot)

    rhstorage2 = [i.rheobase for i in rhstorage]
    rhstorage=rhstorage2
    iter_ = zip(list_of_models,rhstorage)
    print('gets here a')
    score_matrixt=list(futures.map(gs.func2map,iter_))#list_of_models,rhstorage))
    print(score_matrixt)
    import pdb
    pdb.set_trace()
    import pickle
    with open('score_matrix.pickle', 'wb') as handle:
        pickle.dump(score_matrixt, handle)

    score_matrix=[]
    attrs=[]
    score_typev=[]
    #below score is just the floats associated with RatioScore and Z-scores.
    for score,attr,_ in score_matrixt:
        for i in score:
            for j in i:
                if j==None:
                    j=10.0
        score_matrix.append(score)
        attrs.append(attr)
        print(attr,score)

    score_matrix=np.array(score_matrix)
    for i in score_matrix:
        for j in i:
            if type(j)==None:
                j=10.0
            if j==None:
                j=10.0


    with open('score_matrix.pickle', 'rb') as handle:
        matrix=pickle.load(handle)


    matrix3=[]
    for x,y, rheobase in matrix:
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



    def build_single(attrs):
        #This method is only used to check singlular sets of hard coded parameters.]
        #This medthod is probably only useful for diagnostic purposes.
        import sciunit.scores as scores
        import quantities as qt
        vm = VirtuaModel()
        rh_value=searcher(check,rh_param,vms)
        get_neab.suite.tests[0].prediction={}
        get_neab.suite.tests[0].prediction['value']=rh_value*qt.pA
        score = get_neab.suite.judge(model)#passing in model, changes model

    build_single(attrs)

#    return model
    #else:
    #    return 10.0



    import pdb; pdb.set_trace()
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

                #print(j)
