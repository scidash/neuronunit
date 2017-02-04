
'''
import os
os.system('ipcluster start --profile=jovyan --debug &')
os.system('sleep 5')
import ipyparallel as ipp
rc = ipp.Client(profile='jovyan')
print('hello from before cpu ')
print(rc.ids)
#quit()
v = rc.load_balanced_view()
#from scoop import futures
#,'k']#,'C']#,'c','d','v0','k','vt','vpeak']#,'d'

attrs={}
#pdb.set_trace()
attrs['//izhikevich2007Cell']={}
#for  (i,j,k) in iter_stuff:
score_matrix=[]
#iter_stuff=[ (i,j,k) for i in a for j in b for k in vr ]

#pdb.set_trace()
import time
init_start=time.time()
import get_neab

import numpy as np
#import pdb
from neuronunit.models import backends
from neuronunit.models.reduced import ReducedModel
import time
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')


model=model.load_model()
vanila_start=time.time()
model.local_run()

'''

import mpi4py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

import get_neab

import numpy as np
#from neuronunit.models import backends
from neuronunit.models.reduced import ReducedModel
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model=model.load_model()
vr = np.linspace(-75.0,-50.0,10)
a = np.linspace(0.015,0.045,10)
b = np.linspace(-3.5*10E-9,-0.5*10E-9,10)

def func2map(iter_arg):
    print('entered')
    print('entered?',iter_arg)
    attrs={}
    attrs['//izhikevich2007Cell']={}
    score_matrix=[]

    #c = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,10)
    param=['a','b','vr']
    iter_stuff=[ (i,j,k) for i in a for j in b for k in vr ]

    i,j,k=iter_stuff[iter_arg]
    print(i,j,k)
    #values_tuple=(x,i,y,j,z,k)
    model.name=str(i)+str(j)+str(k)
    attrs['//izhikevich2007Cell']['a']=i
    attrs['//izhikevich2007Cell']['b']=j
    attrs['//izhikevich2007Cell']['vr']=k
    model.update_run_params(attrs)
    score = get_neab.suite.judge(model)#passing in model, changes model
    #score_matrix.append(score)
    model.run_number+=1
    RUN_TIMES='{}{}{}'.format('counting simulation run times on models',model.results['run_number'],model.run_number)
    return score
    #individual.results=model.results

#if __name__ == "__main__":
    #results = futures.map(func, data)
#iter_arg=[ i for i in range(0,999)]


iter_arg = [ i for i in range(RANK, 999, SIZE) ]

#score_matrix= map(func2map,iter_arg)
score_matrix=[]
for i in iter_arg:
   print(i)
   score_matrix.append(func2map(i))

COMM.barrier()
score_matrix2 = COMM.gather(score_matrix, root=0)
if RANK == 0:
    score_matrix=[]
    for p in score_matrix:
        score_matrix.extend(p)
    print(score_matrix)

    with open('score_matrix.pickle', 'wb') as handle:
        pickle.dump(score_matrix, handle)
else:
   score_matrix=None
#pop = COMM.bcast(pop, root=0)
#print(pop)
