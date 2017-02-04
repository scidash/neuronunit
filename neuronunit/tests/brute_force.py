

import mpi4py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

import get_neab
import numpy as np
import time
#from neuronunit.models import backends
from neuronunit.models.reduced import ReducedModel
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
model=model.load_model()
vr = np.linspace(-75.0,-50.0,10)
a = np.linspace(0.015,0.045,10)
b = np.linspace(-3.5*10E-9,-0.5*10E-9,10)
iter_list=[ (i,j,k) for i in a for j in b for k in vr ]

def func2map(iter_arg):
    attrs={}
    attrs['//izhikevich2007Cell']={}
    param=['a','b','vr']
    i,j,k=iter_list[iter_arg]
    model.name=str(i)+str(j)+str(k)
    attrs['//izhikevich2007Cell']['a']=i
    attrs['//izhikevich2007Cell']['b']=j
    attrs['//izhikevich2007Cell']['vr']=k
    model.update_run_params(attrs)
    score = get_neab.suite.judge(model)#passing in model, changes model
    model.run_number+=1
    RUN_TIMES='{}{}{}'.format('counting simulation run times on models',model.results['run_number'],model.run_number)
    return score

iter_arg = [ i for i in range(RANK, 999, SIZE) ]

bg_bf=time.time()
score_matrix = [func2map(i) for i in iter_arg]

COMM.barrier()
score_matrix2 = COMM.gather(score_matrix, root=0)
if RANK == 0:
    score_matrix=[]
    for p in score_matrix:
        score_matrix.extend(p)
    print(score_matrix)
    end_bf=time.time()
    whole_time=end_bf-bg_bf
    with open('score_matrix.pickle', 'wb') as handle:
        pickle.dump(score_matrix, handle)

    f=open('brute_force_time','w')
    f.write(str(whole_time))
    f.close()
else:
   score_matrix=None
