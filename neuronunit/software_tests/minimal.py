


import mpi4py
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

pop = [ i for i in range(RANK, 10, SIZE) ]

pop2 = COMM.gather(pop, root=0)
if RANK == 0:
    pop=[]

    for p in pop2:
        pop.extend(p)

    print(pop)
    
else:
   pop=None    
pop = COMM.bcast(pop, root=0)
print(pop)
    #pdb.set_trace()
