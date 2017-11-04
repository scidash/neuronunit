import os
import pickle

from neuronunit.optimization import exhaustive_search as es
from neuronunit.optimization import nsga_parallel as nsgap

###
# GA parameters
#MU = 12,
NGEN = 3
CXPB = 0.9
MU = 6; NGEN = 4; CXPB = 0.9
nparams = 2
npoints = 2
NSGA = True

###
# less than 10 params currently unsupported.

#dtcoffspring2,history2,logbook2 = lists[0],lists[1],lists[2]
import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]
from ipyparallel import depend, require, dependent
dtcpop_grid = list(es.run_grid(npoints,nparams))
for d in dtcpop_grid:
    print(d.scores)
#print(dtcpop)
#parallel_method = es.parallel_method
#dtcpop = dview.map_sync(parallel_method,dtcpop)
#dtcpop = list(dtcpop)
#print(dtcpop)

#print(dtcpop,scores_exh)
if NSGA:
	difference_progress, fitnesses, pf, logbook, pop, dtcpop, stats, scores_nsgaW = nsgap.main(MU=MU, \
                                                                             NGEN=NGEN, \
                                                                             CXPB=CXPB,\
                                                                             nparams=nparams)
	with open('complete_dump.p','wb') as handle:
		pickle.dump([difference_progress, fitnesses, pf, logbook, pop, dtcpop, stats, scores_nsga],handle)

if os.path.isfile('complete_dump.p'):
        lists = pickle.load(open('complete_dump.p','rb'))
