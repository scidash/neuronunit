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
npoints = 3
NSGA = False

###
# less than 10 params currently unsupported.

if NSGA:
	difference_progress, fitnesses, pf, logbook, pop, dtcpop, stats, scores_nsgaW = nsgap.main(MU=MU, \
                                                                             NGEN=NGEN, \
                                                                             CXPB=CXPB,\
                                                                             nparams=nparams)
	with open('complete_dump.p','wb') as handle:
		pickle.dump([difference_progress, fitnesses, pf, logbook, pop, dtcpop, stats, scores_nsga],handle)

if os.path.isfile('complete_dump.p'):
	lists = pickle.load(open('complete_dump.p','rb'))
#dtcoffspring2,history2,logbook2 = lists[0],lists[1],lists[2]

scores_exh, dtcpop = es.run_grid(npoints,nparams)
