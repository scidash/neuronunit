###
# GA parameters
#MU = 12,
NGEN = 3
CXPB = 0.9
MU = 6; NGEN = 4; CXPB = 0.9
nparam = 1

from neuronunit.optimization import nsga_parallel as nsgap

###
difference_progress, fitnesses, pf, logbook, pop, dtcpop, stats, scores_nsga = nsgap.main(MU=MU, \
                                                                             NGEN=NGEN, \
                                                                             CXPB=CXPB,\
                                                                             nparams=10)
import pickle
with open('complete_dump.p','wb') as handle:
   pickle.dump([difference_progress, fitnesses, pf, logbook, pop, dtcpop, stats, scores],handlese)

lists = pickle.load(open('complete_dump.p','rb'))
#dtcoffspring2,history2,logbook2 = lists[0],lists[1],lists[2]
from neuronunit.optimization import exhaustive_search as es
npoints = 3
nparams = 10
scores_exh, dtcpop = es.run_grid(npoints,nparams)
