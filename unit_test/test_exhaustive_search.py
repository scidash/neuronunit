###
# GA parameters
#MU = 12,
NGEN = 3
CXPB = 0.9
MU = 6; NGEN = 4; CXPB = 0.9
nparams = 3
npoints = 3
NSGA = True

from neuronunit.optimization import exhaustive_search as es

'''
###
# less than 10 params currently unsupported.
difference_progress, fitnesses, pf, logbook, pop, dtcpop, stats, scores_nsgaW = nsgap.main(MU=MU, \
                                                                             NGEN=NGEN, \
                                                                             CXPB=CXPB,\
                                                                             nparams=10)
import pickle
with open('complete_dump.p','wb') as handle:
   pickle.dump([difference_progress, fitnesses, pf, logbook, pop, dtcpop, stats, scores_nsga],handle)
lists = pickle.load(open('complete_dump.p','rb'))
'''

dtcpop = es.run_grid(npoints,nparams)
for d in dtcpop:
    print(d.scores)
print(dtcpop)
print('job completed, gracefuly quiting')
exit()