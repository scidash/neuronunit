###
# GA parameters:
##
NGEN = 2
MU = 4; NGEN = 3; CXPB = 0.9
# about 8, models will be made, excluding rheobase search.

##
# Grid search parameters:
# only 5 models, will be made excluding rheobase search
##
npoints = 5
nparams = 1



import ipyparallel as ipp
rc = ipp.Client(profile='default')
from ipyparallel import depend, require, dependent
dview = rc[:]

from neuronunit.optimization.nsga_object import NSGA
from neuronunit.optimization import exhaustive_search as es
###
NSGAO = NSGA()
NSGAO.setnparams(nparams=nparams)
invalid_dtc, pop, logbook, fitnesses = NSGAO.main(MU, NGEN)

#uncomment to facilitate checkpointing.
import pickle
with open('ga_dump.p','wb') as handle:
   pickle.dump([invalid_dtc, pop, logbook, fitnesses],handle)
ga_lists = pickle.load(open('ga_dump.p','rb'))
#invalid_dtc, pop, logbook, fitnesses = lists[0],lists[1],lists[2], lists[3]

dtcpopg = es.run_grid(npoints,nparams)

#uncomment to facilitate checkpointing.
import pickle
with open('grid_dump.p','wb') as handle:
   pickle.dump(dtcpopg,handle)
grid_lists = pickle.load(open('grid_dump.p','rb'))
#invalid_dtc, pop, logbook, fitnesses = lists[0],lists[1],lists[2], lists[3]

def min_find(dtcpop):
    # This function searches virtual model data containers to find the values with the best scores.

    from numpy import sqrt, mean, square
    import numpy as np
    sovg = []
    for i in dtcpop:
        rt = 0 # running total
        #for values in i.scores.values():
        rt = sqrt(mean(square(list(i.scores.values()))))
        sovg.append(rt)
    dtc = invalid_dtc[np.where(sovg==np.min(sovg))[0][0]]
    return dtc
def min_max(dtcpop):
    # This function searches virtual model data containers to find the values with the worst scores.

    from numpy import sqrt, mean, square
    import numpy as np
    sovg = []
    for i in dtcpop:
        rt = 0 # running total
        #for values in i.scores.values():
        rt = sqrt(mean(square(list(i.scores.values()))))
        sovg.append(rt)
    dtc = invalid_dtc[np.where(sovg==np.max(sovg))[0][0]]
    return dtc

minimaga = min_find(invalid_dtc)
minimagr = min_find(dtcpopg)
maximagr = min_max(dtcpopg)
# quantize distance between minimimum error and maximum error.
quantize_distance = list(np.linspace(minimagr,maximagr,10))
# check that the nsga error is in the bottom 1/5th of the entire error range.
print('Report: ')

print(bool(minimaga < quantize_distance[1]))
print(' the nsga error is in the bottom 1/5th of the entire error range',minimaga,quantize_distance[1])

print('maximum error:', maximagr)

# This function reports on the deltas brute force obtained versus the GA found attributes.
from neuronunit.optimization import model_parameters as modelp
mp = modelp.model_params
for k,v in minimagr.attrs.items():
    #hvgrid = np.linspace(np.min(mp[k]),np.max(mp[k]),10)
    dimension_length = np.max(mp[k]) - np.min(mp[k])
    solution_distance_in_1D = np.abs(minimaga.attrs[k])-np.abs(float(v))
    relative_distance = dimension_length/solution_distance_in_1D
    print('the difference between brute force candidates model parameters and the GA\'s model parameters:')
    print(float(minimaga.attrs[k])-float(v),minimaga.attrs[k],v,k)
    print('the relative distance scaled by the length of the parameter dimension of interest:')
    print(relative_distance)

print('the difference between the bf error and the GA\'s error:')
print('grid search:')
from numpy import square, mean, sqrt
rmsg=sqrt(mean(square(list(minimagr.scores.values()))))
print(rmsg)
print('ga:')
rmsga=sqrt(mean(square(list(minimaga.scores.values()))))
print(rmsga)

# Two things to find:
# How close togethor are the model parameters in parameter space (hyper volume), relative to the dimensions of the HV?
# ie get the euclidian distance between the two sets of model parameters.

#
#
#

exit()
#quit()





#print(scores_exh)
