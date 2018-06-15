USE_CACHED_GS = False
npoints = 3
nparams = 10
from neuronunit.optimization import get_neab
import os
electro_path = str(os.getcwd())+'/pipe_tests.p'
import pickle
print(electro_path)
#try:
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)

electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
#electro_tests = []
purkinje = { 'nlex_id':'sao471801888'}#'NLXWIKI:sao471801888'} # purkinje
fi_basket = {'nlex_id':'100201'}
pvis_cortex = {'nlex_id':'nifext_50'} # Layer V pyramidal cell
olf_mitral = { 'nlex_id':'nifext_120'}
ca1_pyr = { 'nlex_id':'830368389'}

pipe = [ fi_basket, pvis_cortex, olf_mitral, ca1_pyr, purkinje ]


from neuronunit.optimization import exhaustive_search
from neuronunit.optimization import get_neab
from neuronunit.optimization import optimization_management
from neuronunit.optimization import optimization_management
import pickle
import dask.bag as db
#from deap import base
#toolbox = base.Toolbox()
#pop = toolbox.population(n=len(grid_points))


class WSListIndividual(list):
    """Individual consisting of list with weighted sum field"""
    def __init__(self, *args, **kwargs):
        """Constructor"""
        self.rheobase = None
        super(WSListIndividual, self).__init__(*args, **kwargs)
    
grid_points = exhaustive_search.create_grid(npoints = npoints,nparams = nparams)
pre_pop = [ list(g.values()) for g in grid_points ]
pop = [ WSListIndividual(p) for p in pre_pop ]

tds = [ list(g.keys()) for g in grid_points ]
td = tds[0]
N = 3
cnt = 0
for test, observation in electro_tests:
    pop = optimization_management.update_deap_pop(pop, test, td)
    with open('grid_cell'+str(pipe[cnt])+'.p','wb') as f:
        pickle.dump(pop,f)
        cnt+=1
