
# coding: utf-8
import sys
sys.path[0]='/home/jovyan/mnt/neuronunit'
# In[1]:
#from ../neuronunit import neuronunit

from neuronunit.tests import get_neab
import neuronunit
print(neuronunit.tests.__file__)
from neuronunit.tests import utilities as outils
outils.get_neab = get_neab

from neuronunit.tests import model_parameters as modelp
import numpy as np
model = outils.model

class Score:
    def __init__(self, score):
        self.score = score

class Test:
    def _optimize(self, model,modelp):
        '''
        The implementation of optimization, consisting of implementation details.
        Inputs a model, and model parameter ranges to expore
        Private method for programmer designer.
        Outputs the optimal model, its attributes and the low error it resulted in.
        '''
        from neuronunit.tests import nsga
        gap = nsga.GAparams(model)
        # The number of generations is 2
        gap.NGEN = 2
        # The population of genes is 4
        gap.MU = 4
        gap.BOUND_LOW = [ np.min(i) for i in modelp.model_params.values() ]
        gap.BOUND_UP = [ np.max(i) for i in modelp.model_params.values() ]

        vmpop, pop, invalid_ind, pf = nsga.main(gap.NGEN,gap.MU,model,modelp)
        some_tuples = (vmpop, pop, invalid_ind, pf)
        return pop[0], some_tuples
        #return vmpop, pop, invalid_ind, pf
        
    def _get_optimization_parameters(self, some_tuples):
        vmpop = some_tuples[0]
        # Your specific unpacking of tuples that _optimize returns
        scores = vmpop[0].score
        pop = some_tupes[1]
        errors = -pop[0].fitness
        parameters = vmpop[0].attrs
        #parameters,errors,scores,_ = zip(*some_tuples)
        return parameters,scores

    def optimize(self, model, modelp):
        '''
        The Class users version of optimize
        where details are hidden in _optimizae
        '''

        # Do optimization including repeated calls to judge
        model, some_tuples = self._optimize(model,modelp)
        parameters, scores = self._get_optimization_parameters(some_tuples)
        # Maybe rebuild the original model
        # (i.e. restore the true class from the virtual version)

        # Your code keeps parameter sets and associated scores
        # All the organizing stuff
        
        
        # this a way of looking at solved model parameters, ie candidate solutions from 
        # the pareto front.

        models = [model.__class__(p) for p in parameters]

        # Make a ScoreArray (which is basically a pandas dataframe)
        path = sciunit.ScoreArray(models,scores=scores)
        return model, path, some_tuples

    
t = Test()


model,path,some_tuples = t.optimize(model,modelp)   


# In[ ]:


#gap.NGEN
#print(vmpop,pop,invalid_ind,pf)
import pickle
import pandas as pd


try:
    ground_error = pickle.load(open('big_model_evaulated.pickle','rb'))
except:
    '{0} it seems the error truth data does not yet exist, lets create it now '.format(str(False))
    ground_error = list(futures.map(util.func2map, ground_truth))
    pickle.dump(ground_error,open('big_model_evaulated.pickle','wb'))
ground_error_nsga=list(zip(vmpop,pop,invalid_ind))
pickle.dump(ground_error_nsga,open('nsga_evaulated.pickle','wb'))

#Get the differences between values obtained via brute force, and those obtained otherwise:
sum_errors = [ i[0] for i in ground_error ]
composite_errors = [ i[1] for i in ground_error ]
attrs = [ i[2] for i in ground_error ]
rheobase = [ i[3] for i in ground_error ]

indexs = [i for i,j in enumerate(sum_errors) if j==np.min(sum_errors) ][0]
indexc = [i for i,j in enumerate(composite_errors) if j==np.min(composite_errors) ][0]
#assert indexs == indexc
vmpop = some_tupeles[0]
df_0 = pd.DataFrame([ (k,v,vmpop[0].attrs[k],float(v)-float(vmpop[0].attrs[k])) for k,v in ground_error[indexc][2].items() ])
df_1 = pd.DataFrame([ (k,v,vmpop[1].attrs[k],float(v)-float(vmpop[1].attrs[k])) for k,v in ground_error[indexc][2].items() ])


    


# In[ ]:


path


# In[ ]:




df_0


# In[ ]:


df_1


# In[ ]:




