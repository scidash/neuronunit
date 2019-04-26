
# coding: utf-8

# Assumptions, the environment for running this notebook was arrived at by building a dedicated docker file.
#
# https://cloud.docker.com/repository/registry-1.docker.io/russelljarvis/nuo
# or more recently:
# https://cloud.docker.com/u/russelljarvis/repository/docker/russelljarvis/network_unit_opt
# You can run use dockerhub to get the appropriate file, and launch this notebook using Kitematic.

# # Import libraries
# To keep the standard running version of minimal and memory efficient, not all available packages are loaded by default. In the cell below I import a mixture common python modules, and custom developed modules associated with NeuronUnit (NU) development
#!pip install dask distributed seaborn
#!bash after_install.sh


# goals.
# given https://www.nature.com/articles/nn1352
# Goal is based on this. Don't optimize to a singular point, optimize onto a cluster.
# Golowasch, J., Goldman, M., Abbott, L.F, and Marder, E. (2002)
# Failure of averaging in the construction
# of conductance-based neuron models. J. Neurophysiol., 87: 11291131.

import numpy as np
import os
import pickle
import pandas as pd
from neuronunit.tests.fi import RheobaseTestP
#from neuronunit.optimisation.model_parameters import reduced_dict, reduced_cells
from sciunit import scores# score_type

from neuronunit.optimisation.data_transport_container import DataTC
from neuronunit.tests.fi import RheobaseTestP# as discovery
from neuronunit.optimisation.optimisation_management import dtc_to_rheo, format_test, nunit_evaluation, grid_search
import quantities as pq
from neuronunit.models.reduced import ReducedModel
from neuronunit.optimisation.model_parameters import path_params
LEMS_MODEL_PATH = path_params['model_path']
list_to_frame = []
#from neuronunit.tests.fi import RheobaseTestP
import copy
from sklearn.model_selection import ParameterGrid
from neuronunit.models.interfaces import glif
import matplotlib.pyplot as plt
from neuronunit.optimisation import get_neab

import pickle
from neuronunit import tests
from neuronunit import neuroelectro
import neuronunit.optimisation.model_parameters as model_params
MODEL_PARAMS = model_params.MODEL_PARAMS
MODEL_PARAMS['results'] = {}

from neuronunit.optimisation import optimisation_management as om


from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import itertools
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import pylab as pl


from sklearn import preprocessing
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
import sklearn

from neuronunit.optimisation.optimisation_management import stochastic_gradient_descent
import seaborn as sns


def scale(X):
    before = copy.copy(X)
    for i in range(0,np.shape(X)[1]):
        temp = X[:,i]
        X[:,i] = (temp-np.mean(temp))/(np.std(temp))
    return X, before

def un_scale(X,before):
    try:
        for i in range(0,np.shape(before)[1]):
            X[0,i] = (X[0,i]*np.std(before[:,i]))+ np.mean(before[:,i])
    except:
        for i in range(0,np.shape(before)[1]):
            X[i] = (X[i]*np.std(before[:,i]))+ np.mean(before[:,i])
    return X



def mean_new(X):
    before = copy.copy(X)
    mean_param = []
    for i in range(0,np.shape(X)[1]):
        temp = X[:,i]
        mean_param.append(np.mean(temp))
    return mean_param



from collections import Iterable, OrderedDict
import quantities as qt
rts,complete_map = pickle.load(open('../tests/russell_tests.p','rb'))
local_tests = [value for value in rts['Hippocampus CA1 pyramidal cell'].values() ]

try:
    #assert 1==2
    ga_out = pickle.load(open('ca1.p','rb'))
except:
    ga_out, DO = om.run_ga(MODEL_PARAMS['RAW'], 100, local_tests, free_params = MODEL_PARAMS['RAW'],
                                NSGA = True, MU = 10, model_type = str('RAW'))#,seed_pop=seeds[key])
    pickle.dump(ga_out,open('ca1.p','wb'))


#X0_ = np.matrix([ list(ind.dtc.attrs.values()) for ind in ga_out['pf'] ])
#Y = [ np.sum(ind.fitness.values) for ind in ga_out['pf']  ]

Y = [ np.sum(v.fitness.values) for k,v in ga_out['history'].genealogy_history.items() ]
X0_ = np.matrix([ list(v.dtc.attrs.values()) for k,v in ga_out['history'].genealogy_history.items() ])

ordered_attrs = list(ga_out['history'].genealogy_history[1].dtc.attrs.keys())
#X1 = np.matrix(X1)
new,b4 = scale(X0_)



data0 = pd.DataFrame(X0_, columns = ordered_attrs)
data1 = pd.DataFrame(X1, columns = ordered_attrs)
le = preprocessing.LabelEncoder()
le.fit(ordered_attrs)
le.classes_


X_train, X_test, Y_train, Y_test = train_test_split(new, Y, test_size = 0.3)
sgd = SGDRegressor(penalty='l2', max_iter=1000, learning_rate='constant' , eta0=0.001  )
sgd.fit(X_train, Y_train)
hyperplane = sgd.coef_#+sgd.intercept_
recovered = un_scale(hyperplane,b4)
gt  = ga_out['pf'][0].dtc.attrs
new_ind = copy.copy(ga_out['pf'][0])

for i, k in enumerate(ordered_attrs):
    print(gt[k],recovered[i],k)
    new_ind[i] = recovered[i]
    new_ind.dtc.attrs[k] = recovered[i]

from neuronunit.optimisation.optimisation_management import test_runner
_,dtc_pop = test_runner([new_ind],ordered_attrs,local_tests)
print(sum(dtc_pop[0].scores.values()))

#pop,dtc_pop = test_runner([new_ind],ordered_attrs,local_tests)

#import pdb; pdb.set_trace()

#opt = un_scale(hyperplane,before)
#scaled_instances = scaler.transform([hyperplane,hyperplane])
#opt = scaled_instances[0]
#print(scaled_instances)
#import pdb; pdb.set_trace()

#gtc  = copy.copy(ga_out['pf'][0])#.attrs

#print(opt)
#print(ga_out['pf'][0].dtc.attrs)

#Y_pred = sgd.predict(X_test)
#x_mean = sgd.predict(np.array(mean_new(X_test)))
#sklearn_sgd_predictions = sgd.predict(np.mean(X_test))


#print(sklearn_sgd_predictions)
#delta_y = Y_test - sklearn_sgd_predictions;
#pl.matshow(cm)
#pl.title('Confusion matrix of the classifier')
#pl.colorbar()
#pl.show()

#stochastic_gradient_descent(ga_out)
#import sciunit
