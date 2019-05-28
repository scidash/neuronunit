
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
#MODEL_PARAMS['results'] = {}

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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
import sklearn
import neuronunit.optimisation.model_parameters as model_params

from neuronunit.optimisation.optimisation_management import stochastic_gradient_descent
import seaborn as sns


def scale(X):
    before = copy.copy(X)
    for i in range(0,np.shape(X)[1]):
        temp = X[:,i]
        X[:,i] = (temp-np.mean(temp))/(np.std(temp))
    return X, before

def scale2(X, other=None):
    before = copy.copy(X)
    new = []
    for i in range(0,np.shape(X)[1]):
        temp = X[:,i]
        #print(np.mean(temp)/np.std(temp))
        ti = other[i]
        new.append((ti-np.mean(temp))/(np.std(temp)))
    print(new)
    return new, before

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
    assert 1==2
    ga_out = pickle.load(open('ca1.p','rb'))
except:
    ga_out, DO = om.run_ga(model_params.MODEL_PARAMS['BAE1'], 10, local_tests, free_params = model_params.MODEL_PARAMS['BAE1'],
                                NSGA = True, MU = 5, model_type = str('ADEXP'))#,seed_pop=seeds[key])
    pickle.dump(ga_out,open('ca1.p','wb'))
new_ind = copy.copy(ga_out['pf'][0])


Y = [ np.sum(v.fitness.values) for k,v in ga_out['history'].genealogy_history.items() ]
Y_ = np.matrix([ list(v.fitness.values) for k,v in ga_out['history'].genealogy_history.items() ])

X0_ = np.matrix([ list(v.dtc.attrs.values()) for k,v in ga_out['history'].genealogy_history.items() ])

ordered_attrs = list(ga_out['history'].genealogy_history[1].dtc.attrs.keys())
new,b4 = scale(X0_)



data0 = pd.DataFrame(X0_, columns = ordered_attrs)
#data1 = pd.DataFrame(X1, columns = ordered_attrs)
le = preprocessing.LabelEncoder()
le.fit(ordered_attrs)
le.classes_
new_ind.backend=str('ADEXP')
new_ind.boundary_dict = model_params.MODEL_PARAMS['BAE1']

from neuronunit.optimisation.optimisation_management import test_runner, cell_to_test_mapper
from neuronunit.optimisation.optimisation_management import inject_and_plot


import pdb; pdb.set_trace()
_,dtc_pop = test_runner([new_ind],ordered_attrs,local_tests)
print(sum(dtc_pop[0].scores.values()))

def generate_dataset(n_train, n_test, n_features, noise=0.1, verbose=False):
    """Generate a regression dataset with the given parameters."""
    if verbose:
        print("generating dataset...")

    random_seed = 13
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_train, test_size=n_test, random_state=random_seed)
    X_train, y_train = shuffle(X_train, y_train, random_state=random_seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train[:, None])[:, 0]
    y_test = y_scaler.transform(y_test[:, None])[:, 0]

    return X_train, y_train, X_test, y_test


X_scaler = StandardScaler()
X0_ = X_scaler.fit_transform(X0_)
X = X_scaler.transform(X0_)
Y = X_scaler.fit_transform(Y_)
Y = X_scaler.transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
sgd = SGDRegressor(penalty='l2', max_iter=1000, learning_rate='constant' , eta0=0.001  )

mlp = MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

#sgd.fit(X_train, Y_train)
mlp.fit(X_train, Y_train)

hyperplane = mlp.coef_#+sgd.intercept_

recovered = un_scale(hyperplane,b4)
test_val = [ ga_out['pf'][0].dtc.attrs[k] for k in ordered_attrs ]
X0_ = np.matrix([ list(v.dtc.attrs.values()) for k,v in ga_out['history'].genealogy_history.items() ])

other,_ = scale2(X0_,other=test_val)
pred = sgd.predict([other])
#sgd.predict([other])


gt  = ga_out['pf'][0].dtc.attrs



for t in X_test:
    print(ordered_attrs)
    contents = un_scale(t,b4)#,ordered_attrs)
    print([(contents[0,i],j) for i,j in enumerate(ordered_attrs)])
    print(sgd.predict(t))

for i, k in enumerate(ordered_attrs):
    print(gt[k],recovered[i],k)
    new_ind[i] = recovered[i]
    new_ind.dtc.attrs[k] = recovered[i]
#cell_to_test_mapper

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
