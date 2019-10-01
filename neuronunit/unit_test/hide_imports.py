
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
from neuronunit.optimisation.optimization_management import OptMan
# dtc_to_rheo, format_test, nunit_evaluation, grid_search
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

from neuronunit.optimisation import optimisations as om


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

#from neuronunit.optimisation.optimization_management import stochastic_gradient_descent
import seaborn as sns


#from optimisation_management import init_dm_tests
#from neuronunit.optimisation.optimisation_management import init_dm_tests
from neuronunit.optimisation.optimization_management import mint_generic_model
#from neuronunit.optimisation.optimisation_management import add_dm_properties_to_cells
from collections import Iterable, OrderedDict
import quantities as qt
import os
import neuronunit
anchor = neuronunit.__file__
anchor = os.path.dirname(anchor)

mypath = os.path.join(os.sep,anchor,'tests/russell_tests.p')
#print(anchor,mypath)
#import pdb; pdb.set_trace()
rts,complete_map = pickle.load(open(mypath,'rb'))
df = pd.DataFrame(rts)

#import pdb
#pdb.set_trace()
#rts,complete_map = pickle.load(open(mypath),'rb')
local_tests = [value for value in rts['Hippocampus CA1 pyramidal cell'].values() ]
