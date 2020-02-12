from neuronunit.optimisation.optimization_management import TSD
"""Tests of NeuronUnit test classes"""
import unittest
import os
import sys
#import dask
#from dask import bag
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from itertools import repeat
import quantities as pq

import copy
import unittest
import pickle

import numpy as np
import pickle
#import dask.bag as db
import os

#from neuronunit.optimisation import get_neab
from neuronunit.optimisation.data_transport_container import DataTC

from neuronunit.optimisation.optimization_management import dtc_to_rheo#, mint_generic_model
from neuronunit.optimisation.optimization_management import OptMan,inject_and_plot_model

#from neuronunit import tests as nu_tests, neuroelectro
from neuronunit.tests import passive, waveform, fi
from neuronunit.optimisation import exhaustive_search
# from neuronunit.optimisation.model_parameters import MODEL_PARAMS
# from neuronunit.tests import dynamics
#from neuronunit.optimisation import data_transport_container

from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
#from neuronunit import aibs
# from neuronunit.optimisation.optimisations import run_ga
#from neuronunit.optimisation import model_parameters
from neuronunit.optimisation import mint_tests
from neuronunit.optimisation import get_neab
test_frame = get_neab.process_all_cells()
test_frame.pop('Olfactory bulb (main) mitral cell',None)
from IPython.display import HTML, display
