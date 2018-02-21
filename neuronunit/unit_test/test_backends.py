from neuronunit.tests.fi import RheobaseTest, RheobaseTestP
#from neuronunit.optimization.get_neab import fi_basket_tests
from neuronunit.models.reduced import ReducedModel
from neuronunit import aibs
from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization import get_neab
from neuronunit.optimization.model_parameters import model_params
from neuronunit.optimization import exhaustive_search
from neuronunit.optimization.optimization_management import dtc_to_rheo
from neuronunit.optimization.optimization_management import nunit_evaluation
from neuronunit.optimization.optimization_management import format_test
from neuronunit.optimization.exhaustive_search import update_dtc_grid

import os
import dask.bag as db

from neuronunit.optimization import get_neab
from neuronunit.models.reduced import ReducedModel


npoints = 2
nparams = 10
provided_keys = list(model_params.keys())
USE_CACHED_GS = False
grid_points = exhaustive_search.create_grid(npoints = npoints,nparams = nparams)
#dtcpop = list(update_dtc_pop(grid_points,td = td))
b0 = db.from_sequence(grid_points, npartitions=8)
dtcpop = list(db.map(update_dtc_grid,b0).compute())
print(dtcpop)
N = 3


def sub_test_backend(dtc):
    import copy
    import unittest

    dtc = copy.copy(dtc)
    dtc.scores = {}
    #unittest.TestCase
    #from unittest.TestCase import assertGreater
    size = len(list(dtc.attrs.values()))
    unittest.case.TestCase.assertGreater(size,0)
    model = ReducedModel(get_neab.LEMS_MODEL_PATH, name= str('vanilla'), backend=('NEURON', {'DTC':dtc}))
    rbt = get_neab.tests[0]
    scoreN = rbt.judge(model,stop_on_error = False, deep_error = True)
    import copy
    dtc.scores[str(rbt)] = copy.copy(scoreN.sort_key)
    dtc.rheobase = copy.copy(scoreN.prediction)
    return dtc



def test_backend(grid_points):
    second_point = grid_points[int(len(grid_points)/2)]
    dtcpop = list(map(exhaustive_search.update_dtc_grid,[grid_points[0],second_point,grid_points[-1]]))
    for i, dtc in enumerate(dtcpop):
        dtcpop[i] = sub_test_backend(dtc)
    return dtcpop

def test_serial(grid_points):
    models = test_backend(grid_points)
    models = list(map(format_test,models).compute())
    models = list(map(nunit_evaluation,models).compute())
    self.modelss = models
    return models

def test_parallel(grid_points):
    models = test_backend(grid_points)
    b0 = db.from_sequence(models, npartitions=8)
    models = list(db.map(format_test,b0).compute())
    b0 = db.from_sequence(models, npartitions=8)
    modelss = list(db.map(nunit_evaluation,b0).compute())
    self.modelsp = models
    return chunk


def test_subset(self):
    from neuronunit.optimization import create_subset
    create_subset(5)

def test_update_deap_pop(self):
    from neuronunit.optimization import update_deap_pop

def test_dtc_to_rheo(self):
    from neuronunit.optimization import dtc_to_rheo
    dtc_to_rheo(dtc)

def serial_equals_parallel(self):
    #self.test_parallel()
    self.test_serial(grid_points)
    self.test_parallel(grid_points)
    for i,d0 in enumerate(self.modelsp):
        for k, v in enumerate(d0.tests):
            print(v, self.modelss[i].tests[k])
            assert v == self.modelss[i].tests[k]

def test_evaluate(self,dtc):
    from neuronunit.optimization_management import evaluate
    assert dtc.scores is not None
    evauate(dtc)
