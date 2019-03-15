"""Tests of NeuronUnit test classes"""
import unittest
import os
import sys
from sciunit.utils import NotebookTools
import dask
from dask import bag
import dask.bag as db
import matplotlib as mpl
mpl.use('Agg') # Avoid any problems with Macs or headless displays.

from sciunit.utils import NotebookTools,import_all_modules
from neuronunit import neuroelectro,bbp,aibs

from base import *


def grid_points():
    from neuronunit.optimization.optimization_management import map_wrapper

    npoints = 2
    nparams = 10
    from neuronunit.optimization.model_parameters import model_params
    provided_keys = list(model_params.keys())
    USE_CACHED_GS = False
    from neuronunit.optimization import exhaustive_search
    ## exhaustive_search

    grid_points = exhaustive_search.create_grid(npoints = npoints,nparams = nparams)
    b0 = db.from_sequence(grid_points[0:2], npartitions=8)
    dtcpop = list(db.map(exhaustive_search.update_dtc_grid,b0).compute())
    assert dtcpop is not None
    return dtcpop

def test_compute_score(dtcpop):
    from neuronunit.optimization.optimization_management import map_wrapper
    from neuronunit.optimization import get_neab
    from neuronunit.optimization.optimization_management import dtc_to_rheo
    from neuronunit.optimization.optimization_management import nunit_evaluation
    from neuronunit.optimization.optimization_management import format_test
    dtclist = list(map(dtc_to_rheo,dtcpop))
    for d in dtclist:
        assert len(list(d.attrs.values())) > 0
    dtclist = map_wrapper(format_test, dtclist)
    dtclist = map_wrapper(nunit_evaluation, dtclist)
    return dtclist

class testOptimizationBackend(NotebookTools,unittest.TestCase):

    def setUp(self):
        self.predictions = None
        self.predictionp = None
        self.score_p = None
        self.score_s = None
        self.grid_points = grid_points()
        dtcpop = self.grid_points
        self.test_compute_score = test_compute_score
        self.dtcpop = test_compute_score(dtcpop)
        self.dtc = self.dtcpop[0]
        self.rheobase = self.dtc.rheobase
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        self.standard_model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend='NEURON')
        self.model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend='NEURON')

    def test_rheobase_on_list(self):
        from neuronunit.optimization import exhaustive_search
        grid_points = self.grid_points
        second_point = grid_points[int(len(grid_points)/2)]
        three_points = [grid_points[0],second_point,grid_points[-1]]
        self.assertEqual(len(three_points),3)
        dtcpop = list(map(exhaustive_search.update_dtc_grid,three_points))
        for d in self.dtcpop:
            assert len(list(d.attrs.values())) > 0
        dtcpop = self.test_compute_score(self.dtcpop)
        self.dtcpop = dtcpop
        return dtcpop


    def test_map_wrapper(self):
        ''
        npoints = 2
        nparams = 3
        from neuronunit.optimization.model_parameters import model_params
        provided_keys = list(model_params.keys())
        USE_CACHED_GS = False
        from neuronunit.optimization import exhaustive_search
        from neuronunit.optimization.optimization_management import map_wrapper
        grid_points = exhaustive_search.create_grid(npoints = npoints,nparams = nparams)
        b0 = db.from_sequence(grid_points[0:2], npartitions=8)
        dtcpop = list(db.map(exhaustive_search.update_dtc_grid,b0).compute())
        assert dtcpop is not None
        dtcpop_compare = map_wrapper(exhaustive_search.update_dtc_grid,grid_points[0:2])
        for i,j in enumerate(dtcpop):
            for k,v in dtcpop_compare[i].attrs.items():
                print(k,v,i,j)
                self.assertEqual(j.attrs[k],v)
        return True

    def test_grid_dimensions(self):
        from neuronunit.optimization.model_parameters import model_params
        provided_keys = list(model_params.keys())
        USE_CACHED_GS = False
        from neuronunit.optimization import exhaustive_search
        from neuronunit.optimization.optimization_management import map_wrapper
        import dask.bag as db
        npoints = 2
        nparams = 3
        for i in range(1,10):
            for j in range(1,10):
                grid_points = exhaustive_search.create_grid(npoints = i, nparams = j)
                b0 = db.from_sequence(grid_points[0:2], npartitions=8)
                dtcpop = list(db.map(exhaustive_search.update_dtc_grid,b0).compute())
                self.assertEqual(i*j,len(dtcpop))
                self.assertNotEqual(dtcpop,type(None))

                dtcpop_compare = map_wrapper(exhaustive_search.update_dtc_grid,grid_points[0:2])
                self.assertNotEqual(dtcpop_compare,type(None))
                self.assertEqual(len(dtcpop_compare),len(dtcpop))
                for i,j in enumerate(dtcpop):
                    for k,v in dtcpop_compare[i].attrs.items():
                        print(k,v,i,j)
                        self.assertEqual(j.attrs[k],v)

        return True



    def test_neuron_set_attrs(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        self.assertNotEqual(self.dtcpop,None)
        dtc = self.dtcpop[0]
        self.model = ReducedModel(get_neab.LEMS_MODEL_PATH, backend=('NEURON',{'DTC':dtc}))
        temp = [ v for v in self.model.attrs.values() ]
        assert len(temp) > 0
        self.assertGreater(len(temp),0)



    import numpy as np

    def test_set_model_attrs(self):
        from neuronunit.optimization.model_parameters import model_params
        provided_keys = list(model_params.keys())
        from bluepyopt.deapext.optimisations import DEAPOptimisation
        DO = DEAPOptimisation()
        for i in range(1,10):
            for j in range(1,10):
                provided_keys = list(model_params.keys())[j]
                DO.setnparams(nparams = i, provided_keys = provided_keys)



    def test_frp(self):
        from neuronunit.models.reduced import ReducedModel
        from neuronunit.optimization import get_neab
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend=('NEURON'))
        attrs = {'a':0.02, 'b':0.2, 'c':-65+15*0.5, 'd':8-0.5**2 }
        from neuronunit.optimization.data_transport_container import DataTC
        dtc = DataTC()
        from neuronunit.tests import fi
        model.set_attrs(attrs)
        from neuronunit.optimization import get_neab
        rtp = get_neab.tests[0]
        rheobase = rtp.generate_prediction(model)
        self.assertTrue(float(rheobase['value']))

if __name__ == '__main__':
    unittest.main()
