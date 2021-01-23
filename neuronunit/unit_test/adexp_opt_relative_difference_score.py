import unittest
import matplotlib

matplotlib.use("Agg")
from neuronunit.allenapi.allen_data_driven import opt_setup, opt_setup_two, opt_exec
from neuronunit.allenapi.allen_data_driven import opt_to_model, meta_setup
from neuronunit.allenapi.utils import dask_map_function

# except:
#    from bluepyopt.allenapi.allen_data_driven import opt_setup, opt_setup_two, opt_exec, opt_to_model
#    from bluepyopt.allenapi.allen_data_driven import opt_to_model
#    from bluepyopt.allenapi.utils import dask_map_function

from neuronunit.optimization.optimization_management import check_bin_vm15
from neuronunit.optimization.model_parameters import (
    MODEL_PARAMS,
    BPO_PARAMS,
    to_bpo_param,
)
from neuronunit.optimization.optimization_management import (
    dtc_to_rheo,
    inject_and_plot_model,
)
import numpy as np
from neuronunit.optimization.data_transport_container import DataTC
import efel
from jithub.models import model_classes
import matplotlib.pyplot as plt
import quantities as qt
import os
from sciunit.scores import RelativeDifferenceScore, ZScore


class testOptimization(unittest.TestCase):
    def setUp(self):
        self = self
        self.ids = [
            324257146,
            325479788,
            476053392,
            623893177,
            623960880,
            482493761,
            471819401,
        ]

    def test_opt_relative_diff(self):
        specimen_id = self.ids[1]
        model_type = "ADEXP"

        if model_type == "IZHI":
            model = model_classes.IzhiModel()
        if model_type == "MAT":
            model = model_classes.MATModel()
        if model_type == "ADEXP":
            model = model_classes.ADEXPModel()

        target_num_spikes = 8  # This is the number of spikes to look for in the data

        fixed_current = 122 * qt.pA
        NGEN = 15
        MU = 12
        mapping_funct = dask_map_function
        cell_evaluator, simple_cell, suite, target_current, spk_count = meta_setup(
            specimen_id,
            model_type,
            target_num_spikes,
            template_model=model,
            fixed_current=False,
            cached=True,
            score_type=RelativeDifferenceScore,
        )
        final_pop, hall_of_fame, logs, hist = opt_exec(
            MU, NGEN, mapping_funct, cell_evaluator
        )
        opt, target = opt_to_model(
            hall_of_fame, cell_evaluator, suite, target_current, spk_count
        )
        best_ind = hall_of_fame[0]
        fitnesses = cell_evaluator.evaluate_with_lists(best_ind)
        self.assertGreater(0.7, np.sum(fitnesses))

    def test_opt_ZScore(self):
        specimen_id = self.ids[1]
        model_type = "ADEXP"

        if model_type == "IZHI":
            model = model_classes.IzhiModel()
        if model_type == "MAT":
            model = model_classes.MATModel()
        if model_type == "ADEXP":
            model = model_classes.ADEXPModel()

        target_num_spikes = 8
        dtc = DataTC()
        dtc.backend = model_type
        dtc._backend = model._backend
        dtc.attrs = model.attrs
        dtc.params = {k: np.mean(v) for k, v in MODEL_PARAMS[model_type].items()}

        fixed_current = 122 * qt.pA
        NGEN = 15
        MU = 12
        mapping_funct = dask_map_function
        cell_evaluator, simple_cell, suite, target_current, spk_count = meta_setup(
            specimen_id,
            model_type,
            target_num_spikes,
            template_model=model,
            fixed_current=False,
            cached=True,
            score_type=ZScore,
        )
        final_pop, hall_of_fame, logs, hist = opt_exec(
            MU, NGEN, mapping_funct, cell_evaluator
        )
        opt, target = opt_to_model(
            hall_of_fame, cell_evaluator, suite, target_current, spk_count
        )
        best_ind = hall_of_fame[0]
        fitnesses = cell_evaluator.evaluate_with_lists(best_ind)
        self.assertGreater(0.7, np.sum(fitnesses))


# if __name__ == '__main__':
#    unittest.main()

tt = testOptimization()
tt.setUp()

# +

tt.test_opt_relative_diff()
# -

tt.test_opt_ZScore()
