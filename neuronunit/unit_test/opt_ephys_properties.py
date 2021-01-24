#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
import os
import scipy
import quantities as qt
import unittest

from neuronunit.optimization.optimization_management import _opt
from neuronunit.optimization.optimization_management import TSD
from neuronunit.optimization.model_parameters import MODEL_PARAMS, BPO_PARAMS
from neuronunit.allenapi.allen_data_driven import make_allen_hard_coded_limited as make_allen_hard_coded

from sciunit.scores import RelativeDifferenceScore, ZScore
from sciunit import TestSuite
from sciunit.utils import config_set, config_get
config_set('PREVALIDATE', False)
assert config_get('PREVALIDATE') is False

import warnings
warnings.filterwarnings("ignore")


class testOptimizationEphysCase(unittest.TestCase):
    def setUp(self):
        _,_,_,a_cells = make_allen_hard_coded()
        self.MU = 10
        self.NGEN = 10
        self.a_cells = a_cells
        if os.path.exists("processed_multicellular_constraints.p"):
            with open("processed_multicellular_constraints.p","rb") as f:
                experimental_constraints = pickle.load(f)
        else:
            experimental_constraints = process_all_cells()


        NC = TSD(experimental_constraints["Neocortex pyramidal cell layer 5-6"])
        NC.pop("InjectedCurrentAPWidthTest",None)
        NC.pop("InjectedCurrentAPAmplitudeTest",None)
        NC.pop("InjectedCurrentAPThresholdTest",None)
        self.NC = NC
        CA1 = TSD(experimental_constraints["Hippocampus CA1 pyramidal cell"])
        CA1.pop("InjectedCurrentAPWidthTest",None)
        CA1.pop("InjectedCurrentAPAmplitudeTest",None)
        CA1.pop("InjectedCurrentAPThresholdTest",None)
        self.CA1 = CA1
    def test_allen_good_agreement_opt(self):
        final_pop, hall_of_fame, logs, hist,best_ind,best_fit_val,opt = _opt(
            self.a_cells['471819401'],
            BPO_PARAMS,
            '471819401',
            "ADEXP",
            self.MU,
            self.NGEN,
            "IBEA",
            use_streamlit=False,
            score_type=RelativeDifferenceScore
            )
    def test_allen_fi_curve_opt(self):

        final_pop, hall_of_fame, logs, hist,best_ind,best_fit_val,opt = _opt(
            self.a_cells['fi_curve'],
            BPO_PARAMS,
            'fi_curve',
            "ADEXP",
            self.MU,
            self.NGEN,
            "IBEA",
            use_streamlit=False,
            score_type=RelativeDifferenceScore
            )
    def test_neuro_electro_adexp_opt(self):
        self.MU = 10
        self.NGEN = 10
        final_pop, hall_of_fame, logs, hist,best_ind,best_fit_val,opt = _opt(
            self.NC,
            BPO_PARAMS,
            "Neocortex pyramidal cell layer 5-6",
            "ADEXP",
            self.MU,
            self.NGEN,
            "IBEA",
            use_streamlit=False,
            score_type=ZScore
            )


    '''
    Rick, some of these bellow are unit tests
    that cannot pass without changes to sciunit complete
    '''
    @skip_incapable
    def test_neuro_electro_adexp_opt_ca1(self):
        self.MU = 35
        self.NGEN = 10
        final_pop, hall_of_fame, logs, hist,best_ind,best_fit_val,opt = _opt(
            self.CA1,
            BPO_PARAMS,
            "Hippocampus CA1 pyramidal cell",
            "ADEXP",
            self.MU,
            self.NGEN,
            "IBEA",
            score_type=ZScore
            )

    @skip_incapable
    def test_neuro_electro_izhi_opt_pyr(self):
        self.MU = 100
        self.NGEN = 1

        final_pop, hall_of_fame, logs, hist,best_ind,best_fit_val,opt = _opt(
            self.NC,
            BPO_PARAMS,
            "Neocortex pyramidal cell layer 5-6",
            "IZHI",
            self.MU,
            self.NGEN,
            "IBEA",
            score_type=ZScore
            )
        old_result = np.sum(best_fit_val)
        self.NGEN = 35

        final_pop, hall_of_fame, logs, hist,best_ind,best_fit_val,opt = _opt(
            self.NC,
            BPO_PARAMS,
            "Neocortex pyramidal cell layer 5-6",
            "IZHI",
            self.MU,
            self.NGEN,
            "IBEA",
            use_streamlit=False,
            score_type=ZScore
            )
        new_result = np.sum(best_fit_val)
        assert new_result<old_result
