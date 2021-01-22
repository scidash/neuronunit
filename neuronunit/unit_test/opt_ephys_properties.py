#!/usr/bin/env python
# coding: utf-8
import pandas as pd
pd.set_option('max_columns',None)
pd.set_option('max_rows',None)
import matplotlib.pyplot as plt
import seaborn as sns
cm = sns.light_palette("green", as_cmap=True)
import pickle
import glob
import os
import scipy
import quantities as qt
import unittest

from neuronunit.optimization.optimization_management import instance_opt
from neuronunit.optimization.optimization_management import TSD
from neuronunit.optimization.model_parameters import MODEL_PARAMS, BPO_PARAMS
from neuronunit.allenapi.allen_data_driven import make_allen_hard_coded_limited as make_allen_hard_coded

from sciunit.scores import RelativeDifferenceScore, ZScore
from sciunit import TestSuite

import warnings
warnings.filterwarnings("ignore")

if os.path.exists("processed_multicellular_constraints.p"):
    cells = pickle.load(open("processed_multicellular_constraints.p","rb"))
else:
    cells = process_all_cells()

ncl5 = TSD(cells["Neocortex pyramidal cell layer 5-6"])
ncl5.name = str("Neocortex pyramidal cell layer 5-6")
ncl5_vr = ncl5["RestingPotentialTest"].observation['mean']

ca1 = TSD(cells["Hippocampus CA1 pyramidal cell"])
ca1.name = str("Hippocampus CA1 pyramidal cell")

ca1_vr = ca1["RestingPotentialTest"].observation['mean']
experimental_constraints= cells

class testOptimizationEphysCase(unittest.TestCase):
    def setUp(self):
        _,_,_,a_cells = make_allen_hard_coded()
        self.MU = 100
        self.NGEN = 40
        self.a_cells = a_cells
        NC = TSD(experimental_constraints["Neocortex pyramidal cell layer 5-6"])
        NC.pop("InjectedCurrentAPWidthTest",None)
        NC.pop("InjectedCurrentAPAmplitudeTest",None)
        NC.pop("InjectedCurrentAPThresholdTest",None)
        self.NC = NC
    def test_0_allen_good_agreement_opt_0(self):
        final_pop, hall_of_fame, logs, hist,best_ind,best_fit_val,opt = instance_opt(
            self.a_cells['471819401'],
            BPO_PARAMS,
            '471819401',
            "ADEXP",
            self.MU,
            self.NGEN,
            "IBEA",
            use_streamlit=False,score_type=RelativeDifferenceScore
            )
    def test_1_allen_fi_curve_opt_1(self):
        print('second')

        final_pop, hall_of_fame, logs, hist,best_ind,best_fit_val,opt = instance_opt(
            self.a_cells['fi_curve'],
            BPO_PARAMS,
            'fi_curve',
            "ADEXP",
            self.MU,
            self.NGEN,
            "IBEA",
            use_streamlit=False
            )
    def test_4_neuro_electro_adexp_opt_4(self):
        self.MU = 35
        self.NGEN = 100
        final_pop, hall_of_fame, logs, hist,best_ind,best_fit_val,opt = instance_opt(
            self.NC,
            BPO_PARAMS,
            "Neocortex pyramidal cell layer 5-6",
            "ADEXP",
            self.MU,
            self.NGEN,
            "IBEA",
            use_streamlit=False
            )



    def test_2_neuro_electro_izhi_opt_pyr_2(self):
        self.MU = 100
        self.NGEN = 100

        final_pop, hall_of_fame, logs, hist,best_ind,best_fit_val,opt = instance_opt(
            self.NC,
            BPO_PARAMS,
            "Neocortex pyramidal cell layer 5-6",
            "IZHI",
            self.MU,
            self.NGEN,
            "IBEA",
            use_streamlit=False
            )

    def test_3_neuro_electro_adexp_opt_ca1_3(self):
        self.MU = 100
        self.NGEN = 100
        print('first ?')
        final_pop, hall_of_fame, logs, hist,best_ind,best_fit_val,opt = instance_opt(
            self.NC,
            BPO_PARAMS,
            "Hippocampus CA1 pyramidal cell",
            "ADEXP",
            self.MU,
            self.NGEN,
            "IBEA",
            use_streamlit=False
            )
