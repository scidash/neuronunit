"""F/I neuronunit tests, e.g. investigating firing rates and patterns as a
function of input current"""

import os
import multiprocessing
global cpucount
npartitions = cpucount = multiprocessing.cpu_count()
from .base import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
from .. import optimisation

from neuronunit.optimisation.data_transport_container import DataTC
import os
import quantities
import neuronunit
from neuronunit.models.reduced import ReducedModel# , VeryReducedModel
import dask.bag as db
import quantities as pq
import numpy as np
import copy
import pdb
#from numba import jit
import time
import numba
import copy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from neuronunit.capabilities.spike_functions import get_spike_waveforms, spikes2amplitudes, threshold_detection
#
# When using differentiation based spike detection is used this is faster.



class NetTest(VmTest):

    required_capabilities = (cap.ProducesMultiMembranePotentials,
                             cap.ProducesSpikeRasters)

    params = {'injected_square_current':
                {'mean':100.0*pq.pA, 'standard':std*pq.pA, 'duration':DURATION}}

    description = ("A test of the nTE")
    units = pq.pA
    name = 'Transfer Entropy'
    score_type = scores.RatioScore

    def generate_prediction(self, model, current, syn_weights):
        """Implementation of sciunit.Test.generate_prediction."""
        (vms,binary_train,exhaustive_data) = model.inject_noise_current(stim_current, syn_weights)
        prediction['vms'] = vms
        prediction['binary_train'] = binary_train
        prediction['exhaustive_data'] = exhaustive_data
        return prediction
    
    def get_observations():
        try:
            os.system('pip install rpy2')
            from rpy2.robjects.packages import importr
            rpy2('install.packages("remotes")')
            rpy2('install.packages("osfr")')
            rpy2('library(osfr)')
            rpy2('cr_project <- osf_retrieve_node("64jhz")')

        except:

            os.system('wget https://osf.io/64jhz/download')


class ISITest(NetTest):
    pass

class CVTest(NetTest):
    pass


class TE_TEST(NetTest):
    self.prediction = {}
    self.prediction['vms'] = None
    self.prediction['binary_train'] = None
    self.prediction['exhaustive_data'] = None

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        (vms,binary_train,exhaustive_data) = model.inject_noise_current(current)
        self.prediction['vms'] = vms
        self.prediction['binary_train'] = binary_train
        xs = [0,0,1,1,1,1,0,0,0]
        ys = [0,1,1,1,1,0,0,0,1]
        transfer_entropy(ys, xs, k=1)
        #0.8112781244591329
        # https://elife-asu.github.io/PyInform/timeseries.html#module-pyinform.transferentropy
        self.prediction['exhaustive_data'] = exhaustive_data
        return self.prediction


    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        if self.prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(TE_TEST,self).\
                        compute_score(observation, self.prediction)
        return score

    def bind_score(self, score, model, observation, prediction):
        super(TE_TEST,self).bind_score(score, model,
                                            observation, self.prediction)
        if self.rheobase_vm is not None:
            score.related_data['exhaustive_data'] = self.prediction['exhaustive_data']
