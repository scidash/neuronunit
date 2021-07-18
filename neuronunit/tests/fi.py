"""F/I neuronunit tests.

For example, investigating firing rates and patterns as a
function of input current.
"""

import os
import multiprocessing
import copy

import dask.bag as db    
import neuronunit
from neuronunit.models.reduced import ReducedModel
from sciunit import log

from .base import np, pq, ncap, VmTest, scores, AMPL, DELAY, DURATION
import quantities
import neuronunit

import numpy as np
import copy
import time
import copy
import dask
from neuronunit.capabilities.spike_functions import (
    get_spike_waveforms,
    spikes2amplitudes,
    threshold_detection,
)

global cpucount
npartitions = cpucount = multiprocessing.cpu_count()
tolerance = 0.0


class RheobaseTest(VmTest):
    """
    --Synopsis:
        Serial implementation of a binary search to test the rheobase.

        Strengths: this algorithm is faster than the parallel class, present in
        this file under important and limited circumstances: this serial algorithm
        is faster than parallel for model backends that are able to implement
        numba jit optimization.


    """

    def __init__(
        self,
        observation=None,
        prediction=None,
        name="RheobaseTest",
        **params
    ):
        super(RheobaseTest, self).__init__(observation=observation, name=name)
        
    required_capabilities = (ncap.ReceivesSquareCurrent, ncap.ProducesSpikes)

    name = "Rheobase test"
    description = (
        "A test of the rheobase, i.e. the minimum injected current "
        "needed to evoke at least one spike."
    )
    units = pq.pA
    ephysprop_name = "Rheobase"
    default_params = dict(VmTest.default_params)
    default_params.update(
        {
            "amplitude": 100 * pq.pA,
            "duration": DURATION,
            "delay": DELAY,
            "high" : 900 * pq.pA,
            "small" : 0 * pq.pA,
            "tmax": 2000 * pq.ms,
            "tolerance": 0.5 * pq.pA,
            "target_number_spikes": 1,
            "max_iters": 20,
        }
    )

    params_schema = dict(VmTest.params_schema)
    params_schema.update(
        {"tolerance": {"type": "current", "min": 1e-5, "required": False},
         "high": {"type": "current", "min": 0, "max": 1e6, "required": False},
         "small": {"type": "current", "min": -1e-6, "max": 1e6, "required": False},
         "target_number_spikes": {"type": "integer", "min": 1, "required": False},
         "max_iters": {"type": "integer", "min": 1, "required": False},
         }
    )

    def condition_model(self, model):
        model.set_run_params(t_stop=self.params["tmax"])

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        self.condition_model(model)
        prediction = {"value": None}
        model.rerun = True

        if self.observation is not None:
            try:
                units = self.observation["value"].units
            except KeyError:
                units = self.observation["mean"].units
        else:
            units = pq.pA
        lookup = self.threshold_FI(model, units)
        sub = np.array([x for x in lookup if lookup[x] == 0])*units
        supra = np.array([x for x in lookup if lookup[x] > 0])*units
        if len(sub):
            log("Highest subthreshold current is %s" % (float(sub.max())*units))
        else:
            log("No subthreshold current was tested.")
        if len(supra):
            log("Lowest suprathreshold current is %s" % supra.min())
        else:
            log("No suprathreshold current was tested.")
        if len(sub) and len(supra):
            rheobase = supra.min()
        else:
            rheobase = None
        
        prediction["value"] = rheobase
        self.FI = lookup
        self.prediction = prediction
        
        return prediction

    def extract_features(self, model):
        prediction = self.generate_prediction(model)
        self.prediction = prediction
        return prediction

    def threshold_FI(self, model, units, guess=None):
        """Use binary search to generate an FI curve including rheobase."""
        lookup = {}  # A lookup table global to the function below.

        def f(ampl):
            if float(ampl) not in lookup:
                current = self.get_injected_square_current()
                current["amplitude"] = float(ampl) * pq.pA
                model.inject_square_current(**current)
                n_spikes = model.get_spike_count()
                    # if self.target_num_spikes == 1:
                    # ie this is rheobase search
                    # vm = model.get_membrane_potential()
                    # if vm[-1]>0 and n_spikes==1:
                    # this means current was not strong enough
                    # to evoke an early spike.
                    # the voltage deflection did not come back down below zero.
                    # treat this as zero spikes because a slightly higher
                    # spike will give a cleaner rheobase waveform.
                log("Injected %s current and got %d spikes" % (ampl, n_spikes), level=10)
                lookup[float(ampl)] = n_spikes
                spike_counts = np.array([n for x, n in lookup.items() if n > 0])
                if n_spikes and n_spikes <= spike_counts.min():
                    self.rheobase_vm = model.get_membrane_potential()

        f(self.params['high'])
        f(self.params['small'])
        
        i = 0

        while i <= self.params['max_iters']:
            # sub means below threshold, or no spikes
            sub = np.array([x for x in lookup if lookup[x] == 0]) * units
            
            # supra means above threshold,
            # but possibly too high above threshold.
            supra = np.array([x for x in lookup if lookup[x] > 0]) * units
           
            if len(supra) and len(sub):
                delta = supra.min() - sub.max()
                if delta < self.params['tolerance']:
                    break

            # Its this part that should be like an evaluate function
            # that is passed to futures map.
            if len(sub) and len(supra):
                f((supra.min() + sub.max()) / 2)
            elif len(sub):
                f(max(small, sub.max() * 10))
            elif len(supra):
                f(min(-small, supra.min() * 2))
            i += 1

        return lookup

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        # from sciunit.scores import BooleanScore
        #
        # if type(self.score_type) == BooleanScore:
        #    print('warning using unusual score type')
        if prediction is None or (
            isinstance(prediction, dict) and prediction["value"] is None
        ):
            score = scores.InsufficientDataScore(None)
        else:

            score = super(RheobaseTest, self).compute_score(
                observation, prediction
            )  # max
        return score

    def bind_score(self, score, model, observation, prediction):
        """Bind additional attributes to the test score."""
        super(RheobaseTest, self).bind_score(score, model, observation, prediction)
        if self.rheobase_vm is not None:
            score.related_data["vm"] = self.rheobase_vm
