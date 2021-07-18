from typing import Any, Dict, List, Optional, Tuple, Type, Union, Text

import pickle
import seaborn as sns
import os
from scipy.interpolate import interp1d

import bluepyopt as bpop
import bluepyopt.ephys as ephys
from bluepyopt.parameters import Parameter

import matplotlib.pyplot as plt
import copy
import numpy as np
from collections.abc import Iterable
import pandas as pd

from sciunit.scores import RelativeDifferenceScore
from sciunit import TestSuite
from sciunit.scores import ZScore
from sciunit.scores.collections import ScoreArray

from neuronunit.allenapi import make_allen_tests_from_id
from neuronunit.allenapi.make_allen_tests_from_id import *
from neuronunit.allenapi.make_allen_tests import AllenTest
from neuronunit.optimization.optimization_management import inject_model_soma
from neuronunit.optimization.model_parameters import BPO_PARAMS

def match_current_amp_to_model_param(spk_count,
                                    model_type,
                                    template_model,
                                    fixed_current):
    observation_range = {}
    observation_range["value"] = spk_count
    if fixed_current:
        uc = {
            "amplitude": fixed_current,
            "duration": ALLEN_DURATION,
            "delay": ALLEN_DELAY,
        }
        target_current = None
    else:
        scs = SpikeCountSearch(observation_range)
        target_current = scs.generate_prediction(template_model)
    return target_current

def opt_setup(
    specimen_id,
    model_type,
    target_num,
    template_model=None,
    cached=None,
    fixed_current=False,
    score_type=ZScore,
    efel_filter_iterable=None,
):
    if cached:
        with open(str(specimen_id) + "later_allen_NU_tests.p", "rb") as f:
            suite = pickle.load(f)

    else:

        sweep_numbers, data_set, sweeps = make_allen_tests_from_id.allen_id_to_sweeps(
            specimen_id
        )
        (
            vmm,
            stimulus,
            sn,
            spike_times,
        ) = make_allen_tests_from_id.get_model_parts_sweep_from_spk_cnt(
            target_num, data_set, sweep_numbers, specimen_id
        )
        (
            suite,
            specimen_id,
        ) = make_allen_tests_from_id.make_suite_known_sweep_from_static_models(
            vmm, stimulus, specimen_id, efel_filter_iterable=efel_filter_iterable
        )
        with open(str(specimen_id) + "later_allen_NU_tests.p", "wb") as f:
            pickle.dump(suite, f)
    if "vm_soma" in suite.traces.keys():
        target = StaticModel(vm=suite.traces["vm_soma"])
        target.vm_soma = suite.traces["vm_soma"]
    else:
        target = StaticModel(vm=suite.traces["vm15"])
        target.vm_soma = suite.traces["vm15"]

    nu_tests = suite.tests

    attrs = {k: np.mean(v) for k, v in MODEL_PARAMS[model_type].items()}
    spktest = [ t for t in nu_tests if t.name == "Spikecount"][0]
    spk_count = float(spktest.observation["mean"])
    template_model.backend = model_type
    template_model.allen = True
    template_model.NU = True
    target_current = match_current_amp_to_model_param(spk_count,
                            model_type,
                            template_model,
                            fixed_current)
    template_model.seeded_current = target_current["value"]

    cell_evaluator, template_model = opt_setup_two(
        model_type,
        suite,
        nu_tests,
        target_current,
        spk_count,
        template_model=template_model,
        score_type=score_type,
        efel_filter_iterable=efel_filter_iterable
    )
    return suite, target_current, spk_count, cell_evaluator, template_model


class NUFeatureAllenMultiSpike(object):
    def __init__(
        self, test, model, cnt, target, spike_obs, print_stuff=False, score_type=None
    ):
        self.test = test
        self.model = model
        self.spike_obs = spike_obs
        self.cnt = cnt
        self.target = target
        self.score_type = score_type
        self.score_array = None

    def calculate_score(self, responses):
        if not "features" in responses.keys():
            return 1000.0
        features = responses["features"]
        if features is None:
            return 1000.0
        self.test.score_type = self.score_type
        feature_name = self.test.name
        if feature_name not in features.keys():
            return 1000.0

        if features[feature_name] is None:
            return 1000.0
        if type(features[self.test.name]) is type(Iterable):
            features[self.test.name] = np.mean(features[self.test.name])

        self.test.observation["mean"] = np.mean(self.test.observation["mean"])
        self.test.set_prediction(np.mean(features[self.test.name]))
        if "Spikecount" == feature_name:
            delta = np.abs(
                features[self.test.name] - np.mean(self.test.observation["mean"])
            )
            if np.nan == delta or delta == np.inf:
                delta = 1000.0
            return delta
        else:
            if features[feature_name] is None:
                return 1000.0

            prediction = {"value": np.mean(features[self.test.name])}
            score_gene = self.test.judge(responses["model"], prediction=prediction)
            if score_gene is not None:
                if self.test.score_type is RelativeDifferenceScore:
                    if score_gene.log_norm_score is not None:
                        delta = np.abs(float(score_gene.log_norm_score))
                    else:
                        if score_gene.raw is not None:
                            delta = np.abs(float(score_gene.raw))
                        else:
                            delta = 1000.0

                else:
                    if score_gene.raw is not None:
                        delta = np.abs(float(score_gene.raw))
                    else:
                        delta = 1000.0
            else:
                delta = 1000.0
            if np.nan == delta or delta == np.inf:
                delta = 1000.0
            return delta


def opt_setup_two(
    model_type,
    suite,
    nu_tests,
    target_current,
    spk_count,
    template_model=None,
    score_type=ZScore,
    efel_filter_iterable=None
):
    assert template_model.backend == model_type
    template_model.params = BPO_PARAMS[model_type]
    template_model.params_by_names(list(BPO_PARAMS[model_type].keys()))
    template_model.seeded_current = target_current["value"]
    template_model.spk_count = spk_count
    sweep_protocols = []
    protocol = ephys.protocols.NeuronUnitAllenStepProtocol(
        "multi_spiking", [None], [None]
    )
    sweep_protocols.append(protocol)
    onestep_protocol = ephys.protocols.SequenceProtocol(
        "multi_spiking_wraper", protocols=sweep_protocols
    )
    objectives = []
    spike_obs = []
    for tt in nu_tests:
        if "Spikecount" == tt.name:
            spike_obs.append(tt.observation)
    spike_obs = sorted(spike_obs, key=lambda k: k["mean"], reverse=True)
    for cnt, tt in enumerate(nu_tests):
        feature_name = "%s" % (tt.name)
        ft = NUFeatureAllenMultiSpike(
            tt, template_model, cnt, target_current, spike_obs, score_type=score_type
        )
        objective = ephys.objectives.SingletonObjective(feature_name, ft)
        objectives.append(objective)
    score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives)
    template_model.params_by_names(BPO_PARAMS[template_model.backend].keys())
    template_model.efel_filter_iterable = efel_filter_iterable
    cell_evaluator = ephys.evaluators.CellEvaluator(
        cell_model=template_model,
        param_names=list(BPO_PARAMS[template_model.backend].keys()),
        fitness_protocols={onestep_protocol.name: onestep_protocol},
        fitness_calculator=score_calc,
        sim="euler",
    )
    assert cell_evaluator.cell_model is not None
    return cell_evaluator, template_model


"""
def multi_layered(MU, NGEN, mapping_funct, cell_evaluator2):
    optimisation = bpop.optimisations.DEAPOptimisation(
        evaluator=cell_evaluator2,
        offspring_size=MU,
        map_function=map,
        selector_name="IBEA",
        mutpb=0.05,
        cxpb=0.6,
        current_fixed=from_outer,
    )
    final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=NGEN)
    return final_pop, hall_of_fame, logs, hist
"""


def opt_exec(MU, NGEN, mapping_funct, cell_evaluator, mutpb=0.05, cxpb=0.6,neuronunit=True):

    optimisation = bpop.optimisations.DEAPOptimisation(
        evaluator=cell_evaluator,
        offspring_size=MU,
        eta=25,
        map_function=map,
        selector_name="IBEA",
        mutpb=mutpb,
        cxpb=cxpb,
        neuronunit=neuronunit,
    )
    final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=NGEN)
    return final_pop, hall_of_fame, logs, hist


def opt_to_model(hall_of_fame, cell_evaluator, suite, target_current, spk_count):
    best_ind = hall_of_fame[0]
    model = cell_evaluator.cell_model
    tests = suite.tests
    scores = []
    obs_preds = []

    for t in tests:
        scores.append(t.judge(model, prediction=t.prediction))
        obs_preds.append(
            (t.name, t.observation["mean"], t.prediction["mean"], scores[-1])
        )
    df = pd.DataFrame(obs_preds)

    opt = model#.model_to_dtc()
    opt.attrs = {
        str(k): float(v) for k, v in cell_evaluator.param_dict(best_ind).items()
    }
    target = copy.copy(opt)
    if "vm_soma" in suite.traces.keys():
        target.vm_soma = suite.traces["vm_soma"]
    else:  # backwards compatibility
        target.vm_soma = suite.traces["vm15"]
    opt.seeded_current = target_current["value"]
    opt.spk_count = spk_count
    params = opt.attrs_to_params()

    target.seeded_current = target_current["value"]
    target.spk_count = spk_count
    target = inject_model_soma(
        target, solve_for_current=target_current["value"]
    )
    opt = inject_model_soma(opt, solve_for_current=target_current["value"])

    return opt, target, scores, obs_preds, df


''' Not used
def downsample(array, npts):
    interpolated = interp1d(
        np.arange(len(array)), array, axis=0, fill_value="extrapolate"
    )
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled
'''

def make_stim_waves_func():
    import allensdk.core.json_utilities as json_utilities
    import pickle

    neuronal_model_id = 566302806
    # download model metadata
    try:
        ephys_sweeps = json_utilities.read("ephys_sweeps.json")
    except:
        from allensdk.api.queries.glif_api import GlifApi

        glif_api = GlifApi()
        nm = glif_api.get_neuronal_models_by_id([neuronal_model_id])[0]

        from allensdk.core.cell_types_cache import CellTypesCache

        # download information about the cell
        ctc = CellTypesCache()
        ctc.get_ephys_data(nm["specimen_id"], file_name="stimulus.nwb")
        ctc.get_ephys_sweeps(nm["specimen_id"], file_name="ephys_sweeps.json")
        ephys_sweeps = json_utilities.read("ephys_sweeps.json")

    ephys_file_name = "stimulus.nwb"
    sweep_numbers = [
        s["sweep_number"] for s in ephys_sweeps if s["stimulus_units"] == "Amps"
    ]
    stimulus = [
        s
        for s in ephys_sweeps
        if s["stimulus_units"] == "Amps"
        if s["num_spikes"] != None
        if s["stimulus_name"] != "Ramp" and s["stimulus_name"] != "Short Square"
    ]
    amplitudes = [s["stimulus_absolute_amplitude"] for s in stimulus]
    durations = [s["stimulus_duration"] for s in stimulus]
    expeceted_spikes = [s["num_spikes"] for s in stimulus]
    delays = [s["stimulus_start_time"] for s in stimulus]
    sn = [s["sweep_number"] for s in stimulus]
    make_stim_waves = {}
    for i, j in enumerate(sn):
        make_stim_waves[j] = {}
        make_stim_waves[j]["amplitude"] = amplitudes[i]
        make_stim_waves[j]["delay"] = delays[i]
        make_stim_waves[j]["durations"] = durations[i]
        make_stim_waves[j]["expeceted_spikes"] = expeceted_spikes[i]
    pickle.dump(make_stim_waves, open("waves.p", "wb"))
    return make_stim_waves
