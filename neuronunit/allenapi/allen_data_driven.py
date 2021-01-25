from typing import Any, Dict, List, Optional, Tuple, Type, Union, Text

import pickle
import seaborn as sns
import os

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
    dtc = DataTC(backend=model_type, attrs=attrs)
    for t in nu_tests:
        if t.name == "Spikecount":
            spk_count = float(t.observation["mean"])
            break
    observation_range = {}
    observation_range["value"] = spk_count
    template_model.backend = model_type
    template_model.allen = True
    template_model.NU = True
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
    template_model.seeded_current = target_current["value"]

    return template_model, suite, nu_tests, target_current, spk_count


def wrap_setups(
    specimen_id,
    model_type,
    target_num_spikes,
    template_model=None,
    fixed_current=False,
    cached=False,
    score_type=ZScore,
    efel_filter_iterable=None,
):
    """
    if os.path.isfile("325479788later_allen_NU_tests.p"):
        template_model, suite, nu_tests, target_current, spk_count = opt_setup(
            specimen_id,
            model_type,
            target_num_spikes,
            template_model=template_model,
            fixed_current=False,
            cached=False,
            score_type=score_type,
            efel_filter_iterable=efel_filter_iterable
        )
    else:
    """
    template_model, suite, nu_tests, target_current, spk_count = opt_setup(
        specimen_id,
        model_type,
        target_num_spikes,
        template_model=template_model,
        fixed_current=False,
        cached=None,
        score_type=score_type,
        efel_filter_iterable=efel_filter_iterable,
    )
    template_model.seeded_current = target_current["value"]
    template_model.allen = True
    template_model.seeded_current
    template_model.NU = True
    template_model.backend = model_type
    template_model.efel_filter_iterable = efel_filter_iterable

    cell_evaluator, template_model = opt_setup_two(
        model_type,
        suite,
        nu_tests,
        target_current,
        spk_count,
        template_model=template_model,
        score_type=score_type,
        efel_filter_iterable=efel_filter_iterable,
    )
    cell_evaluator.cell_model.params = BPO_PARAMS[model_type]
    assert cell_evaluator.cell_model is not None
    cell_evaluator.suite = None
    cell_evaluator.suite = suite
    return cell_evaluator, template_model, suite, target_current, spk_count


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
                if score_gene.log_norm_score is not None:
                    delta = np.abs(float(score_gene.log_norm_score))
                else:
                    delta = 1000.0
            else:
                delta = 1000.0
            if np.nan == delta or delta == np.inf:
                delta = np.abs(float(score_gene.raw))
            if np.nan == delta or delta == np.inf:
                delta = 1000.0
            # if delta == 1000.0:
            #    print(self.test.name)

            return delta


def opt_setup_two(
    model_type,
    suite,
    nu_tests,
    target_current,
    spk_count,
    template_model=None,
    score_type=ZScore,
    efel_filter_iterable=None,
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


def opt_exec(MU, NGEN, mapping_funct, cell_evaluator, mutpb=0.05, cxpb=0.6):

    optimisation = bpop.optimisations.DEAPOptimisation(
        evaluator=cell_evaluator,
        offspring_size=MU,
        eta=25,
        map_function=map,
        selector_name="IBEA",
        mutpb=mutpb,
        cxpb=cxpb,
        ELITISM=True,
        NEURONUNIT=True,
    )
    final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=NGEN)
    return final_pop, hall_of_fame, logs, hist


def opt_to_model(hall_of_fame, cell_evaluator, suite, target_current, spk_count):
    best_ind = hall_of_fame[0]
    model = cell_evaluator.cell_model
    tests = cell_evaluator.suite.tests
    scores = []
    obs_preds = []

    for t in tests:
        scores.append(t.judge(model, prediction=t.prediction))
        obs_preds.append(
            (t.name, t.observation["mean"], t.prediction["mean"], scores[-1])
        )
    df = pd.DataFrame(obs_preds)

    opt = model.model_to_dtc()
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
    opt = opt.attrs_to_params()

    target.seeded_current = target_current["value"]
    target.spk_count = spk_count
    _, _, _, _, target = inject_model_soma(
        target, solve_for_current=target_current["value"]
    )
    _, _, _, _, opt = inject_model_soma(opt, solve_for_current=target_current["value"])

    return opt, target, scores, obs_preds, df


def make_allen_hard_coded_complete():
    """
    Manually specificy 4-5
    different passive/static electrical properties
    over 4 Allen specimen id's.
    FITest
    623960880
    623893177
    471819401
    482493761
    """
    from neuronunit.optimization.optimization_management import TSD

    ##
    # 623960880
    ##
    rt = RheobaseTest(observation={"mean": 70 * qt.pA, "std": 70 * qt.pA})  # yes
    tc = TimeConstantTest(observation={"mean": 23.8 * qt.ms, "std": 23.8 * qt.ms})
    ir = InputResistanceTest(observation={"mean": 241 * qt.MOhm, "std": 241 * qt.MOhm})
    rp = RestingPotentialTest(observation={"mean": -65.1 * qt.mV, "std": 65.1 * qt.mV})

    # capacitance = ((tc.observation['mean']))/((ir.observation['mean']))#*qt.pF

    # ct = CapacitanceTest(observation={'mean':capacitance,'std':capacitance})
    fislope = FITest(
        observation={"value": 0.18 * (pq.Hz / pq.pA), "mean": 0.18 * (pq.Hz / pq.pA)}
    )
    fislope.score_type = RelativeDifferenceScore

    allen_tests = [tc, rp, ir, rt]
    for t in allen_tests:
        t.score_type = RelativeDifferenceScore
    allen_suite_623960880 = TestSuite(allen_tests)
    allen_suite_623960880.name = (
        "http://celltypes.brain-map.org/mouse/experiment/electrophysiology/623960880"
    )

    ##
    # ID	623893177
    ##
    rt = RheobaseTest(observation={"mean": 190 * qt.pA, "std": 190 * qt.pA})  # yes
    tc = TimeConstantTest(observation={"mean": 27.8 * qt.ms, "std": 27.8 * qt.ms})
    ir = InputResistanceTest(observation={"mean": 136 * qt.MOhm, "std": 136 * qt.MOhm})
    rp = RestingPotentialTest(observation={"mean": -77.0 * qt.mV, "std": 77.0 * qt.mV})

    capacitance = ((tc.observation["mean"])) / ((ir.observation["mean"]))  # *qt.pF

    ct = CapacitanceTest(observation={"mean": capacitance, "std": capacitance})

    ##
    fislope = FITest(
        observation={"value": 0.12 * (pq.Hz / pq.pA), "mean": 0.12 * (pq.Hz / pq.pA)}
    )
    fislope.score_type = RelativeDifferenceScore
    ##

    allen_tests = [tc, rp, ir, rt]
    for t in allen_tests:
        t.score_type = RelativeDifferenceScore
    allen_suite_623893177 = TestSuite(allen_tests)
    allen_suite_623893177.name = (
        "http://celltypes.brain-map.org/mouse/experiment/electrophysiology/623893177"
    )
    cells = {}
    cells["623960880"] = TSD(allen_suite_623960880)
    cells["623893177"] = TSD(allen_suite_623893177)

    ##
    # 482493761
    ##

    rt = RheobaseTest(observation={"mean": 70 * qt.pA, "std": 70 * qt.pA})  # yes
    tc = TimeConstantTest(observation={"mean": 24.4 * qt.ms, "std": 24.4 * qt.ms})
    ir = InputResistanceTest(observation={"mean": 132 * qt.MOhm, "std": 132 * qt.MOhm})
    rp = RestingPotentialTest(observation={"mean": -71.6 * qt.mV, "std": 71.6 * qt.mV})

    fislope = FITest(
        observation={"value": 0.09 * (pq.Hz / pq.pA), "mean": 0.09 * (pq.Hz / pq.pA)}
    )
    fislope.score_type = RelativeDifferenceScore

    allen_tests = [rt, tc, rp, ir]
    for t in allen_tests:
        t.score_type = RelativeDifferenceScore
    # allen_tests[-1].score_type = ZScore
    allen_suite482493761 = TestSuite(allen_tests)
    allen_suite482493761.name = (
        "http://celltypes.brain-map.org/mouse/experiment/electrophysiology/482493761"
    )
    ##
    # 471819401
    ##
    rt = RheobaseTest(observation={"mean": 190 * qt.pA, "std": 190 * qt.pA})  # yes
    tc = TimeConstantTest(observation={"mean": 13.8 * qt.ms, "std": 13.8 * qt.ms})
    ir = InputResistanceTest(observation={"mean": 132 * qt.MOhm, "std": 132 * qt.MOhm})
    rp = RestingPotentialTest(observation={"mean": -77.5 * qt.mV, "std": 77.5 * qt.mV})

    fislope = FITest(
        observation={"value": 0.18 * (pq.Hz / pq.pA), "mean": 0.18 * (pq.Hz / pq.pA)}
    )
    fislope.score_type = RelativeDifferenceScore

    # F/I Curve Slope	0.18
    allen_tests = [rt, tc, rp, ir]
    for t in allen_tests:
        t.score_type = RelativeDifferenceScore
    allen_suite471819401 = TestSuite(allen_tests)
    allen_suite471819401.name = (
        "http://celltypes.brain-map.org/mouse/experiment/electrophysiology/471819401"
    )
    list_of_dicts = []
    cells["471819401"] = TSD(allen_suite471819401)
    cells["482493761"] = TSD(allen_suite482493761)

    for k, v in cells.items():
        observations = {}
        for k1 in cells["623960880"].keys():
            vsd = TSD(v)
            if k1 in vsd.keys():
                vsd[k1].observation["mean"]

                observations[k1] = np.round(vsd[k1].observation["mean"], 2)
                observations["name"] = k
        list_of_dicts.append(observations)
    df = pd.DataFrame(list_of_dicts, index=cells.keys())
    df

    return (
        allen_suite_623960880,
        allen_suite_623893177,
        allen_suite471819401,
        allen_suite482493761,
        cells,
        df,
    )


def make_allen_hard_coded_limited():
    # hard/hand code ephys constraints
    from quantities import Hz, pA
    from neuronunit.optimization.optimization_management import TSD
    import pandas as pd

    rt = RheobaseTest(observation={"mean": 70 * qt.pA, "std": 70 * qt.pA})
    tc = TimeConstantTest(observation={"mean": 24.4 * qt.ms, "std": 24.4 * qt.ms})
    ir = InputResistanceTest(observation={"mean": 132 * qt.MOhm, "std": 132 * qt.MOhm})
    rp = RestingPotentialTest(observation={"mean": -71.6 * qt.mV, "std": 77.5 * qt.mV})

    allen_tests = [rt, tc, rp, ir]
    for t in allen_tests:
        t.score_type = RelativeDifferenceScore
    allen_suite482493761 = TestSuite(allen_tests)
    allen_suite482493761.name = (
        "http://celltypes.brain-map.org/mouse/experiment/electrophysiology/482493761"
    )
    rt = RheobaseTest(observation={"mean": 190 * qt.pA, "std": 190 * qt.pA})
    tc = TimeConstantTest(observation={"mean": 13.8 * qt.ms, "std": 13.8 * qt.ms})
    ir = InputResistanceTest(observation={"mean": 132 * qt.MOhm, "std": 132 * qt.MOhm})
    rp = RestingPotentialTest(observation={"mean": -77.5 * qt.mV, "std": 77.5 * qt.mV})
    fi = FITest(observation={"mean": 0.09 * Hz / pA})  #'std':77.5*qt.mV})

    allen_tests = [rt, tc, rp, ir]  # ,fi]
    for t in allen_tests:
        t.score_type = RelativeDifferenceScore
    # allen_tests[-1].score_type = ZScore
    allen_suite471819401 = TestSuite(allen_tests)
    allen_suite471819401.name = (
        "http://celltypes.brain-map.org/mouse/experiment/electrophysiology/471819401"
    )
    list_of_dicts = []
    cells = {}
    cells["471819401"] = TSD(allen_suite471819401)
    cells["482493761"] = TSD(allen_suite482493761)

    allen_tests = [rt, fi]
    for t in allen_tests:
        t.score_type = RelativeDifferenceScore
    allen_suite3 = TestSuite(allen_tests)

    cells["fi_curve"] = TSD(allen_suite3)

    for k, v in cells.items():
        observations = {}
        for k1 in cells["482493761"].keys():
            vsd = TSD(v)
            if k1 in vsd.keys():
                vsd[k1].observation["mean"]

                observations[k1] = np.round(vsd[k1].observation["mean"], 2)
                observations["name"] = k
        list_of_dicts.append(observations)
    df = pd.DataFrame(list_of_dicts)
    return allen_suite471819401, allen_suite482493761, df, cells
