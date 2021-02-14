import dask


def dask_map_function(eval_, invalid_ind):
    results = []
    for x in invalid_ind:
        y = dask.delayed(eval_)(x)
        results.append(y)
    fitnesses = dask.compute(*results)
    return fitnesses


from sciunit.scores import ZScore

# from sciunit import TestSuite
from sciunit.scores.collections import ScoreArray
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from neuronunit.optimization.optimization_management import (
    model_to_rheo,
    switch_logic,
    active_values,
)
from neuronunit.tests.base import AMPL, DELAY, DURATION
import quantities as pq

PASSIVE_DURATION = 500.0 * pq.ms
PASSIVE_DELAY = 200.0 * pq.ms
import sciunit
import numpy as np
from bluepyopt.parameters import Parameter

from neuronunit.optimization.optimization_management import TSD

import numpy as np
import matplotlib.pyplot as plt


def glif_specific_modifications(tests):

    """
    Now appropriate for all tests
    """
    tests = TSD(tests)
    # tests.pop('RheobaseTest',None)
    tests.pop("InjectedCurrentAPAmplitudeTest", None)
    tests.pop("InjectedCurrentAPThresholdTest", None)
    tests.pop("InjectedCurrentAPWidthTest", None)
    # tests.pop('InputResistanceTest',None)
    # tests.pop('CapacitanceTest',None)
    # tests.pop('TimeConstantTest',None)

    return tests


def l5pc_specific_modifications(tests):
    tests = TSD(tests)
    tests.pop("InputResistanceTest", None)
    tests.pop("CapacitanceTest", None)
    tests.pop("TimeConstantTest", None)
    # tests.pop('RestingPotentialTest',None)

    return tests


import streamlit as st


def make_evaluator(
    nu_tests,
    MODEL_PARAMS,
    experiment=str("Neocortex pyramidal cell layer 5-6"),
    model=str("ADEXP"),
):
    objectives = []

    nu_tests[0].score_type = RelativeDifferenceScore

    if "GLIF" in model:
        nu_tests_ = glif_specific_modifications(nu_tests)
        nu_tests = list(nu_tests_.values())
        simple_cell.name = "GLIF"

    elif "L5PC" in model:
        nu_tests_ = l5pc_specific_modifications(nu_tests)
        nu_tests = list(nu_tests_.values())
        simple_cell.name = "L5PC"

    else:
        simple_cell.name = model + experiment
    simple_cell.backend = model
    simple_cell.params = {k: np.mean(v) for k, v in simple_cell.params.items()}

    lop = {}
    for k, v in MODEL_PARAMS[model].items():
        p = Parameter(name=k, bounds=v, frozen=False)
        lop[k] = p

    simple_cell.params = lop

    for tt in nu_tests:
        feature_name = tt.name
        ft = NUFeature_standard_suite(tt, simple_cell)
        objective = ephys.objectives.SingletonObjective(feature_name, ft)
        objectives.append(objective)

    score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives)

    sweep_protocols = []
    for protocol_name, amplitude in [("step1", 0.05)]:
        protocol = ephys.protocols.SweepProtocol(protocol_name, [None], [None])
        sweep_protocols.append(protocol)
    twostep_protocol = ephys.protocols.SequenceProtocol(
        "twostep", protocols=sweep_protocols
    )

    cell_evaluator = ephys.evaluators.CellEvaluator(
        cell_model=simple_cell,
        param_names=MODEL_PARAMS[model].keys(),
        fitness_protocols={twostep_protocol.name: twostep_protocol},
        fitness_calculator=score_calc,
        sim="euler",
    )
    simple_cell.params_by_names(MODEL_PARAMS[model].keys())
    return cell_evaluator, simple_cell, score_calc, [tt.name for tt in nu_tests]


def trace_explore_widget(optimal_model_params=None):
    """
    move this to utils file.
    Allow app user to explore model behavior around the optimal,
    by panning across parameters and then viewing resulting spike shapes.
    """

    attrs = {k: np.mean(v) for k, v in MODEL_PARAMS["IZHI"].items()}
    plt.clf()
    cnt = 0
    slider_value = st.slider(
        "parameter a", min_value=0.01, max_value=0.1, value=0.05, step=0.001
    )
    if optimal_model_params is None:
        dtc = DataTC(backend="IZHI", attrs=attrs)
    else:
        dtc = DataTC(backend="IZHI", attrs=optimal_model_params)
    dtc.attrs["a"] = slider_value
    dtc = model_to_mode(dtc)
    temp_rh = dtc.rheobase
    model = dtc.dtc_to_model()
    model.attrs = model._backend.default_attrs
    model.attrs.update(dtc.attrs)

    uc = {"amplitude": temp_rh, "duration": DURATION, "delay": DELAY}
    model._backend.inject_square_current(uc)
    vm = model.get_membrane_potential()
    plt.plot(vm.times, vm.magnitude)

    cnt += 1
    st.pyplot()


def basic_expVar(trace1, trace2):
    # https://github.com/AllenInstitute/GLIF_Teeter_et_al_2018/blob/master/query_biophys/query_biophys_expVar.py
    """This is the fundamental calculation that is used in all different types of explained variation.
    At a basic level, the explained variance is calculated between two traces.  These traces can be PSTH's
    or single spike trains that have been convolved with a kernel (in this case always a Gaussian)
    Input:
        trace 1 & 2:  1D numpy array containing values of the trace.  (This function requires numpy array
                        to ensure that this is not a multidemensional list.)
    Returns:
        expVar:  float value of explained variance
    """

    var_trace1 = np.var(trace1)
    var_trace2 = np.var(trace2)
    var_trace1_minus_trace2 = np.var(trace1 - trace2)

    if var_trace1_minus_trace2 == 0.0:
        return 1.0
    else:
        return (var_trace1 + var_trace2 - var_trace1_minus_trace2) / (
            var_trace1 + var_trace2
        )


def hof_to_euclid(hof, MODEL_PARAMS, target):
    lengths = {}
    tv = 1
    cnt = 0
    constellation0 = hof[0]
    constellation1 = hof[1]
    subset = list(MODEL_PARAMS.keys())
    tg = target.dtc_to_gene(subset_params=subset)
    if len(MODEL_PARAMS) == 1:

        ax = plt.subplot()
        for k, v in MODEL_PARAMS.items():
            lengths[k] = np.abs(np.abs(v[1]) - np.abs(v[0]))

            x = [h[cnt] for h in hof]
            y = [np.sum(h.fitness.values()) for h in hof]
            ax.set_xlim(v[0], v[1])
            ax.set_xlabel(k)
            tgene = tg[cnt]
            yg = 0

        ax.scatter(x, y, c="b", marker="o", label="samples")
        ax.scatter(tgene, yg, c="r", marker="*", label="target")
        ax.legend()

        plt.show()

    if len(MODEL_PARAMS) == 2:

        ax = plt.subplot()
        for k, v in MODEL_PARAMS.items():
            lengths[k] = np.abs(np.abs(v[1]) - np.abs(v[0]))

            if cnt == 0:
                tgenex = tg[cnt]
                x = [h[cnt] for h in hof]
                ax.set_xlim(v[0], v[1])
                ax.set_xlabel(k)
            if cnt == 1:
                tgeney = tg[cnt]

                y = [h[cnt] for h in hof]
                ax.set_ylim(v[0], v[1])
                ax.set_ylabel(k)
            cnt += 1

        ax.scatter(x, y, c="r", marker="o", label="samples", s=5)
        ax.scatter(tgenex, tgeney, c="b", marker="*", label="target", s=11)

        ax.legend()

        plt.sgow()
    # print(len(MODEL_PARAMS))
    if len(MODEL_PARAMS) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for k, v in MODEL_PARAMS.items():
            lengths[k] = np.abs(np.abs(v[1]) - np.abs(v[0]))

            if cnt == 0:
                tgenex = tg[cnt]

                x = [h[cnt] for h in hof]
                ax.set_xlim(v[0], v[1])
                ax.set_xlabel(k)
            if cnt == 1:
                tgeney = tg[cnt]

                y = [h[cnt] for h in hof]
                ax.set_ylim(v[0], v[1])
                ax.set_ylabel(k)
            if cnt == 2:
                tgenez = tg[cnt]

                z = [h[cnt] for h in hof]
                ax.set_zlim(v[0], v[1])
                ax.set_zlabel(k)

            cnt += 1
        ax.scatter(x, y, z, c="r", marker="o")
        ax.scatter(tgenex, tgeney, tgenez, c="b", marker="*", label="target", s=11)

        plt.show()


def initialise_test(v, rheobase):
    v = switch_logic([v])
    v = v[0]
    k = v.name
    if not hasattr(v, "params"):
        v.params = {}
    if not "injected_square_current" in v.params.keys():
        v.params["injected_square_current"] = {}
    if v.passive == False and v.active == True:
        keyed = v.params["injected_square_current"]
        v.params = active_values(keyed, rheobase)
        v.params["injected_square_current"]["delay"] = DELAY
        v.params["injected_square_current"]["duration"] = DURATION
    if v.passive == True and v.active == False:

        v.params["injected_square_current"]["amplitude"] = -10 * pq.pA
        v.params["injected_square_current"]["delay"] = PASSIVE_DELAY
        v.params["injected_square_current"]["duration"] = PASSIVE_DURATION

    if v.name in str("RestingPotentialTest"):
        v.params["injected_square_current"]["delay"] = PASSIVE_DELAY
        v.params["injected_square_current"]["duration"] = PASSIVE_DURATION
        v.params["injected_square_current"]["amplitude"] = 0.0 * pq.pA

    return v


from sciunit.scores import ZScore
import bluepyopt as bpop
import bluepyopt.ephys as ephys


class NUFeature_standard_suite(object):
    def __init__(self, test, model):
        self.test = test
        self.model = model

    def calculate_score(self, responses):
        model = responses["model"].dtc_to_model()
        model.attrs = responses["params"]
        self.test = initialise_test(self.test, responses["rheobase"])
        if "RheobaseTest" in str(self.test.name):
            self.test.score_type = ZScore
            prediction = {"value": responses["rheobase"]}
            score_gene = self.test.compute_score(self.test.observation, prediction)
            # lns = np.abs(np.float(score_gene.log_norm_score))
            # return lns
        else:
            try:
                score_gene = self.test.judge(model)
            except:
                # print(self.test.observation,self.test.name)
                # print(score_gene,'\n\n\n')

                return 100.0

        if not isinstance(type(score_gene), type(None)):
            if not isinstance(score_gene, sciunit.scores.InsufficientDataScore):
                if not isinstance(type(score_gene.log_norm_score), type(None)):
                    try:

                        lns = np.abs(np.float(score_gene.log_norm_score))
                    except:
                        # works 1/2 time that log_norm_score does not work
                        # more informative than nominal bad score 100
                        lns = np.abs(np.float(score_gene.raw))
                else:
                    # works 1/2 time that log_norm_score does not work
                    # more informative than nominal bad score 100

                    lns = np.abs(np.float(score_gene.raw))
            else:
                # print(prediction,self.test.observation)
                # print(score_gene,'\n\n\n')
                lns = 100
        if lns == np.inf:
            lns = np.abs(np.float(score_gene.raw))
        # print(lns,self.test.name)
        return lns
