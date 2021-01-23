# Its not that this file is responsible for doing plotting,
# but it calls many modules that are, such that it needs to pre-empt
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Text

import dask
from tqdm import tqdm
import warnings

SILENT = True
if SILENT:
    warnings.filterwarnings("ignore")
import matplotlib


import cython
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.model_selection import ParameterGrid
from collections import OrderedDict
from collections.abc import Iterable

import math

# import quantities as pq
import numpy
import deap
from deap import creator
from deap import base
import array
import copy
from frozendict import frozendict
from itertools import repeat
import seaborn as sns

sns.set(context="paper", font="monospace")
import random

# import pandas as pd

import bluepyopt as bpop
import bluepyopt.ephys as ephys
from bluepyopt.parameters import Parameter
import quantities as pq

PASSIVE_DURATION = 500.0 * pq.ms
PASSIVE_DELAY = 200.0 * pq.ms
import plotly.graph_objects as go
from elephant.spike_train_generation import threshold_detection


from neuronunit.optimization.data_transport_container import DataTC

from neuronunit.tests.base import AMPL, DELAY, DURATION
from neuronunit.tests.target_spike_current import (
    SpikeCountSearch,
    SpikeCountRangeSearch,
)
import neuronunit.capabilities.spike_functions as sf
from neuronunit.optimization.model_parameters import MODEL_PARAMS, BPO_PARAMS

from neuronunit.tests.fi import RheobaseTest
from neuronunit.capabilities.spike_functions import get_spike_waveforms, spikes2widths

from jithub.models import model_classes

# import time


from sciunit.scores.collections import ScoreArray
import sciunit
from sciunit import TestSuite
from sciunit import scores
from sciunit.scores import RelativeDifferenceScore


try:
    import plotly.offline as py
except:
    warnings.warn("plotly")
try:
    import plotly

    plotly.io.orca.config.executable = "/usr/bin/orca"
except:
    print("silently fail on plotly")


try:
    import efel
except:
    warnings.warn("Blue brain feature extraction not available, consider installing")
try:
    import seaborn as sns
except:
    warnings.warn("Seaborne plotting sub library not available, consider installing")
try:
    from sklearn.cluster import KMeans
except:
    warnings.warn("SKLearn library not available, consider installing")

class TSD(dict):
    """
    -- Synopsis:
    Test Suite Dictionary class
    A container for sciunit tests, Indexable by dictionary keys.
    Contains a method called optimize.
    """

    def __init__(self, tests={}, use_rheobase_score=True):
        self.DO = None
        self.use_rheobase_score = use_rheobase_score
        self.backend = None
        self.three_step = None
        if type(tests) is TestSuite:
            tests = OrderedDict({t.name: t for t in tests.tests})
        if type(tests) is type(dict()):
            pass
        if type(tests) is type(list()):
            tests = OrderedDict({t.name: t for t in tests})
        super(TSD, self).__init__()
        self.update(tests)

    def display(self):
        from IPython.display import display

        if hasattr(self, "ga_out"):
            return display(self.ga_out["pf"][0].dtc.obs_preds)
        else:
            return None

    def to_pickleable_dict(self):
        """
        -- Synopsis:
        # A pickleable version of instance object.
        # https://joblib.readthedocs.io/en/latest/

        # This might work joblib.dump(self, filename + '.compressed', compress=True)
        # somewhere in job lib there are tools for pickling more complex objects
        # including simulation results.
        """
        del self.ga_out.DO
        del self.DO
        return {k: v for k, v in self.items()}


    def optimize(self,**kwargs):
        import shelve
        defaults = {'param_edges':None,
                    'backend':None,\
                    'protocol':{'allen': False, 'elephant': True},\
                    'MU':5,\
                    'NGEN':5,\
                    'free_parameters':None,\
                    'seed_pop':None,\
                    'hold_constant':None,
                    'plot':False,'figname':None,
                    'use_rheobase_score':self.use_rheobase_score,
                    'ignore_cached':False
                    }
        defaults.update(kwargs)
        kwargs = defaults
        d = shelve.open('opt_models_cache')  # open -- file may get suffix added by low-level
        query_key = str(kwargs['NGEN']) +\
        str(kwargs['free_parameters']) +\
        str(kwargs['backend']) +\
        str(kwargs['MU']) +\
        str(kwargs['protocol']) +\
        str(kwargs['hold_constant'])
        flag = query_key in d

        if flag and not kwargs['ignore_cached']:
            ###
            # Hack
            ###
            #creator.create("FitnessMin", base.Fitness, weights=tuple(-1.0 for i in range(0,10)))
            #creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
            ###
            # End hack
            ###
            ga_out = d[query_key]

            d.close()
            del d
            return ga_out
        else:
            d.close()
            del d
            if type(kwargs['param_edges']) is type(None):
                from neuronunit.optimization import model_parameters
                param_edges = model_parameters.MODEL_PARAMS[kwargs['backend']]
            if type(kwargs['free_parameters']) is type(None):
                if type(kwargs['param_edges']) is not type(None):
                    free_parameters=kwargs['param_edges'].keys()
                else:
                    from neuronunit.optimization import model_parameters
                    free_parameters = model_parameters.MODEL_PARAMS[kwargs['backend']].keys()
            else:
                free_parameters = kwargs['free_parameters']

            if kwargs['hold_constant'] is None:
                if len(free_parameters) < len(param_edges):
                    pass
            '''
            PROBABLY REWRITE THIS

            self.DO = make_ga_DO(param_edges, \
                                kwargs['NGEN'], \
                                self, \
                                free_parameters=free_parameters, \
                                backend=kwargs['backend'], \
                                MU = kwargs['MU'], \
                                protocol=kwargs['protocol'],
                                seed_pop = kwargs['seed_pop'], \
                                hc=kwargs['hold_constant']
                                )
            '''
            self.MU = self.DO.MU = kwargs['MU']
            self.NGEN = self.DO.NGEN = kwargs['NGEN']

            ga_out = self.DO.run(NGEN = self.DO.NGEN)
            self.backend = kwargs['backend']
            return ga_out


import numpy, time


@cython.boundscheck(False)
@cython.wraparound(False)
def random_p(model_type):
    ranges = MODEL_PARAMS[model_type]
    date_int = int(time.time())
    numpy.random.seed(date_int)
    random_param1 = {}  # randomly sample a point in the viable parameter space.
    for k in ranges.keys():
        sample = random.uniform(ranges[k][0], ranges[k][1])
        random_param1[k] = sample
    return random_param1


@cython.boundscheck(False)
@cython.wraparound(False)
def process_rparam(backend, free_parameters):
    random_param = random_p(backend)
    if "GLIF" in str(backend):
        random_param["init_AScurrents"] = [0.0, 0.0]
        random_param["asc_tau_array"] = [0.3333333333333333, 0.01]
        rp = random_param
    else:
        random_param.pop("Iext", None)
        rp = random_param
    if free_parameters is not None:
        reduced_parameter_set = {}
        for k in free_parameters:
            reduced_parameter_set[k] = rp[k]
        rp = reduced_parameter_set
    dsolution = DataTC(backend=backend, attrs=rp)
    temp_model = dsolution.dtc_to_model()
    dsolution.attrs = temp_model.default_attrs
    dsolution.attrs.update(rp)
    return dsolution, rp, None, random_param


def write_models_for_nml_db(dtc):
    with open(str(list(dtc.attrs.values())) + ".csv", "w") as writeFile:
        df = pd.DataFrame([dtc.attrs])
        writer = csv.writer(writeFile)
        writer.writerows(df)


def write_opt_to_nml(path, param_dict) -> None:
    """
    -- Inputs: desired file path, model parameters to encode in NeuroML2
    -- Outputs: NeuroML2 file.
    -- Synopsis: Write optimimal simulation parameters back to NeuroML2.
    """
    more_attributes = pynml.read_lems_file(
        orig_lems_file_path, include_includes=True, debug=False
    )
    for i in more_attributes.components:
        new = {}
        if str("izhikevich2007Cell") in i.type:
            for k, v in i.parameters.items():
                units = v.split()
                if len(units) == 2:
                    units = units[1]
                else:
                    units = "mV"
                new[k] = str(param_dict[k]) + str(" ") + str(units)
            i.parameters = new
    fopen = open(path + ".nml", "w")
    more_attributes.export_to_file(fopen)
    fopen.close()
    return



def get_rh(dtc:DataTC,
            rtest_class:RheobaseTest,
            bind_vm:bool=False) -> DataTC:
    """
    --args:
        :param object dtc:
        :param object Rheobase Test Class:
    :-- returns: object dtc:
    -- Synopsis: This is used to recover/produce
     a rheobase test class instance,
     given unknown experimental observations.
    """
    place_holder = {"mean": 10 * pq.pA}
    backend_ = dtc.backend
    rtest = RheobaseTest(observation=place_holder, name="RheobaseTest")
    rtest.score_type = RelativeDifferenceScore
    dtc.rheobase = None
    assert len(dtc.attrs)
    model = dtc.dtc_to_model()
    rtest.params["injected_square_current"] = {}
    rtest.params["injected_square_current"]["delay"] = DELAY
    rtest.params["injected_square_current"]["duration"] = DURATION
    dtc.rheobase = rtest.generate_prediction(model)["value"]
    temp_vm = model.get_membrane_potential()
    if bind_vm:
        dtc.vmrh = temp_vm
    if np.isnan(np.min(temp_vm)):
        # rheobase exists but the waveform is nuts.
        # this is the fastest way to filter out a gene
        dtc.rheobase = None
    return dtc


def get_new_rtest(dtc:DataTC)->RheobaseTest:
    place_holder = {"mean": 10 * pq.pA}
    f = RheobaseTest
    rtest = f(observation=place_holder, name="RheobaseTest")
    rtest.score_type = RelativeDifferenceScore
    return rtest


def get_rtest(DataTC:dtc)->RheobaseTest:
    if not hasattr(dtc, "tests"):
        rtest = get_new_rtest(dtc)
    else:
        if type(dtc.tests) is type(list()):
            rtests = [t for t in dtc.tests if "rheo" in t.name.lower()]
        else:
            rtests = [v for k, v in dtc.tests.items() if "rheo" in str(k).lower()]

        if len(rtests):
            rtest = rtests[0]

        else:
            rtest = get_new_rtest(dtc)
    return rtest


def dtc_to_rheo(dtc: DataTC, bind_vm: bool = False) -> DataTC:
    """
    --Synopsis: If  test taking data, and objects are present (observations etc).
    Take the rheobase test and store it in the data transport container.
    """

    if hasattr(dtc, "tests"):
        if type(dtc.tests) is type({}) and str("RheobaseTest") in dtc.tests.keys():
            rtest = dtc.tests["RheobaseTest"]
        else:
            rtest = get_rtest(dtc)
    else:
        rtest = get_rtest(dtc)

    if rtest is not None:
        model = dtc.dtc_to_model()
        if dtc.attrs is not None:
            model.attrs = dtc.attrs
        if isinstance(rtest, Iterable):
            rtest = rtest[0]
        dtc.rheobase = rtest.generate_prediction(model)["value"]
        temp_vm = model.get_membrane_potential()
        min = np.min(temp_vm)
        if np.isnan(temp_vm.any()):
            dtc.rheobase = None
        if bind_vm:
            dtc.vmrh = temp_vm
            # rheobase does exist but lets filter out this bad gene.
        return dtc
    else:
        # otherwise, if no observation is available, or if rheobase test score is not desired.
        # Just generate rheobase predictions, giving the models the freedom of rheobase
        # discovery without test taking.
        dtc = get_rh(dtc, rtest, bind_vm=bind_vm)
        if bind_vm:
            dtc.vmrh = temp_vm
    return dtc


def basic_expVar(trace1, trace2):
    """
    https://github.com/AllenInstitute/GLIF_Teeter_et_al_2018/blob/master/query_biophys/query_biophys_expVar.py
    --Synopsis: This is the fundamental calculation that is used in all different types of explained variation.
    At a basic level, the explained variance is calculated between two traces.  These traces can be PSTH's
    or single spike trains that have been convolved with a kernel (in this case always a Gaussian)
    --Args:
        trace 1 & 2:  1D numpy array containing values of the trace.  (This function requires numpy array
                        to ensure that this is not a multidemensional list.)
    --Returns:
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


"""
def jrt(use_tests,model_type,
        protocol={'elephant':True,'allen':False})->OptMan:
    use_tests = TSD(use_tests)
    use_tests.use_rheobase_score = True
    edges = model_parameters.MODEL_PARAMS[model_type]
    OM = OptMan(use_tests,
        backend=model_type,
        boundary_dict=edges,
        protocol=protocol)
    return OM
"""


def train_length(dtc: DataTC) -> DataTC:
    if not hasattr(dtc, "everything"):
        dtc.everything = {}
    vm = copy.copy(dtc.vm_soma)
    train_len = float(len(sf.get_spike_train(vm)))
    dtc.everything["Spikecount_1.5x"] = copy.copy(train_len)
    return dtc


def three_step_protocol(dtc, solve_for_current=None):
    """
    Rename this to multi spiking feature extraction
    """
    if solve_for_current is None:
        _, _, _, _, dtc = inject_model_soma(dtc)
        if dtc.vm_soma is None:
            return dtc
        try:
            dtc = efel_evaluation(dtc)
        except:
            dtc.vm_soma = None
    else:
        _, _, _, dtc = inject_model_soma(dtc, solve_for_current=solve_for_current)
        if dtc.vm_soma is None:
            return dtc
        dtc = efel_evaluation(dtc)

    dtc = rekeyed(dtc)
    if dtc.everything is not None:
        dtc = train_length(dtc)
    return dtc


def rekeyed(dtc: Any = object()) -> Any:
    rekey = {}
    if hasattr(dtc, "allen_30"):
        for k, v in dtc.allen_30.items():
            rekey[str(k) + str("_3.0x")] = v
    if hasattr(dtc, "allen_15"):
        for k, v in dtc.allen_15.items():
            rekey[str(k) + str("_1.5x")] = v
    if hasattr(dtc, "efel_30"):
        for k, v in dtc.efel_30[0].items():
            rekey[str(k) + str("_3.0x")] = v
    if hasattr(dtc, "efel_15"):
        if dtc.efel_15 is not None:
            for k, v in dtc.efel_15[0].items():
                rekey[str(k) + str("_1.5x")] = v
        else:
            rekey = None
    dtc.everything = rekey
    return dtc


def constrain_ahp(vm_used: Any = object()) -> dict:
    efel.reset()
    efel.setThreshold(0)
    trace3 = {
        "T": [float(t) * 1000.0 for t in vm_used.times],
        "V": [float(v) for v in vm_used.magnitude],
    }
    DURATION = 1100 * pq.ms
    DELAY = 100 * pq.ms
    trace3["stim_end"] = [float(DELAY) + float(DURATION)]
    trace3["stim_start"] = [float(DELAY)]
    simple_yes_list = ["AHP_depth", "AHP_depth_abs", "AHP_depth_last"]
    results = efel.getMeanFeatureValues([trace3], simple_yes_list, raise_warnings=False)
    return results


def exclude_non_viable_deflections(responses: dict = {}) -> float:
    """
    Synopsis: reject waveforms that would otherwise score well but have
    unrealistically huge AHP
    """
    if responses["response"] is not None:
        vm = responses["response"]
        results = constrain_ahp(vm)
        results = results[0]
        if results["AHP_depth"] is None or np.abs(results["AHP_depth_abs"]) >= 80:
            return 1000.0
        if np.abs(results["AHP_depth"]) >= 105:
            return 1000.0
        if np.max(vm) >= 0:
            snippets = get_spike_waveforms(vm)
            widths = spikes2widths(snippets)
            spike_train = threshold_detection(vm, threshold=0 * pq.mV)
            if not len(spike_train):
                return 1000.0

            if (spike_train[0] + 2.5 * pq.ms) > vm.times[-1]:
                too_long = True
                return 1000.0
            if isinstance(widths, Iterable):
                for w in widths:
                    if w >= 3.5 * pq.ms:
                        return 1000.0
            else:
                width = widths
                if width >= 2.0 * pq.ms:
                    return 1000.0
            if float(vm[-1]) == np.nan or np.isnan(vm[-1]):
                return 1000.0
            if float(vm[-1]) >= 0.0:
                return 1000.0
            assert vm[-1] < 0 * pq.mV
        return 0


class NUFeature_standard_suite(object):
    def __init__(self, test, model):
        self.test = test
        self.model = model
        self.score_array = None

    def calculate_score(self, responses: dict = {}) -> float:
        dtc = responses["dtc"]
        model = dtc.dtc_to_model()
        model.attrs = responses["params"]
        self.test = initialise_test(self.test)
        if self.test.active and responses["dtc"].rheobase is not None:
            result = exclude_non_viable_deflections(responses)
            if result != 0:
                return result
        self.test.prediction = self.test.generate_prediction(model)
        if responses["rheobase"] is not None:
            if self.test.prediction is not None:
                score_gene = self.test.judge(
                    model, prediction=self.test.prediction, deep_error=True
                )
            else:
                return 1000.0
        else:
            return 1000.0
        if not isinstance(type(score_gene), type(None)):
            if not isinstance(score_gene, sciunit.scores.InsufficientDataScore):
                try:
                    if not isinstance(type(score_gene.log_norm_score), type(None)):
                        lns = np.abs(np.float(score_gene.log_norm_score))
                    else:
                        if not isinstance(type(score_gene.raw), type(None)):
                            # works 1/2 time that log_norm_score does not work
                            # more informative than nominal bad score 100
                            lns = np.abs(np.float(score_gene.raw))
                            # works 1/2 time that log_norm_score does not work
                            # more informative than nominal bad score 100

                except:
                    lns = 1000
            else:
                lns = 1000
        else:
            lns = 1000
        if lns == np.inf or lns == np.nan:
            lns = 1000

        return lns


def make_evaluator(
    nu_tests,
    PARAMS,
    experiment=str("Neocortex pyramidal cell layer 5-6"),
    model=str("IZHI"),
    score_type=RelativeDifferenceScore,
) -> Union[Any, Any, Any, List[Any]]:

    if type(nu_tests) is type(dict()):
        nu_tests = list(nu_tests.values())
    if model == "IZHI":
        simple_cell = model_classes.IzhiModel()
    if model == "MAT":
        simple_cell = model_classes.MATModel()
    if model == "ADEXP":
        simple_cell = model_classes.ADEXPModel()
    simple_cell.params = PARAMS[model]
    simple_cell.NU = True
    simple_cell.name = model + experiment
    objectives = []
    for tt in nu_tests:
        feature_name = tt.name
        tt.score_type = score_type
        ft = NUFeature_standard_suite(tt, simple_cell)
        objective = ephys.objectives.SingletonObjective(feature_name, ft)
        objectives.append(objective)

    score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives)
    # cell_evaluator.SA = ScoreArray(nu_tests, objectives)

    sweep_protocols = []
    protocol = ephys.protocols.SweepProtocol("step1", [None], [None])
    sweep_protocols.append(protocol)
    onestep_protocol = ephys.protocols.SequenceProtocol(
        "onestep", protocols=sweep_protocols
    )
    cell_evaluator = ephys.evaluators.CellEvaluator(
        cell_model=simple_cell,
        param_names=list(copy.copy(BPO_PARAMS)[model].keys()),
        fitness_protocols={onestep_protocol.name: onestep_protocol},
        fitness_calculator=score_calc,
        sim="euler",
    )

    simple_cell.params_by_names(copy.copy(BPO_PARAMS)[model].keys())
    return (cell_evaluator, simple_cell, score_calc, [tt.name for tt in nu_tests])


def get_binary_file_downloader_html(bin_file_path, file_label="File"):
    """
    Used with streamlit
    """
    with open(bin_file_path, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file_path)}">Download {file_label}</a>'
    return href


def _opt(
    constraints,
    PARAMS,
    test_key,
    model_value,
    MU,
    NGEN,
    diversity,
    full_test_list=None,
    use_streamlit=False,
    score_type=RelativeDifferenceScore,
):
    """
    used with streamlit
    """

    if type(constraints) is not type(list()):
        constraints = list(constraints.values())
    cell_evaluator, simple_cell, score_calc, test_names = make_evaluator(
        constraints, PARAMS, test_key, model=model_value, score_type=score_type
    )
    model_type = str("_best_fit_") + str(model_value) + "_" + str(test_key) + "_.p"
    mut = 0.05
    cxp = 0.4
    optimization = bpop.optimisations.DEAPOptimisation(
        evaluator=cell_evaluator,
        offspring_size=MU,
        map_function=map,
        selector_name=diversity,
        mutpb=mut,
        cxpb=cxp,
    )

    final_pop, hall_of_fame, logs, hist = optimization.run(max_ngen=NGEN)

    best_ind = hall_of_fame[0]
    best_ind_dict = cell_evaluator.param_dict(best_ind)
    model = cell_evaluator.cell_model
    cell_evaluator.param_dict(best_ind)
    model.attrs = {
        str(k): float(v) for k, v in cell_evaluator.param_dict(best_ind).items()
    }
    opt = model.model_to_dtc()
    opt.attrs = {
        str(k): float(v) for k, v in cell_evaluator.param_dict(best_ind).items()
    }
    best_fit_val = best_ind.fitness.values
    return final_pop, hall_of_fame, logs, hist, best_ind, best_fit_val, opt


def full_statistical_description(
    constraints,
    exp_cell,
    MODEL_PARAMS,
    test_key,
    model_value,
    MU,
    NGEN,
    diversity,
    full_test_list=None,
    use_streamlit=False,
    tf=None,
    dry_run=True,
):

    if type(constraints) is type(list()):
        constraints = TSD(constraints)

    if hasattr(constraints, "keys"):
        keys = constraints.keys()

    else:
        constraints_d = {}
        for t in constraints:
            constraints_d[t.name] = t
        constraints = constraints_d
        keys = constraints.keys()

    final_pop, hall_of_fame, logs, hist, opt, obs_preds, chi_sqr_opt, p_value = _opt(
        constraints,
        MODEL_PARAMS,
        test_key,
        model_value,
        MU,
        NGEN,
        diversity,
        full_test_list=keys,
        use_streamlit=False,
    )
    temp = final_pop, hall_of_fame, logs, hist, opt, obs_preds, chi_sqr_opt, p_value
    opt_pickle = opt, obs_preds, chi_sqr_opt, p_value
    pickle.dump(
        opt_pickle, open(str(exp_cell) + str("_") + str(model_value) + "_.p", "wb")
    )

    gen_numbers = logs.select("gen")
    min_fitness = logs.select("min")
    max_fitness = logs.select("max")
    mean_fitness = logs.select("avg")
    final_pop, hall_of_fame, logs, hist, opt, obs_preds, chi_sqr_opt, p_value = temp
    if not dry_run:
        if tf is None:
            tf = open(str(exp_cell) + str(model_value) + str(".tex"), "w")

        tf.write("\subsubsection{" + str(exp_cell) + " " + str(model_value) + "}")
        tf.write(
            pd.DataFrame([{"chi_square": chi_sqr_opt, "p_value": p_value}]).T.to_latex()
        )
        best_params_frame = pd.DataFrame([opt.attrs])
        tf.write("optimal model parameters")
        tf.write(best_params_frame.to_latex())
        try:
            opt.make_pretty(opt.tests)

            df = opt.obs_preds
            tf.write(df.to_latex())
        except:
            tf.write("failed model")
        os.system("cat " + str(exp_cell) + str(model_value) + str(".tex"))
    results_dict = {}
    results_dict["model_value"] = model_value
    results_dict["exp_cell"] = exp_cell  # = {}
    results_dict["chi_square"] = chi_sqr_opt
    results_dict["p_value"] = p_value
    df = pd.DataFrame([results_dict])

    return df, results_dict, opt


def inject_model_soma(
    dtc: DataTC, figname=None, solve_for_current=None, fixed: bool = False
) -> Tuple[Any, Any, dict, Any, Any]:
    from neuronunit.tests.target_spike_current import SpikeCountSearch

    """
    -- args: dtc
    -- outputs: voltage at 3.0 rheobase,
                voltage at 1.5 rheobase,
                current Injection Parameters,
                dtc
    -- Synopsis:
     produce an rheobase injection value
     produce an object of class Neuronunit runnable Model
     with known attributes and known rheobase current injection value.
    """
    if type(solve_for_current) is not type(None):
        observation_range = {}
        model = dtc.dtc_to_model()
        temp = copy.copy(model.attrs)

        if not fixed:
            observation_range["value"] = dtc.spk_count
            scs = SpikeCountSearch(observation_range)
            target_current = scs.generate_prediction(model)
            if type(target_current) is not type(None):
                solve_for_current = target_current["value"]
            dtc.solve_for_current = solve_for_current
            ALLEN_DELAY = 1000.0 * pq.ms
            ALLEN_DURATION = 2000.0 * pq.ms
        uc = {
            "amplitude": solve_for_current,
            "duration": ALLEN_DURATION,
            "delay": ALLEN_DELAY,
        }
        model = dtc.dtc_to_model()
        model._backend.attrs = temp
        model.inject_square_current(**uc)
        if hasattr(dtc, "spikes"):
            dtc.spikes = model._backend.spikes
        vm15 = model.get_membrane_potential()
        dtc.vm_soma = copy.copy(vm15)
        del model
        return vm15, uc, None, dtc

    if dtc.rheobase is None:
        rt = RheobaseTest(observation={"mean": 0 * pq.pA})
        rt.score_type = RelativeDifferenceScore
        dtc.rheobase = rt.generate_prediction(dtc.dtc_to_model())
        if dtc.rheobase is None:
            return [None, None, None, None, dtc]
    model = dtc.dtc_to_model()
    if type(dtc.rheobase) is type(dict()):
        if dtc.rheobase["value"] is None:
            return [None, None, None, None, dtc]
        else:
            rheobase = dtc.rheobase["value"]
    else:
        rheobase = dtc.rheobase
    model = dtc.dtc_to_model()
    ALLEN_DELAY = 1000.0 * pq.ms
    ALLEN_DURATION = 2000.0 * pq.ms
    uc = {"amplitude": rheobase, "duration": ALLEN_DURATION, "delay": ALLEN_DELAY}
    model._backend.inject_square_current(**uc)
    dtc.vmrh = None
    dtc.vmrh = model.get_membrane_potential()
    del model
    model = dtc.dtc_to_model()
    ########
    # A thing to note
    # rheobase = 300 * pq.pA
    # A thing to note
    ########
    if hasattr(dtc, "current_spike_number_search"):
        from neuronunit.tests import SpikeCountSearch

        observation_spike_count = {}
        observation_spike_count["value"] = dtc.current_spike_number_search
        scs = SpikeCountSearch(observation_spike_count)
        assert model is not None
        target_current = scs.generate_prediction(model)

        uc = {
            "amplitude": target_current,
            "duration": ALLEN_DURATION,
            "delay": ALLEN_DELAY,
        }
        model.inject_square_current(uc)
        vm15 = model.get_membrane_potential()
        dtc.vm_soma = copy.copy(vm15)

        del model
        model = dtc.dtc_to_model()
        uc = {"amplitude": 0 * pq.pA, "duration": ALLEN_DURATION, "delay": ALLEN_DELAY}
        params = {
            "amplitude": rheobase,
            "duration": ALLEN_DURATION,
            "delay": ALLEN_DELAY,
        }
        model.inject_square_current(uc)
        vr = model.get_membrane_potential()
        dtc.vmr = np.mean(vr)
        del model
        return [vm30, vm15, params, None, dtc]

    else:

        uc = {
            "amplitude": 1.5 * rheobase,
            "duration": ALLEN_DURATION,
            "delay": ALLEN_DELAY,
        }
        model._backend.inject_square_current(**uc)
        vm15 = model.get_membrane_potential()
        dtc.vm_soma = copy.copy(vm15)
        del model
        model = dtc.dtc_to_model()

        uc = {
            "amplitude": 3.0 * rheobase,
            "duration": ALLEN_DURATION,
            "delay": ALLEN_DELAY,
        }
        model._backend.inject_square_current(**uc)

        vm30 = model.get_membrane_potential()
        dtc.vm30 = copy.copy(vm30)
        del model
        model = dtc.dtc_to_model()
        uc = {"amplitude": 00 * pq.pA, "duration": DURATION, "delay": DELAY}
        params = {"amplitude": rheobase, "duration": DURATION, "delay": DELAY}
        model._backend.inject_square_current(**uc)

        vr = model.get_membrane_potential()
        dtc.vmr = np.mean(vr)
        del model
        return [vm30, vm15, params, None, dtc]


def efel_evaluation(instance_obj: Any, specific_filter_list: List = None,current:float=None) -> Any:
    """
    -- Synopsis: evaluate efel feature extraction criteria against on
    reduced cell models and probably efel data.
    """
    vm_used = instance_obj.vm_soma
    try:
        efel.reset()
    except:
        pass
    efel.setThreshold(0)
    if current is None:
        if hasattr(instance_obj, "solve_for_current"):
            current = instance_obj.solve_for_current
    trace3 = {
        "T": [float(t) * 1000.0 for t in vm_used.times],
        "V": [float(v) for v in vm_used.magnitude],
        "stimulus_current": [current],
    }
    ALLEN_DURATION = 2000 * pq.ms
    ALLEN_DELAY = 1000 * pq.ms
    trace3["stim_end"] = [float(ALLEN_DELAY) + float(ALLEN_DURATION)]
    trace3["stim_start"] = [float(ALLEN_DELAY)]
    efel_list = list(efel.getFeatureNames())
    if np.min(vm_used.magnitude) < 0:
        if not np.max(vm_used.magnitude) > 0:
            vm_used_mag = [v + np.mean([0, float(np.max(v))]) * pq.mV for v in vm_used]
            if not np.max(vm_used_mag) > 0:
                instance_obj.efel_15 = None
                return instance_obj

            trace3["V"] = vm_used_mag
        if specific_filter_list is None:
            specific_filter_list = [
                "burst_ISI_indices",
                "burst_mean_freq",
                "burst_number",
                "single_burst_ratio",
                "ISI_log_slope",
                "mean_frequency",
                "adaptation_index2",
                "first_isi",
                "ISI_CV",
                "median_isi",
                "Spikecount",
                "all_ISI_values",
                "ISI_values",
                "time_to_first_spike",
                "time_to_last_spike",
                "time_to_second_spike",
                "Spikecount",
            ]
        results = efel.getMeanFeatureValues(
            [trace3], specific_filter_list, raise_warnings=False
        )
        if "MAT" not in instance_obj.backend:
            thresh_cross = threshold_detection(vm_used, 0 * pq.mV)
            for index, tc in enumerate(thresh_cross):
                results[0]["spike_" + str(index)] = float(tc)
        else:
            if hasattr(instance_obj, "spikes"):
                instance_obj.spikes = model._backend.spikes
                for index, tc in enumerate(instance_obj.spikes):
                    results[0]["spike_" + str(index)] = float(tc)
        nans = {k: v for k, v in results[0].items() if type(v) is type(None)}
        instance_obj.efel_15 = None
        instance_obj.efel_15 = results
        efel.reset()
    return instance_obj


def inject_and_plot_model(
    dtc: DataTC, figname=None, plotly=True, verbose=False
) -> Union[Any, Any, Any]:
    """
    -- Synopsis: produce rheobase injection value
    produce an object of class sciunit Runnable Model
    with known attributes
    and known rheobase current injection value.
    """
    dtc = dtc_to_rheo(dtc)
    model = dtc.dtc_to_model()
    uc = {"amplitude": dtc.rheobase, "duration": DURATION, "delay": DELAY}
    if dtc.jithub or "NEURON" in str(dtc.backend):
        vm = model._backend.inject_square_current(**uc)
    else:
        vm = model.inject_square_current(uc)
    vm = model.get_membrane_potential()
    if verbose:
        if vm is not None:
            print(vm[-1], vm[-1] < 0 * pq.mV)
    if vm is None:
        return [None, None, None]
    if not plotly:
        plt.clf()
        plt.figure()
        if dtc.backend in str("HH"):
            plt.title("Conductance based model membrane potential plot")
        else:
            plt.title("Membrane potential plot")
        plt.plot(vm.times, vm.magnitude, "k")
        plt.ylabel("V (mV)")
        plt.xlabel("Time (s)")

        if figname is not None:
            plt.savefig(str(figname) + str(".png"))
        plt.plot(vm.times, vm.magnitude)

    if plotly:
        fig = px.line(x=vm.times, y=vm.magnitude, labels={"x": "t (s)", "y": "V (mV)"})
        if figname is not None:
            fig.write_image(str(figname) + str(".png"))
        else:
            return vm, fig, dtc
    return [vm, plt, dtc]


def switch_logic(xtests):  # ->
    try:
        atsd = TSD()
    except:
        atsd = neuronunit.optimization.optimization_management.TSD()

    if type(xtests) is type(atsd):
        xtests = list(xtests.values())
    for t in xtests:
        if str("FITest") == t.name:
            t.active = True
            t.passive = False

        if str("RheobaseTest") == t.name:
            t.active = True
            t.passive = False
        elif str("InjectedCurrentAPWidthTest") == t.name:
            t.active = True
            t.passive = False
        elif str("InjectedCurrentAPAmplitudeTest") == t.name:
            t.active = True
            t.passive = False
        elif str("InjectedCurrentAPThresholdTest") == t.name:
            t.active = True
            t.passive = False
        elif str("RestingPotentialTest") == t.name:
            t.passive = False
            t.active = False
        elif str("InputResistanceTest") == t.name:
            t.passive = True
            t.active = False
        elif str("TimeConstantTest") == t.name:
            t.passive = True
            t.active = False
        elif str("CapacitanceTest") == t.name:
            t.passive = True
            t.active = False
        else:
            t.passive = False
            t.active = False
    return xtests


def active_values(keyed, rheobase, square=None):
    keyed["injected_square_current"] = {}
    if square is None:
        if isinstance(rheobase, type(dict())):
            keyed["injected_square_current"]["amplitude"] = (
                float(rheobase["value"]) * pq.pA
            )
        else:
            keyed["injected_square_current"]["amplitude"] = rheobase
    return keyed


def passive_values(keyed: dict = {}) -> dict:
    PASSIVE_DURATION = 500.0 * pq.ms
    PASSIVE_DELAY = 200.0 * pq.ms
    keyed["injected_square_current"] = {}
    keyed["injected_square_current"]["delay"] = PASSIVE_DELAY
    keyed["injected_square_current"]["duration"] = PASSIVE_DURATION
    keyed["injected_square_current"]["amplitude"] = -10 * pq.pA
    return keyed


def neutral_values(keyed: dict = {}) -> dict:
    PASSIVE_DURATION = 500.0 * pq.ms
    PASSIVE_DELAY = 200.0 * pq.ms
    keyed["injected_square_current"] = {}
    keyed["injected_square_current"]["delay"] = PASSIVE_DELAY
    keyed["injected_square_current"]["duration"] = PASSIVE_DURATION
    keyed["injected_square_current"]["amplitude"] = 0 * pq.pA
    return keyed


def initialise_test(v, rheobase=None):
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
    v.params = v.params  # ['injected_square_current']
    if "delay" in v.params["injected_square_current"].keys():
        v.params["tmax"] = (
            v.params["injected_square_current"]["delay"]
            + v.params["injected_square_current"]["duration"]
        )
    else:
        v.params["tmax"] = DELAY + DURATION
    return v
