import os
import sciunit
import neuronunit
import pickle
from neuronunit import tests as _, neuroelectro
from neuronunit.tests import passive, waveform, fi
from neuronunit.tests.fi import RheobaseTestP

from neuronunit.tests import *
import quantities as qt
from sciunit.scores import RatioScore, ZScore, RelativeDifferenceScore
import neuronunit
import pandas as pd

import pickle

anchor = __file__

import copy
import sciunit
from sciunit.suites import TestSuite
import pickle


import urllib.request, json

from neuronunit import neuroelectro
from scipy import stats
import numpy as np

import urllib.request, json


def id_to_frame(df_n, df_e, nxid):
    ntype = str(df_n[df_n["NeuroLex ID"] == nxid]["Neuron Type"].values[0])
    pyr = df_e[df_e["NeuronType"] == ntype]
    return pyr


def column_to_sem(df, column):

    temp = [i for i in df[column].values if not np.isnan(i)]
    sem = stats.sem(temp, axis=None, ddof=0)
    std = np.std(temp)  # , axis=None, ddof=0)
    mean = np.mean(temp)
    # print(column,sem)
    df = pd.DataFrame([{"sem": sem, "std": std, "mean": mean}], index=[column])
    return df


def cell_to_frame(df_n, df_e, nxid):
    pyr = id_to_frame(df_n, df_e, nxid)
    for cnt, key in enumerate(pyr.columns):
        empty = pd.DataFrame()
        if not key in "Species":
            if cnt == 0:
                df_old = column_to_sem(pyr, key)
            else:
                df_new = column_to_sem(pyr, key)
                df_old = pd.concat([df_new, df_old])
        else:
            break
    return df_old.T


def neuroelectro_summary_observation(neuron_name, ontology):
    ephysprop_name = ""
    verbose = False
    reference_data = neuroelectro.NeuroElectroSummary(
        neuron=neuron_name,
        ephysprop={"name": ontology["name"]},
        get_values=True,
        cached=True,
    )
    return reference_data


def get_obs(pipe):
    with urllib.request.urlopen("https://neuroelectro.org/api/1/e/") as url:
        ontologies = json.loads(url.read().decode())
    obs = []
    for p in pipe:
        for l in ontologies["objects"]:
            obs.append(neuroelectro_summary_observation(p, l))
    return obs


def update_amplitude(test, tests, score):
    rheobase = score.prediction["value"]
    for i in [4, 5, 6]:
        tests[i].params["injected_square_current"]["amplitude"] = (
            rheobase * 1.01
        )  # I feel that 1.01 may lead to more than one spike
    return


def substitute_parallel_for_serial(electro_tests):
    for test, obs in electro_tests:
        test[0] = RheobaseTestP(obs["Rheobase"])

    return electro_tests


def substitute_criteria(observations_donar, observations_acceptor):
    # Inputs an observation donar
    # and an observation acceptor
    # Many neuroelectro data sources have std 0 values
    for index, oa in observations_acceptor.items():
        for k, v in oa.items():
            if k == "std" and v == 0.0:
                if k in observations_donar.keys():
                    oa[k] = observations_donar[index][k]
    return observations_acceptor


"""
def substitute_parallel_for_serial(electro_tests):
    for test,obs in electro_tests:
        if str('Rheobase') in obs.keys():

            test[0] = RheobaseTestP(obs['Rheobase'])

    return electro_tests
"""


def replace_zero_std(electro_tests):
    for test, obs in electro_tests:
        if str("Rheobase") in obs.keys():
            test[0] = RheobaseTestP(obs["Rheobase"])
            for k, v in obs.items():
                if v["std"] == 0:

                    obs = substitute_criteria(electro_tests[1][1], obs)

    return electro_tests


def executable_druckman_tests(cell_id, file_name=None):
    # Use neuroelectro experimental obsevations to find test
    # criterion that will be used to inform scientific unit testing.
    # some times observations are not sourced from neuroelectro,
    # but they are imputated or borrowed from other TestSuite
    # if that happens make test objections using observations external
    # to this method, and provided as a method argument.
    tests = []
    observations = None
    test_classes = None

    dm.AP1AP1AHPDepthTest.ephysprop_name = None
    dm.AP1AP1AHPDepthTest.ephysprop_name = "AHP amplitude"
    dm.AP2AP1AHPDepthTest.ephysprop_name = None
    dm.AP2AP1AHPDepthTest.ephysprop_name = "AHP amplitude"
    dm.AP1AP1WidthHalfHeightTest.ephysprop_name = None
    dm.AP1AP1WidthHalfHeightTest.ephysprop_name = "spike half-width"
    dm.AP1AP1WidthPeakToTroughTest.ephysprop_name = None
    dm.AP1AP1WidthPeakToTroughTest.ephysprop_name = "spike width"
    # dm.IinitialAccomodationMeanTest.ephysprop_name = None
    # dm.IinitialAccomodationMeanTest.ephysprop_name = 'adaptation_percent'

    test_classes = [
        fi.RheobaseTest,
        dm.AP1AP1AHPDepthTest,
        dm.AP2AP1AHPDepthTest,
        dm.AP1AP1WidthHalfHeightTest,
        dm.AP1AP1WidthPeakToTroughTest,
    ]
    observations = {}
    for index, t in enumerate(test_classes):
        obs = t.neuroelectro_summary_observation(cell_id)

        if obs is not None:
            if "mean" in obs.keys():
                tests.append(t(obs))
                observations[t.ephysprop_name] = obs

    suite = sciunit.TestSuite(tests, name="vm_suite")

    if file_name is not None:
        file_name = file_name + ".p"
        with open(file_name, "wb") as f:
            pickle.dump(tests, f)

    return tests, observations


def executable_tests(cell_id, file_name=None):
    # Use neuroelectro experimental obsevations to find test
    # criterion that will be used to inform scientific unit testing.
    # some times observations are not sourced from neuroelectro,
    # but they are imputated or borrowed from other TestSuite
    # if that happens make test objections using observations external
    # to this method, and provided as a method argument.
    tests = []
    observations = None
    test_classes = None
    test_classes = [
        fi.RheobaseTest,
        passive.InputResistanceTest,
        passive.TimeConstantTest,
        passive.CapacitanceTest,
        passive.RestingPotentialTest,
    ]  # ,
    observations = {}
    for index, t in enumerate(test_classes):
        if "RheobaseTest" in t.name:
            t.score_type = sciunit.scores.ZScore
        if "RheobaseTestP" in t.name:
            t.score_type = sciunit.scores.ZScore

        obs = t.neuroelectro_summary_observation(cell_id)

        if obs is not None:
            if "mean" in obs.keys():
                tests.append(t(obs))
                observations[t.ephysprop_name] = obs

    # hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
    # update amplitude at the location in sciunit thats its passed to, without any loss of generality.
    suite = sciunit.TestSuite(tests, name="vm_suite")

    if file_name is not None:
        file_name = file_name + ".p"
        with open(file_name, "wb") as f:
            pickle.dump(tests, f)

    return tests, observations


def get_neuron_criteria(cell_id, file_name=None):  # ,observation = None):
    # Use neuroelectro experimental obsevations to find test
    # criterion that will be used to inform scientific unit testing.
    # some times observations are not sourced from neuroelectro,
    # but they are imputated or borrowed from other TestSuite
    # if that happens make test objections using observations external
    # to this method, and provided as a method argument.
    tests = {}
    observations = None
    test_classes = None
    test_classes = [
        fi.RheobaseTest,
        passive.InputResistanceTest,
        passive.TimeConstantTest,
        passive.CapacitanceTest,
        passive.RestingPotentialTest,
    ]
    observations = {}
    for index, t in enumerate(test_classes):
        obs = t.neuroelectro_summary_observation(cell_id)

        if obs is not None:
            if "mean" in obs.keys():
                print(test_classes[index])
                print(t.name)
                tt = t(obs)
                tests[t.name] = tt
                observations[t.ephysprop_name] = obs
    # hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
    # update amplitude at the location in sciunit thats its passed to, without any loss of generality.
    # suite = sciunit.TestSuite(tests,name="vm_suite")

    if file_name is not None:
        file_name = file_name + ".p"
        with open(file_name, "wb") as f:
            pickle.dump(tests, f)

    return tests, observations


def get_olf_cell():
    cell_constraints = {}
    olf_mitral = {
        "id": 129,
        "name": "Olfactory bulb (main) mitral cell",
        "neuron_db_id": 267,
        "nlex_id": "nlx_anat_100201",
    }

    # NLXANAT:100201

    olf_mitral = {
        "id": 129,
        "name": "Olfactory bulb (main) mitral cell",
        "neuron_db_id": 267,
        "nlex_id": "nlx_anat_100201",
    }
    # olf_mitral['id'] = 'nlx_anat_100201'
    # olf_mitral['nlex_id'] = 'nlx_anat_100201'
    tests, observations = get_neuron_criteria(olf_mitral)
    # import pdb
    # pdb.set_trace()
    cell_constraints["olf_mitral"] = tests
    with open("olf.p", "wb") as f:
        pickle.dump(cell_constraints, f)

    return cell_constraints


def get_all_cells():
    ###
    # sagratio
    ###
    purkinje = {
        "id": 18,
        "name": "Cerebellum Purkinje cell",
        "neuron_db_id": 271,
        "nlex_id": "sao471801888",
    }
    # fi_basket = {"id": 65, "name": "Dentate gyrus basket cell", "neuron_db_id": None, "nlex_id": "nlx_cell_100201"}
    pvis_cortex = {
        "id": 111,
        "name": "Neocortex pyramidal cell layer 5-6",
        "neuron_db_id": 265,
        "nlex_id": "nifext_50",
    }
    # This olfactory mitral cell does not have datum about rheobase, current injection values.
    olf_mitral = {
        "id": 129,
        "name": "Olfactory bulb (main) mitral cell",
        "neuron_db_id": 267,
        "nlex_id": "nlx_anat_100201",
    }
    ca1_pyr = {
        "id": 85,
        "name": "Hippocampus CA1 pyramidal cell",
        "neuron_db_id": 258,
        "nlex_id": "sao830368389",
    }
    cell_list = [olf_mitral, ca1_pyr, purkinje, pvis_cortex]
    cell_constraints = {}
    for cell in cell_list:
        tests, observations = get_neuron_criteria(cell)
        cell_constraints[cell["name"]] = tests
    with open("multicellular_constraints.p", "wb") as f:
        pickle.dump(cell_constraints, f)
    return cell_constraints


def switch_logic(xtests):
    # move this logic into sciunit tests
    """
    Hopefuly depreciated by future NU debugging.
    """
    aTSD = neuronunit.optimisation.optimization_management.TSD()
    if type(xtests) is type(aTSD):
        xtests = list(xtests.values())
    if type(xtests) is type(list()):
        pass
    for t in xtests:
        if str("RheobaseTest") == t.name:
            t.active = True
            t.passive = False
        elif str("RheobaseTestP") == t.name:
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
            t.passive = True
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


def process_all_cells():
    '''
    --Synopsis: download some NeuroElectroSummary
    observations and use values to construct appropriate
    NeuronUnit tests, then pickle them.
    
    TODO rename pull all cells
    see alias below.

    '''
    try:
        with open("processed_multicellular_constraints.p", "rb") as f:
            filtered_cells = pickle.load(f)
        return filtered_cells
    except:
        try:
            cell_constraints = pickle.load(
                open("multicellular_suite_constraints.p", "rb")
            )
        except:
            cell_constraints = get_all_cells()

    filtered_cells = {}
    for key, cell in cell_constraints.items():
        filtered_cell_constraints = []
        if type(cell) is type(dict()):
            for t in cell.values():
                if t.observation is not None:
                    if float(t.observation["std"]) == 0.0:
                        t.observation["std"] = t.observation["mean"]
                    else:
                        filtered_cell_constraints.append(t)
        #    filtered_cells[key] = TestSuite(filtered_cell_constraints)
        else:
            for t in cell.tests:
                if t.observation is not None:
                    if float(t.observation["std"]) == 0.0:
                        t.observation["std"] = t.observation["mean"]
                    else:
                        filtered_cell_constraints.append(t)

        filtered_cells[key] = TestSuite(filtered_cell_constraints)
        """
        for t in filtered_cells[key].tests:
            t = switch_logic(t)
            assert hasattr(t,'active')
            assert hasattr(t,'passive')
        for t in filtered_cells[key].tests:
            assert hasattr(t,'active')
            assert hasattr(t,'passive')
        """
        with open("processed_multicellular_constraints.p", "wb") as f:
            pickle.dump(filtered_cells, f)
    return filtered_cells

def pull_all_cells():
    filtered_cells = process_all_cells()



def get_common_criteria():
    purkinje = {
        "id": 18,
        "name": "Cerebellum Purkinje cell",
        "neuron_db_id": 271,
        "nlex_id": "sao471801888",
    }
    # fi_basket = {"id": 65, "name": "Dentate gyrus basket cell", "neuron_db_id": None, "nlex_id": "nlx_cell_100201"}
    pvis_cortex = {
        "id": 111,
        "name": "Neocortex pyramidal cell layer 5-6",
        "neuron_db_id": 265,
        "nlex_id": "nifext_50",
    }
    # This olfactory mitral cell does not have datum about rheobase, current injection values.
    olf_mitral = {
        "id": 129,
        "name": "Olfactory bulb (main) mitral cell",
        "neuron_db_id": 267,
        "nlex_id": "nlx_anat_100201",
    }
    ca1_pyr = {
        "id": 85,
        "name": "Hippocampus CA1 pyramidal cell",
        "neuron_db_id": 258,
        "nlex_id": "sao830368389",
    }
    pipe = [olf_mitral, ca1_pyr, purkinje, pvis_cortex]
    electro_tests = []
    obs_frame = {}
    test_frame = {}

    try:

        anchor = os.path.dirname(anchor)
        mypath = os.path.join(os.sep, anchor, "optimisation/all_tests.p")

        electro_path = str(os.getcwd()) + "all_tests.p"

        assert os.path.isfile(electro_path) == True
        with open(electro_path, "rb") as f:
            (obs_frame, test_frame) = pickle.load(f)

    except:
        for p in pipe:
            print(p)
            p_observations = get_obs(p)
            p_tests = p(p_observations)
            obs_frame[p["name"]] = p_observations  # , p_tests))
            test_frame[p["name"]] = p_tests  # , p_tests))
        electro_path = str(os.getcwd()) + "all_tests.p"
        with open(electro_path, "wb") as f:
            pickle.dump((obs_frame, test_frame), f)

    return (obs_frame, test_frame)


def get_tests(backend=str("RAW")):
    import neuronunit

    anchor = neuronunit.__file__
    anchor = os.path.dirname(anchor)
    mypath = os.path.join(os.sep, anchor, "unit_test/pipe_tests.p")

    # get neuronunit tests
    # and select out Rheobase test and input resistance test
    # and less about electrophysiology of the membrane.
    # We are more interested in phenomonogical properties.
    electro_path = mypath
    # str(os.getcwd())+'/pipe_tests.p'
    def dont_use_cache():
        assert os.path.isfile(electro_path) == True
        with open(electro_path, "rb") as f:
            electro_tests = pickle.load(f)

    electro_tests = get_common_criteria()
    electro_tests = replace_zero_std(electro_tests)
    test, observation = electro_tests[0]
    tests = copy.copy(electro_tests[0][0])
    suite = sciunit.TestSuite(tests)

    return tests, test, observation, suite


def do_use_cache():
    import neuronunit

    anchor = neuronunit.__file__
    anchor = os.path.dirname(anchor)
    mypath = os.path.join(os.sep, anchor, "unit_test/pipe_tests.p")

    # get neuronunit tests
    # and select out Rheobase test and input resistance test
    # and less about electrophysiology of the membrane.
    # We are more interested in phenomonogical properties.
    electro_path = mypath
    assert os.path.isfile(electro_path) == True
    with open(electro_path, "rb") as f:
        electro_tests = pickle.load(f)
    return electro_tests


def remake_tests():
    # Use neuroelectro experimental obsevations to find test
    # criterion that will be used to inform scientific unit testing.
    # some times observations are not sourced from neuroelectro,
    # but they are imputated or borrowed from other TestSuite
    # if that happens make test objections using observations external
    # to this method, and provided as a method argument.
    tests = []
    observations = None
    test_classes = None

    test_classes = [
        fi.RheobaseTest,
        passive.InputResistanceTest,
        passive.TimeConstantTest,
        passive.CapacitanceTest,
        passive.RestingPotentialTest,
    ]
    electro_obs = do_use_cache()
    test_cell_dict = {}
    for eo in electro_obs:
        tests = []
        for index, t in enumerate(test_classes):
            for ind_obs in eo:
                test = t(ind_obs)
                tests.append(test)
        suite = sciunit.TestSuite(tests)
        # test_cell_dict[]

        if obs is not None:
            if "mean" in obs.keys():
                tests.append(t(obs))
                observations[t.ephysprop_name] = obs

    electro_tests = replace_zero_std(electro_tests)
    test, observation = electro_tests[0]
    tests = copy.copy(electro_tests[0][0])
    suite = sciunit.TestSuite(tests)
    return tests, test, observation, suite
