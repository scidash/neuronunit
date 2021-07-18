"""F/I neuronunit tests, e.g. investigating firing rates and patterns as a
function of input current"""

import os
import multiprocessing

global cpucount
cpucount = multiprocessing.cpu_count()
from .base import np, pq, ncap, VmTest, scores, AMPL

DURATION = 2000
DELAY = 1000

<<<<<<< HEAD
from neuronunit.optimization.data_transport_container import DataTC
=======
#from neuronunit.optimization.data_transport_container import DataTC
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f
import os
import quantities
import neuronunit

<<<<<<< HEAD
import dask.bag as db
=======
#import dask.bag as db
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f
import quantities as pq
import numpy as np
import copy
import pdb
<<<<<<< HEAD
from numba import jit
import time
import numba
=======
#from numba import jit
import time
#import numba
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f
import copy

import matplotlib as mpl

# mpl.use('Agg')
import matplotlib.pyplot as plt
from neuronunit.capabilities.spike_functions import (
    get_spike_waveforms,
    spikes2amplitudes,
    threshold_detection,
)

#
# When using differentiation based spike detection is used this is faster.


class SpikeCountSearch(VmTest):
    """
    A parallel Implementation of a Binary search algorithm,
    which finds a rheobase prediction.

    Strengths: this algorithm is faster than the serial class present in this file for model backends that are not able to
    implement numba jit optimisation.
    Failure to implement JIT happens to be typical of a signifcant number of backends
    """

    def _extra(self):
        self.verbose = 1

    required_capabilities = (ncap.ReceivesSquareCurrent, ncap.ProducesSpikes)
    params = {
        "injected_square_current": {
            "amplitude": 100.0 * pq.pA,
            "delay": DELAY,
            "duration": DURATION,
        }
    }
    name = "Rheobase test"
    description = (
        "A test of the rheobase, i.e. the minimum injected current "
        "needed to evoke at least one spike."
    )
    units = pq.pA
    ephysprop_name = "Rheobase"
    score_type = scores.ZScore

    def generate_prediction(self, model):
        def check_fix_range(dtc):
            """
            Inputs: lookup, A dictionary of previous current injection values
            used to search rheobase
            Outputs: A boolean to indicate if the correct rheobase current was found
            and a dictionary containing the range of values used.
            If rheobase was actually found then rather returning a boolean and a dictionary,
            instead logical True, and the rheobase current is returned.
            given a dictionary of rheobase search values, use that
            dictionary as input for a subsequent search.
            """

            steps = []
            dtc.rheobase = {}
            dtc.rheobase["value"] = None
            sub, supra = get_sub_supra(dtc.lookup)
            if (len(sub) + len(supra)) == 0:
                # This assertion would only be occur if there was a bug
                assert sub.max() <= supra.min()
            elif len(sub) and len(supra):
                # Termination criterion

                steps = np.linspace(sub.max(), supra.min(), cpucount + 1) * pq.pA
                steps = steps[1:-1] * pq.pA
            elif len(sub):
                steps = np.linspace(sub.max(), 2 * sub.max(), cpucount + 1) * pq.pA
                steps = steps[1:-1] * pq.pA
            elif len(supra):
                steps = (
                    np.linspace(supra.min() - 100, supra.min(), cpucount + 1) * pq.pA
                )
                steps = steps[1:-1] * pq.pA

            dtc.current_steps = steps
            return dtc

        def get_sub_supra(lookup):
            sub, supra = [], []
            for current, n_spikes in lookup.items():
                if n_spikes < self.observation["value"]:
                    sub.append(current)
                elif n_spikes > self.observation["value"]:
                    supra.append(current)
                delta = n_spikes - self.observation["value"]
                # print(delta,'difference \n\n\n\nn')
            sub = np.array(sorted(list(set(sub))))
            supra = np.array(sorted(list(set(supra))))
            return sub, supra

        def check_current(dtc):
            """
            Inputs are an amplitude to test and a virtual model
            output is an virtual model with an updated dictionary.
            """
            dtc.boolean = False

<<<<<<< HEAD
            model = dtc.dtc_to_model()
            model._backend.attrs = dtc.attrs
=======
            model = dtc#.dtc_to_model()
            #print(type(model))
            #model._backend.attrs = dtc.attrs
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f
            params = {
                "injected_square_current": {
                    "amplitude": None,
                    "delay": DELAY,
                    "duration": DURATION,
                }
            }

            ampl = float(dtc.ampl)
            if ampl not in dtc.lookup or len(dtc.lookup) == 0:
                uc = {"amplitude": ampl * pq.pA, "duration": DURATION, "delay": DELAY}
<<<<<<< HEAD

                if str("JIT_") in model.backend:
                    _ = model.inject_square_current(**uc)
                    n_spikes = model.get_spike_count()
                    # _ = model._backend.inject_square_current(**uc)
                    # n_spikes_b = model._backend.get_spike_count()
                    # assert n_spikes == n_spikes_b
=======
                #print(model._backend)
                vm = dtc.inject_square_current(**uc)
                n_spikes = dtc.get_spike_count()
                # _ = model._backend.inject_square_current(**uc)
                # n_spikes_b = model._backend.get_spike_count()
                # assert n_spikes == n_spikes_b
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f
                if float(ampl) < -10.0:
                    dtc.rheobase = {}
                    dtc.rheobase["value"] = None
                    dtc.boolean = True
                    return dtc

                target_spk_count = self.observation["value"]
                if n_spikes == target_spk_count:

                    dtc.lookup[float(ampl)] = n_spikes
                    dtc.rheobase = {}
                    dtc.rheobase["value"] = float(ampl) * pq.pA
                    dtc.target_spk_count = None
                    dtc.target_spk_count = dtc.rheobase["value"]
                    dtc.boolean = True
                    if self.verbose > 2:
                        print(
                            "gets here",
                            n_spikes,
                            target_spk_count,
                            n_spikes == target_spk_count,
                        )
                    return dtc

                dtc.lookup[float(ampl)] = n_spikes
            return dtc

        def init_dtc(dtc):
            """
            Exploit memory of last model in genes.
            """
            if dtc.initiated == True:

                dtc = check_current(dtc)
                if dtc.boolean:

                    return dtc

                else:

                    if type(dtc.current_steps) is type(float):
                        dtc.current_steps = [
                            0.75 * dtc.current_steps,
                            1.25 * dtc.current_steps,
                        ]
                    elif type(dtc.current_steps) is type(list):
                        dtc.current_steps = [s * 1.25 for s in dtc.current_steps]
                    dtc.initiated = (
                        True  # logically unnecessary but included for readibility
                    )
            if dtc.initiated == False:
                dtc.boolean = False
                # steps = np.linspace(-12,67,int(8))
                ###
                # These values are important for who knows what reason
                steps = np.linspace(-10, 65, int(7))
                ###
                steps_current = [i * pq.pA for i in steps]
                dtc.current_steps = steps_current
                dtc.initiated = True
            return dtc

        def find_target_current(self, dtc):
            # This line should not be necessary:
            # a class, VeryReducedModel has been made to circumvent this.
<<<<<<< HEAD
            # if hasattr(dtc,'model_path'):
            #    assert os.path.isfile(dtc.model_path), "%s is not a file" % dtc.model_path
            # If this it not the first pass/ first generation
            # then assume the rheobase value found before mutation still holds until proven otherwise.
            # dtc = check_current(model.rheobase,dtc)
            # If its not true enter a search, with ranges informed by memory
=======
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f
            cnt = 0
            sub = np.array([0, 0])
            supra = np.array([0, 0])
            ##
            #
            ## Important number
<<<<<<< HEAD
            # big = 100
=======
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f
            big = 250
            while dtc.boolean == False and cnt < big:

                # negeative spiker
                if len(sub):
                    if sub.max() < -1.0:
                        pass
                dtc_clones = [dtc for i in range(0, len(dtc.current_steps))]
                for i, s in enumerate(dtc.current_steps):
                    dtc_clones[i] = copy.copy(dtc_clones[i])
                    dtc_clones[i].ampl = copy.copy(dtc.current_steps[i])
                    dtc_clones[i].backend = copy.copy(dtc.backend)

                dtc_clones = [d for d in dtc_clones if not np.isnan(d.ampl)]
                set_clones = set([float(d.ampl) for d in dtc_clones])
                dtc_clone = []
                for dtc, sc in zip(dtc_clones, set_clones):
                    dtc = copy.copy(dtc)
                    dtc.ampl = sc * pq.pA
                    dtc = check_current(dtc)
                    dtc_clone.append(dtc)

                for dtc in dtc_clone:
                    if dtc.boolean == True:
                        return dtc

                for d in dtc_clone:
                    dtc.lookup.update(d.lookup)
                dtc = check_fix_range(dtc)

                sub, supra = get_sub_supra(dtc.lookup)
                if len(supra) and len(sub):
                    delta = float(supra.min()) - float(sub.max())
                    # tolerance = 0.0

                if self.verbose >= 2:
                    print("not rheobase alg")
                    print(
                        "Try %d: SubMax = %s; SupraMin = %s"
                        % (
                            cnt,
                            sub.max() if len(sub) else None,
                            supra.min() if len(supra) else None,
                        )
                    )
                cnt += 1
                reversed = {v: k for k, v in dtc.lookup.items()}
            if cnt == big:
                if self.verbose >= 2:
                    print("counted out and thus wrong spike count")
            return dtc

<<<<<<< HEAD
        dtc = DataTC()
        dtc.attrs = {}
        for k, v in model.attrs.items():
            dtc.attrs[k] = v

=======
        #dtc = DataTC()
        #dtc.attrs = {}
        #for k, v in model.attrs.items():
        #    dtc.attrs[k] = v
        dtc = model
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f
        # this is not a perservering assignment, of value,
        # but rather a multi statement assertion that will be checked.
        dtc.backend = model.backend
        dtc = init_dtc(dtc)
        prediction = {}
        temp = find_target_current(self, dtc).rheobase

        if type(temp) is not type(None) and not type(dict):
            prediction["value"] = float(temp) * pq.pA
        elif type(temp) is not type(None) and type(dict):
            if temp["value"] is not None:
                prediction["value"] = float(temp["value"]) * pq.pA
            else:
                prediction = None
        else:
            prediction = None
        # print(prediction,'prediction')
        return prediction


class SpikeCountRangeSearch(VmTest):
    """
    A parallel Implementation of a Binary search algorithm,
    which finds a rheobase prediction.

    Strengths: this algorithm is faster than the serial class, present in this file for model backends that are not able to
    implement numba jit optimisation, which actually happens to be typical of a signifcant number of backends

    Weaknesses this serial class is significantly slower, for many backend implementations including raw NEURON
    NEURON via PyNN, and possibly GLIF.

    """

    def _extra(self):
        self.verbose = 1

    # def __init__(self,other_current=None):
    #     self.other_current = other_current

    required_capabilities = (ncap.ReceivesSquareCurrent, ncap.ProducesSpikes)
    # DELAY = 100.0*pq.ms
    # DURATION = 1000.0*pq.ms
    params = {
        "injected_square_current": {
            "amplitude": 100.0 * pq.pA,
            "delay": DELAY,
            "duration": DURATION,
        }
    }
    name = "Rheobase test"
    description = (
        "A test of the rheobase, i.e. the minimum injected current "
        "needed to evoke at least one spike."
    )
    units = pq.pA
    # tolerance  # Rheobase search tolerance in `self.units`.
    ephysprop_name = "Rheobase"
    score_type = scores.ZScore

    def generate_prediction(self, model):
        def check_fix_range(dtc):
            """
            Inputs: lookup, A dictionary of previous current injection values
            used to search rheobase
            Outputs: A boolean to indicate if the correct rheobase current was found
            and a dictionary containing the range of values used.
            If rheobase was actually found then rather returning a boolean and a dictionary,
            instead logical True, and the rheobase current is returned.
            given a dictionary of rheobase search values, use that
            dictionary as input for a subsequent search.
            """

            steps = []
            dtc.rheobase = None
            sub, supra = get_sub_supra(dtc.lookup)

            if (len(sub) + len(supra)) == 0:
                # This assertion would only be occur if there was a bug
                assert sub.max() <= supra.min()
            elif len(sub) and len(supra):
                # Termination criterion

                steps = np.linspace(sub.max(), supra.min(), cpucount + 1) * pq.pA
                steps = steps[1:-1] * pq.pA
            elif len(sub):
                steps = np.linspace(sub.max(), 2 * sub.max(), cpucount + 1) * pq.pA
                steps = steps[1:-1] * pq.pA
            elif len(supra):
                steps = (
                    np.linspace(supra.min() - 100, supra.min(), cpucount + 1) * pq.pA
                )
                steps = steps[1:-1] * pq.pA

            dtc.current_steps = steps
            return dtc

        def get_sub_supra(lookup):
            sub, supra = [], []
            for current, n_spikes in lookup.items():
                if n_spikes < self.observation["range"][0]:
                    sub.append(current)
                elif n_spikes > self.observation["range"][1]:
                    supra.append(current)

            sub = np.array(sorted(list(set(sub))))
            supra = np.array(sorted(list(set(supra))))
            return sub, supra

        """
        def check_current(dtc):

            #Inputs are an amplitude to test and a virtual model
            #output is an virtual model with an updated dictionary.

            dtc.boolean = False

            model = dtc.dtc_to_model()

            params = {'injected_square_current':
                      {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

            ampl = float(dtc.ampl)
            if ampl not in dtc.lookup or len(dtc.lookup) == 0:
                uc = {'amplitude':ampl*pq.pA,'duration':DURATION,'delay':DELAY}

                dtc.run_number += 1
                model.set_attrs(dtc.attrs)
                model.inject_square_current(uc)
                n_spikes = model.get_spike_count()

                if float(ampl) < -1.0:
                    dtc.rheobase = None
                    dtc.boolean = True
                    return dtc

                #target_spk_count = self.observation['range']
                if self.observation['range'][0] <= n_spikes <= self.observation['range'][1]:
                    dtc.lookup[float(ampl)] = n_spikes
                    dtc.rheobase = {}
                    dtc.rheobase['value'] = float(ampl)*pq.pA
                    dtc.target_spk_count = None
                    dtc.target_spk_count = dtc.rheobase['value']
                    dtc.boolean = True
                    if self.verbose >2:
                        print('gets here',n_spikes,target_spk_count,n_spikes == target_spk_count)
                    return dtc

                dtc.lookup[float(ampl)] = n_spikes
            return dtc
        """

        def init_dtc(dtc):
            """
            Exploit memory of last model in genes.
            """
            # check for memory and exploit it.
            if dtc.initiated == True:

                dtc = check_current(dtc)
                if dtc.boolean:

                    return dtc

                else:
                    # Exploit memory of the genes to inform searchable range.
<<<<<<< HEAD

=======
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f
                    # if this model has lineage, assume it didn't mutate that far away from it's ancestor.
                    # using that assumption, on first pass, consult a very narrow range, of test current injection samples:
                    # only slightly displaced away from the ancestors rheobase value.

                    if type(dtc.current_steps) is type(float):
                        dtc.current_steps = [
                            0.75 * dtc.current_steps,
                            1.25 * dtc.current_steps,
                        ]
                    elif type(dtc.current_steps) is type(list):
                        dtc.current_steps = [s * 1.25 for s in dtc.current_steps]
                    dtc.initiated = (
                        True  # logically unnecessary but included for readibility
                    )
            if dtc.initiated == False:

                dtc.boolean = False

                steps = np.linspace(0, 85.0, int(8))

                steps_current = [i * pq.pA for i in steps]
                dtc.current_steps = steps_current
                dtc.initiated = True
            return dtc

        def find_target_current(self, dtc):
            cnt = 0
            sub = np.array([0, 0])
            supra = np.array([0, 0])
            # if dtc.backend is 'GLIF':
            #    big = 100
            # else:
            big = 255

            while dtc.boolean == False and cnt < big:

                # negeative spiker
                if len(sub):
                    if sub.max() < -1.0:
                        pass
                        # use_diff = True # differentiate the wave to look for spikes

                # be = dtc.backend
                dtc_clones = [dtc for i in range(0, len(dtc.current_steps))]
                for i, s in enumerate(dtc.current_steps):
                    dtc_clones[i] = copy.copy(dtc_clones[i])
                    dtc_clones[i].ampl = copy.copy(dtc.current_steps[i])
                    dtc_clones[i].backend = copy.copy(dtc.backend)

                dtc_clones = [d for d in dtc_clones if not np.isnan(d.ampl)]
                # try:
                #    b0 = db.from_sequence(dtc_clones, npartitions=npartitions)
                #    dtc_clone = list(b0.map(check_current).compute())
                # except:
                set_clones = set([float(d.ampl) for d in dtc_clones])
                dtc_clone = []
                for dtc, sc in zip(dtc_clones, set_clones):
                    dtc = copy.copy(dtc)
                    dtc.ampl = sc * pq.pA
                    dtc = check_current(dtc)
                    dtc_clone.append(dtc)

                for dtc in dtc_clone:
                    if dtc.boolean == True:
                        return dtc

                for d in dtc_clone:
                    dtc.lookup.update(d.lookup)
                dtc = check_fix_range(dtc)

                sub, supra = get_sub_supra(dtc.lookup)
                if len(supra) and len(sub):
                    delta = float(supra.min()) - float(sub.max())
                    # if str("GLIF") in dtc.backend:
                    #    tolerance = 0.0
                    # else:
                    # tolerance = 0.0
                    # tolerance = tolerance
                    if str(supra.min()) == str(sub.max()):
                        if self.verbose >= 2:
                            print(
                                delta,
                                "a neuron, close to the edge! Multi spiking rheobase. # spikes: ",
                                len(supra),
                            )
                        if len(supra) < 100:
                            dtc.rheobase["value"] = float(supra.min())
                            dtc.boolean = True
                            dtc.lookup[float(supra.min())] = len(supra)
                        else:
                            if type(dtc.rheobase) is type(None):
                                dtc.rheobase = {}
                                dtc.rheobase["value"] = None
                                dtc.boolean = False
                                dtc.lookup[float(supra.min())] = len(supra)

                        return dtc

                if self.verbose >= 2:
                    print("not rheobase alg")
                    print(
                        "Try %d: SubMax = %s; SupraMin = %s"
                        % (
                            cnt,
                            sub.max() if len(sub) else None,
                            supra.min() if len(supra) else None,
                        )
                    )
                cnt += 1
            return dtc

<<<<<<< HEAD
        dtc = DataTC()
        dtc.attrs = {}
        for k, v in model.attrs.items():
            dtc.attrs[k] = v

        # this is not a perservering assignment, of value,
        # but rather a multi statement assertion that will be checked.
        dtc.backend = model.backend
=======
        #dtc = DataTC()
        dtc = model
        #dtc.attrs = {}
        #for k, v in model.attrs.items():
        #    dtc.attrs[k] = v

        # this is not a perservering assignment, of value,
        # but rather a multi statement assertion that will be checked.
        #dtc.backend = model.backend
>>>>>>> 9fb0c2e613a1bf059f38eeeae80582d0cfb11f2f
        dtc = init_dtc(dtc)

        prediction = {}

        temp = find_target_current(self, dtc).rheobase
        if type(temp) is not type(None):
            if type(temp) is not type(dict()):
                prediction["value"] = float(temp) * pq.pA
            elif type(temp) is type(dict()):
                if type(temp["value"]) is not type(None):
                    prediction["value"] = float(temp["value"]) * pq.pA
                else:
                    prediction["value"] = None
            elif type(temp) is type(None):
                prediction["value"] = None  # float(temp['value'])* pq.pA

        else:
            prediction = None
        return prediction
