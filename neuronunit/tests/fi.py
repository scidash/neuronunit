"""F/I neuronunit tests, e.g. investigating firing rates and patterns as a
function of input current"""

import os
import multiprocessing
global cpucount
npartitions = cpucount = multiprocessing.cpu_count()
from .base import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
#DURATION = 2000
#DELAY = 200
from .. import optimisation

from neuronunit.optimisation.data_transport_container import DataTC
import os
import quantities
import neuronunit

import dask.bag as db
import quantities as pq
import numpy as np
import copy
import pdb
from numba import jit
import time
import numba
import copy

import matplotlib as mpl
try:
    mpl.use('Agg')
except:
    pass
import matplotlib.pyplot as plt
from neuronunit.capabilities.spike_functions import get_spike_waveforms, spikes2amplitudes, threshold_detection
#
# When using differentiation based spike detection is used this is faster.

#@jit
def diff(vm):
    return np.diff(vm)

#@jit
def get_diff(vm):
    differentiated = np.diff(vm)
    spike_lets = threshold_detection(differentiated,threshold=0.000193667327364)
    n_spikes = len([np.any(differentiated) > 0.000193667327364])
    return spike_lets, n_spikes

tolerance = 0.0#0.00125

class RheobaseTest(VmTest):
    """
    A serial Implementation of a Binary search algorithm,
    which finds a rheobase prediction

    Strengths: this algorithm is faster than the parallel class, present in this file under important and
    limited circumstances: this serial algorithm is faster than parallel for model backends that are able to
    implement numba jit optimisation

    Weaknesses this serial class is significantly slower, for many backend implementations including raw NEURON
    NEURON via PyNN, and possibly GLIF.
    """

    def _extra(self):
        self.prediction = {}
        self.high = 300*pq.pA
        self.small = 0*pq.pA
        self.rheobase_vm = None
        self.verbose = False


    required_capabilities = (cap.ReceivesSquareCurrent,
                             cap.ProducesSpikes)

    params = {'injected_square_current':
                {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

    name = "Rheobase test"
    description = ("A test of the rheobase, i.e. the minimum injected current "
                   "needed to evoke at least one spike.")
    units = pq.pA
    ephysprop_name = 'Rheobase'
    score_type = scores.ZScore

    #score_type = scores.RatioScore

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        prediction = {'value': None}
        model.rerun = True
        try:
            units = self.observation['value'].units
        except KeyError:
            units = self.observation['mean'].units

        begin_rh = time.time()
        lookup = self.threshold_FI(model, units)
        ##
        # New code
        ##
        temp = [ v for v in lookup.values() if v>1 ]
        if len(temp):
            too_many_spikes = np.min(temp)
        else:
            too_many_spikes = 0

        spikes_one = sorted([ (k,v) for k,v in lookup.items() if v==1 ])
        #smaller = sorted([ (dtc.ampl,dtc) for dtc in dtc_clone if dtc.boolean == True ])

        single_spike_found = bool(len(spikes_one))
        sub = np.array([x for x in lookup if lookup[x]==0])*units
        supra = np.array([x for x in lookup if lookup[x]>0])*units
        if self.verbose>1:
            if len(sub):
                print("Highest subthreshold current is %s" \
                      % (float(sub.max())*units))
            else:
                print("No subthreshold current was tested.")
            if len(supra):
                print("Lowest suprathreshold current is %s" \
                      % supra.min())
            else:
                print("No suprathreshold current was tested.")

        if len(sub) and len(supra) and single_spike_found:# or too_many_spikes<5:
            rheobase = supra.min()
            #rheobase = spikes_one[-1][1]


        elif too_many_spikes>=2:
            rheobase = None
        else:
            rheobase = None
        prediction['value'] = rheobase
        return prediction

    def threshold_FI(self, model, units, guess=None):
        lookup = {} # A lookup table global to the function below.

        def f(ampl):
            if float(ampl) not in lookup:

                uc = {'amplitude':ampl,'duration':DURATION,'delay':DELAY}
                model.inject_square_current(uc)
                n_spikes = model._backend.get_spike_count()

                self.n_spikes = n_spikes

                if self.verbose >= 5:
                    pass
                    print("Injected %s current and got %d spikes" % \
                            (ampl,n_spikes))
                lookup[float(ampl)] = n_spikes
                spike_counts = np.array([n for x,n in lookup.items() if n>0])

                if n_spikes and n_spikes <= spike_counts.min():
                    self.rheobase_vm = model.get_membrane_potential()

        max_iters = 45

        #evaluate once with a current injection at 0pA
        high=self.high
        small=self.small
        f(high)
        i = 0

        while True:
            #sub means below threshold, or no spikes
            sub = np.array([x for x in lookup if lookup[x]==0])*units
            #supra means above threshold, but possibly too high above threshold.

            supra = np.array([x for x in lookup if lookup[x]>0])*units
            #The actual part of the Rheobase test that is
            #computation intensive and therefore
            #a target for parellelization.

            if len(supra) and len(sub):
                delta = float(supra.min()) - float(sub.max())
                temp = [ v for v in lookup.values() if v>1 ]
                if len(temp):
                    too_many_spikes = np.min(temp)
                else:
                    too_many_spikes = 0

                if (delta < tolerance or (str(supra.min()) == str(sub.max()))):# and too_many_spikes<15:

                    break

            if i >= max_iters:
                break
            #Its this part that should be like an evaluate function that is passed to futures map.
            if len(sub) and len(supra):
                f((supra.min() + sub.max())/2)

            elif len(sub):
                f(max(small,sub.max()*2))

            elif len(supra):
                f(min(-small,supra.min()*2))
            i += 1

        return lookup

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        #print("%s: Observation = %s, Prediction = %s" % \
        #	 (self.name,str(observation),str(prediction)))
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(RheobaseTest,self).\
                        compute_score(observation, prediction)

            #if self.n_spikes > 1:
            #    score = score*self.n_spikes
             #self.bind_score(score,None,observation,prediction)
        return score

    def bind_score(self, score, model, observation, prediction):
        super(RheobaseTest,self).bind_score(score, model,
                                            observation, prediction)
        if self.rheobase_vm is not None:
            score.related_data['vm'] = self.rheobase_vm




class RheobaseTestP(VmTest):
     """
     A parallel Implementation of a Binary search algorithm,
     which finds a rheobase prediction.

     Strengths: this algorithm is faster than the serial class, present in this file for model backends that are not able to
     implement numba jit optimisation, which actually happens to be typical of a signifcant number of backends

     Weaknesses this serial class is significantly slower, for many backend implementations including raw NEURON
     NEURON via PyNN, and possibly GLIF.

     """
     def _extra(self):
         self.verbose = False

     # def __init__(self,other_current=None):
     #     self.other_current = other_current


     required_capabilities = (cap.ReceivesSquareCurrent,
                              cap.ProducesSpikes)
     #DELAY = 100.0*pq.ms
     # DURATION = 1000.0*pq.ms
     params = {'injected_square_current':
                 {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}
     name = "Rheobase test"
     description = ("A test of the rheobase, i.e. the minimum injected current "
                    "needed to evoke at least one spike.")
     units = pq.pA
     #tolerance  # Rheobase search tolerance in `self.units`.
     ephysprop_name = 'Rheobase'
     score_type = scores.ZScore

     def generate_prediction(self, model):
        def check_fix_range(dtc):
            '''
            Inputs: lookup, A dictionary of previous current injection values
            used to search rheobase
            Outputs: A boolean to indicate if the correct rheobase current was found
            and a dictionary containing the range of values used.
            If rheobase was actually found then rather returning a boolean and a dictionary,
            instead logical True, and the rheobase current is returned.
            given a dictionary of rheobase search values, use that
            dictionary as input for a subsequent search.
            '''

            steps = []
            dtc.rheobase = None
            sub, supra = get_sub_supra(dtc.lookup)

            #if 0. in supra and len(sub) == 0:
            #    dtc.boolean = True
            #    dtc.rheobase = None
            #    return dtc
            if (len(sub) + len(supra)) == 0:
                # This assertion would only be occur if there was a bug
                assert sub.max() <= supra.min()
            elif len(sub) and len(supra):
                # Termination criterion

                steps = np.linspace(sub.max(),supra.min(),cpucount+1)*pq.pA
                steps = steps[1:-1]*pq.pA
            elif len(sub):
                steps = np.linspace(sub.max(),2*sub.max(),cpucount+1)*pq.pA
                steps = steps[1:-1]*pq.pA
            elif len(supra):
                steps = np.linspace(supra.min()-100,supra.min(),cpucount+1)*pq.pA
                steps = steps[1:-1]*pq.pA

            dtc.current_steps = steps
            return dtc

        def get_sub_supra(lookup):
            sub, supra = [], []
            for current, n_spikes in lookup.items():
                if n_spikes == 0:
                    sub.append(current)
                elif n_spikes > 0:
                    supra.append(current)

            sub = np.array(sorted(list(set(sub))))
            supra = np.array(sorted(list(set(supra))))
            return sub, supra

        def check_current(dtc):
            '''
            Inputs are an amplitude to test and a virtual model
            output is an virtual model with an updated dictionary.
            '''
            dtc.boolean = False

            if dtc.backend is str('NEURON') or dtc.backend is str('jNEUROML'):
                LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
                dtc.model_path = LEMS_MODEL_PATH
                from neuronunit.models.reduced import ReducedModel, VeryReducedModel
                model = ReducedModel(dtc.model_path,name='vanilla', backend=(dtc.backend, {'DTC':dtc}))
                dtc.current_src_name = model._backend.current_src_name
                assert type(dtc.current_src_name) is not type(None)
                dtc.cell_name = model._backend.cell_name
            else:
                model = dtc.dtc_to_model()


            params = {'injected_square_current':
                      {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}
            ampl = dtc.ampl
            if float(ampl) not in dtc.lookup or len(dtc.lookup) == 0:
                uc = {'amplitude':dtc.ampl,'duration':DURATION,'delay':DELAY}
                dtc.run_number += 1
                model.attrs = dtc.attrs
                #model.set_attrs(**dtc.attrs)
                model.inject_square_current(uc)
                n_spikes = model.get_spike_count()
                dtc.previous = ampl

                dtc.rheobase = {}

                if float(ampl) < -1.0:
                    dtc.rheobase['value'] = None
                    dtc.boolean = True
                    return dtc

                if n_spikes == 1:
                    dtc.lookup[float(ampl)] = 1
                    dtc.rheobase['value'] = float(ampl)*pq.pA
                    dtc.boolean = True
                    return dtc

                dtc.lookup[float(ampl)] = n_spikes
            return dtc

        def init_dtc(dtc):
            '''
            Exploit memory of last model in genes.
            '''
            # check for memory and exploit it.
            if dtc.initiated == True:

                dtc = check_current(dtc)
                if dtc.boolean:

                    return dtc

                else:
                    # Exploit memory of the genes to inform searchable range.

                    # if this model has lineage, assume it didn't mutate that far away from it's ancestor.
                    # using that assumption, on first pass, consult a very narrow range, of test current injection samples:
                    # only slightly displaced away from the ancestors rheobase value.


                    if type(dtc.current_steps) is type(float):
                        dtc.current_steps = [ 0.75 * dtc.current_steps, 1.25 * dtc.current_steps ]
                    elif type(dtc.current_steps) is type(list):
                        dtc.current_steps = [ s * 1.25 for s in dtc.current_steps ]
                    dtc.initiated = True # logically unnecessary but included for readibility
            if dtc.initiated == False:

                dtc.boolean = False


                if str('PYNN') in dtc.backend:
                    steps = np.linspace(100,1000,7.0)
                else:

                    steps = np.linspace(1,550,7.0)

                steps_current = [ i*pq.pA for i in steps ]
                dtc.current_steps = steps_current
                dtc.initiated = True
            return dtc

        def find_rheobase(self, dtc):
            # This line should not be necessary:
            # a class, VeryReducedModel has been made to circumvent this.
            if hasattr(dtc,'model_path'):
                assert os.path.isfile(dtc.model_path), "%s is not a file" % dtc.model_path
            # If this it not the first pass/ first generation
            # then assume the rheobase value found before mutation still holds until proven otherwise.
            # dtc = check_current(model.rheobase,dtc)
            # If its not true enter a search, with ranges informed by memory
            cnt = 0
            sub = np.array([0,0]); supra = np.array([0,0])

            #use_diff = False
            if dtc.backend is 'GLIF':
                big = 100
            else:
                big = 16

            while dtc.boolean == False and cnt< big:

                # negeative spiker
                if len(sub):
                    if sub.max() < -1.0:
                        pass
                        #use_diff = True # differentiate the wave to look for spikes


                be = dtc.backend
                dtc_clones = [ dtc for i in range(0,len(dtc.current_steps)) ]
                for i,s in enumerate(dtc.current_steps):
                    dtc_clones[i] = copy.copy(dtc_clones[i])
                    dtc_clones[i].ampl = copy.copy(dtc.current_steps[i])
                dtc_clones = [d for d in dtc_clones if not np.isnan(d.ampl)]
                try:
                    b0 = db.from_sequence(dtc_clones, npartitions=npartitions)
                    dtc_clone = list(b0.map(check_current).compute())
                except:
                    set_clones = set([ float(d.ampl) for d in dtc_clones ])
                    dtc_clone = []
                    for dtc,sc in zip(dtc_clones,set_clones):
                        dtc = copy.copy(dtc)
                        dtc.ampl = sc*pq.pA
                        dtc = check_current(dtc)
                        dtc.backend = be
                        dtc_clone.append(dtc)

                if str("BHH") not in dtc.backend:
                    # take smallest spiking if multi spiking rheobase

                    smaller = sorted([ (dtc.ampl,dtc) for dtc in dtc_clone if dtc.boolean == True ])
                    if len(smaller):
                        return smaller[0][1]#int(1*len(smaller)/4.0)]
                else:
                    smaller = sorted([ (dtc.ampl,dtc) for dtc in dtc_clone if dtc.boolean == True ])
                    if len(smaller):
                        return smaller[-1][1]
                        #smaller[3*int(len(smaller)/4.0)][1]
                #for dtc in dtc_clone:
                #    if dtc.boolean == True:
                #        return dtc

                for d in dtc_clone:
                    dtc.lookup.update(d.lookup)
                dtc = check_fix_range(dtc)


                sub, supra = get_sub_supra(dtc.lookup)
                if len(supra) and len(sub):
                    delta = float(supra.min()) - float(sub.max())
                    if str("GLIF") in dtc.backend:
                        tolerance = 0.0
                    elif str("ADEXP") in dtc.backend:
                        tolerance = 0.0
                    else:
                        tolerance = 0.0000125
                        #tolerance = tolerance
                    if delta < tolerance or (str(supra.min()) == str(sub.max())):
                        if self.verbose >= 2:
                            import warning
                            warning(delta, 'a neuron, close to the edge! Multi spiking rheobase. # spikes: ',len(supra))
                        too_many_spikes = np.min([ v for v in dtc.lookup.values() if v>1 ])
                        if too_many_spikes>15:
                            import warning
                            warning(delta, 'elephant tests dont work well on high frequency spikes, therefore, this trace excluded. # spikes: ',len(supra))
                            dtc.rheobase = {}
                            dtc.rheobase['value'] = None
                            dtc.boolean = True
                            dtc.lookup[float(supra.min())] = len(supra)

                        else:
                            if len(supra)<15:

                                dtc.rheobase = float(supra.min())
                                dtc.boolean = True
                                dtc.lookup[float(supra.min())] = len(supra)
                            else:
                                dtc.rheobase = float(supra.min())
                                dtc.boolean = True
                                dtc.lookup[float(supra.min())] = len(supra)

                        return dtc


                if self.verbose >= 2:
                    print("Try %d: SubMax = %s; SupraMin = %s" % \
                    (cnt, sub.max() if len(sub) else None,
                    supra.min() if len(supra) else None))
                cnt += 1
            return dtc

        dtc = DataTC()
        dtc.attrs = {}
        for k,v in model.attrs.items():
            dtc.attrs[k] = v

        # this is not a perservering assignment, of value,
        # but rather a multi statement assertion that will be checked.
        dtc.backend = model.backend

        dtc = init_dtc(dtc)

        if hasattr(model,'orig_lems_file_path'):
            dtc.model_path = model.orig_lems_file_path
            assert os.path.isfile(dtc.model_path), "%s is not a file" % dtc.model_path

        prediction = {}

        temp = find_rheobase(self,dtc).rheobase

        if type(temp) is not type(None):
            if type(temp) is type({'dict':0}):
                if temp['value'] is None:
                    prediction['value'] = None
                    return None
                else:
                    prediction['value'] =  float(temp['value'])* pq.pA
            else:
                prediction['value'] =  float(temp)* pq.pA
        else:
            prediction = None
        return prediction

     def bind_score(self, score, model, observation, prediction):
         super(RheobaseTestP,self).bind_score(score, model,
                                            observation, prediction)

     def compute_score(self, observation, prediction):
         """Implementation of sciunit.Test.score_prediction."""
         score = None

         score = super(RheobaseTestP,self).\
                     compute_score(observation, prediction)
         return score
