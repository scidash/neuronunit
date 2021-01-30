"""F/I neuronunit tests.

For example, investigating firing rates and patterns as a
function of input current.
"""
import warnings

import os
import multiprocessing
global cpucount
npartitions = cpucount = multiprocessing.cpu_count()
from .base import np, pq, ncap, VmTest, scores, AMPL, DELAY, DURATION
import quantities
import neuronunit


import dask.bag as db
import quantities as pq
import numpy as np
import copy
import time
import copy
import dask
from neuronunit.capabilities.spike_functions import get_spike_waveforms, spikes2amplitudes, threshold_detection

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

    def _extra(self):
        self.prediction = {}
        self.high = 900*pq.pA
        self.small = 0*pq.pA
        self.rheobase_vm = None
        self.verbose = False
    def __init__(self,observation=None,prediction=None,target_number_spikes=1,name="RheobaseTest"):
        self._extra()
        super(RheobaseTest,self).__init__(observation=observation,name=name)
        self.prediction = prediction
        self.target_number_spikes = target_number_spikes

    required_capabilities = (ncap.ReceivesSquareCurrent,
                             ncap.ProducesSpikes)

    name = "Rheobase test"
    description = ("A test of the rheobase, i.e. the minimum injected current "
                   "needed to evoke at least one spike.")
    units = pq.pA
    ephysprop_name = 'Rheobase'
    default_params = dict(VmTest.default_params)
    default_params.update({'amplitude': 100*pq.pA,
                           'duration': DURATION,
                           'delay': DELAY,
                           'tolerance': 1.0*pq.pA})

    params_schema = dict(VmTest.params_schema)
    params_schema.update({'tolerance': {'type': 'current', 'min': 1, 'required': False}})

    def condition_model(self, model):
        if not 't_max' in self.params:
            self.params['t_max'] = 2000.0*pq.ms
        else:
            model.set_run_params(t_stop=self.params['t_max'])

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.

        ##
        # Handle case that the
        # Test constructor __init__ originated from
        # a file that is older than this source code
        # because it was recovered from pickle.
        ##
        if not hasattr(self,'target_number_spikes'):
            self.target_number_spikes=1


        self.condition_model(model)
        prediction = {'value': None}
        model.rerun = True

        if self.observation is not None:
            try:
                units = self.observation['value'].units
            except KeyError:
                units = self.observation['mean'].units
        else:
            units = pq.pA
        lookup = self.threshold_FI(model, units)
        ##
        # New code
        ##
        temp = [ v for v in lookup.values() if v>self.target_number_spikes ]
        if len(temp):
            too_many_spikes = np.min(temp)
        else:
            too_many_spikes = 0

        spikes_one = sorted([ (k,v) for k,v in lookup.items() if v==self.target_number_spikes ])

        if len(spikes_one)>=3:
            prediction['value'] = np.abs(spikes_one[0][0]*units)
            return prediction


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

        if len(sub) and len(supra) and single_spike_found:
            rheobase = supra.min()
        elif too_many_spikes>=5:
            rheobase = None
        else:
            rheobase = None
        prediction['value'] = rheobase

        if len(supra) and single_spike_found and str("BHH") in str(model._backend.name):
            prediction['value'] = sorted(supra)[1]
        if len(supra) and single_spike_found and str("HH") in str(model._backend.name):
            prediction['value'] = sorted(supra)[0]
        self.prediction = prediction
        return prediction

    def extract_features(self,model):
        prediction = self.generate_prediction(model)
        self.prediction = prediction
        return prediction

    def threshold_FI(self, model, units, guess=None):
        """Use binary search to generate an FI curve including rheobase."""
        lookup = {}  # A lookup table global to the function below.

        def f(ampl):
            if float(ampl) not in lookup:
                if False:
                    uc = {'amplitude':ampl,'duration':DURATION,'delay':DELAY}

                    model.inject_square_current(uc)
                    n_spikes = model._backend.get_spike_count()
                    assert n_spikes == model.get_spike_count()

                current = self.get_injected_square_current()

                current['amplitude'] = ampl*pq.pA
                if "JIT_" in model.backend:
                    try:
                        model.inject_square_current(**current)
                    except:
                        model._backend.inject_square_current(**current)

                    n_spikes = model.get_spike_count()
                    assert n_spikes == model.get_spike_count()

                else:

                    model._backend.inject_square_current(**current)
                    n_spikes = model._backend.get_spike_count()

                    #if self.target_num_spikes == 1:
                        # ie this is rheobase search
                        #vm = model.get_membrane_potential()
                        #if vm[-1]>0 and n_spikes==1:
                            # this means current was not strong enough
                            # to evoke an early spike.
                            # the voltage deflection did not come back down below zero.
                            # treat this as zero spikes because a slightly higher
                            # spike will give a cleaner rheobase waveform.
                if n_spikes == self.target_number_spikes:

                    self.n_spikes = n_spikes
                if self.verbose >= 2:
                    print("Injected %s current and got %d spikes" % \
                            (ampl,n_spikes))
                lookup[float(ampl)] = n_spikes
                spike_counts = \
                    np.array([n for x, n in lookup.items() if n > 0])
                if n_spikes and n_spikes <= spike_counts.min():
                    self.rheobase_vm = model._backend.get_membrane_potential()

        max_iters = 40

        # evaluate once with a current injection at 0pA
        high = self.high
        small = self.small

        f(high)
        i = 0

        while True:

            # sub means below threshold, or no spikes
            sub = np.array([x for x in lookup if lookup[x] == 0])*units
            # supra means above threshold,
            # but possibly too high above threshold.

            supra = np.array([x for x in lookup if lookup[x] > 0])*units
            # The actual part of the Rheobase test that is
            # computation intensive and therefore
            temp_ = [ v for v in lookup.values() if v==self.target_number_spikes ]

            if len(supra) and len(sub):
                delta = float(supra.min()) - float(sub.max())
                temp = [ v for v in lookup.values() if v>self.target_number_spikes ]
                if len(temp):
                    too_many_spikes = np.min(temp)
                else:
                    too_many_spikes = 0
                if 'tolerance' not in self.params.keys():
                    tolerance = 0.0000001*pq.pA
                else:
                    tolerance = float(self.params['tolerance'].rescale(pq.pA))

                if (str(supra.min()) == str(sub.max())):

                    break

            if i >= max_iters:
                break
            # Its this part that should be like an evaluate function
            # that is passed to futures map.
            if len(sub) and len(supra):
                f((supra.min() + sub.max())/2)

            elif len(sub):
                f(max(small,sub.max()*10))

            elif len(supra):
                f(min(-small, supra.min()*2))
            i += 1

        return lookup

    def compute_score(self,observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        from sciunit.scores import BooleanScore

        if type(self.score_type) == BooleanScore:
            print('warning using unusual score type')
        if prediction is None or \
           (isinstance(prediction, dict) and prediction['value'] is None):
            score = scores.InsufficientDataScore(None)
        else:

            score = super(RheobaseTest, self).\
                            compute_score(observation, prediction)#max
        return score

    def bind_score(self, score, model, observation, prediction):
        """Bind additional attributes to the test score."""
        super(RheobaseTest, self).bind_score(score, model,
                                             observation, prediction)
        if self.rheobase_vm is not None:
            score.related_data['vm'] = self.rheobase_vm


class RheobaseTestP(RheobaseTest):
    """Parallel implementation of a binary search to test the rheobase.

    Strengths: this algorithm is faster than the serial class, present in this
    file for model backends that are not able to implement numba jit
    optimization, which actually happens to be typical of a signifcant number
    of backends.
    """


    def _extra(self,target_number_spikes=1):
        self.verbose = False
        self.target_number_spikes = 1

    def __init__(self,observation=None,prediction=None,target_number_spikes=1,name=""):
        self._extra()
        super(RheobaseTest,self).__init__(observation=observation,name=name)
        self.prediction = prediction
        self.target_number_spikes = target_number_spikes


    required_capabilities = (ncap.ReceivesSquareCurrent,
                             ncap.ProducesSpikes)

    name = "Rheobase test"
    description = ("A test of the rheobase, i.e. the minimum injected current "
                   "needed to evoke at least one spike.")
    units = pq.pA
    ephysprop_name = 'Rheobase'

    get_rheobase_vm = True
    default_params = dict(VmTest.default_params)
    default_params.update({'amplitude': 100*pq.pA,
                           'duration': DURATION,
                           'delay': DELAY,
                           'tolerance': 1.0*pq.pA})

    params_schema = dict(VmTest.params_schema)
    params_schema.update({'tolerance': {'type': 'current', 'min': 1, 'required': False}})

    def condition_model(self, model):
        if not 't_max' in self.params:
            self.params['t_max'] = 2000.0*pq.ms
        else:
            model.set_run_params(t_stop=self.params['t_max'])

    default_params = dict(VmTest.default_params)
    default_params.update({'amplitude': 100*pq.pA,
                           'duration': DURATION,
                           'duration': DELAY,
                           'tolerance': 1.0*pq.pA})
    params_schema = dict(VmTest.params_schema)
    params_schema.update({'tolerance': {'type': 'current', 'min': 1, 'required': False}})
    units = pq.pA

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

                steps = np.linspace(sub.max(),supra.min(),cpucount)*pq.pA
                steps = steps[1:-1]
            elif len(sub):

                steps = np.linspace(sub.max(),10*sub.max(),cpucount)*pq.pA
                steps = steps[1:-1]
            elif len(supra):
                steps = np.linspace(supra.min()-100,supra.min(),cpucount)*pq.pA
                steps = steps[1:-1]

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
        #@dask.delayed
        def check_current(dtc):
            '''
            Inputs are an amplitude to test and a virtual model
            output is an virtual model with an updated dictionary.
            '''
            dtc.boolean = False

            model = dtc.dtc_to_model()
            self.condition_model(model)

            ampl = dtc.ampl
            if float(ampl) not in dtc.lookup or len(dtc.lookup) == 0:

                current = {'amplitude':ampl,'duration':DURATION,'delay':DELAY}
                float(current['delay']) > 100
                current['amplitude'] = ampl
                model.inject_square_current(current)
                n_spikes = model.get_spike_count()


                dtc.previous = ampl
                dtc.rheobase = {}
                if n_spikes == self.target_number_spikes:
                    dtc.lookup[float(ampl)] = self.target_number_spikes
                    dtc.rheobase['value'] = ampl
                    dtc.boolean = True

                    return dtc

                dtc.lookup[float(ampl)] = n_spikes
            return dtc

        def init_dtc(dtc):
            '''
            Exploit memory of last model in genes.
            # Exploit memory of the genes to inform searchable range.
            # if this model has lineage, assume it didn't mutate that far away from it's ancestor.
            # using that assumption, on first pass, consult a very narrow range, of test current injection samples:
            # only slightly displaced away from the ancestors rheobase value.


            if type(dtc.current_steps) is type(float):
                dtc.current_steps = [ 0.80 * dtc.current_steps, 1.20 * dtc.current_steps ]
            elif type(dtc.current_steps) is type(list):
                dtc.current_steps = [ s * 1.25 for s in dtc.current_steps ]
            dtc.initiated = True # logically unnecessary but included for readibility

            '''
            # check for memory and exploit it.
            if dtc.initiated == True:
                dtc = check_current(dtc)
                if dtc.boolean:
                    return dtc
            if dtc.initiated == False:
                dtc.boolean = False
                steps = np.linspace(0.0,550.0,cpucount)
                steps_current = [ i*pq.pA for i in steps ]
                dtc.current_steps = steps_current
                dtc.initiated = True
            return dtc

        def find_rheobase(self, global_dtc):
            units = pq.pA

            # If this it not the first pass/ first generation
            # then assume the rheobase value found before mutation still holds until proven otherwise.
            # dtc = check_current(model.rheobase,dtc)
            # If its not true enter a search, with ranges informed by memory
            cnt = 0
            sub = np.array([0,0]);
            supra = np.array([0,0])

            big = 20

            while global_dtc.boolean == False and cnt< big:

                if len(sub):
                    if sub.max() < -1.0:
                        pass



                dtc_clones = [ global_dtc for i in range(0,len(global_dtc.current_steps)) ]
                for i,s in enumerate(global_dtc.current_steps):
                    dtc_clones[i] = copy.copy(dtc_clones[i])
                    dtc_clones[i].ampl = copy.copy(global_dtc.current_steps[i])
                dtc_clones = [d for d in dtc_clones if not np.isnan(d.ampl)]
                set_clones = set([ float(d.ampl) for d in dtc_clones ])
                dtc_clone = []

                for dtc_local,sc in zip(dtc_clones,set_clones):
                    dtc_local = copy.copy(dtc_local)
                    dtc_local.ampl = sc*pq.pA
                    dtc_clone.append(dtc_local)
                bag = db.from_sequence(dtc_clone,npartitions=8)
                dtc_clone = list(bag.map(check_current).compute())
                spikes_one = sorted([ (dtc.ampl,dtc) for dtc in dtc_clone if dtc.boolean == True ])
                if len(spikes_one)>=2:
                    return spikes_one[0][0]


                for d in dtc_clone:
                    global_dtc.lookup.update(d.lookup)
                dtc = check_fix_range(global_dtc)
                sub, supra = get_sub_supra(dtc.lookup)
                if len(supra) and len(sub):

                    delta = float(supra.min()) - float(sub.max())

                    tolerance = 0.0
                    if delta < tolerance or (str(supra.min()) == str(sub.max())):
                        if self.verbose >= 2:
                            print(delta, 'a neuron, close to the edge! Multi spiking rheobase. # spikes: ',len(supra))
                        too_many_spikes = np.min([ v for v in dtc.lookup.values() if v>self.target_number_spikes ])
                        if too_many_spikes>10:

                            dtc.rheobase = {}
                            dtc.rheobase['value'] = None
                            dtc.boolean = True
                            dtc.lookup[float(supra.min())] = len(supra)

                        else:
                            if len(supra)<=10:

                                dtc.rheobase = float(supra.min())*units
                                dtc.boolean = True
                                dtc.lookup[float(supra.min())] = len(supra)
                            else:
                                dtc.rheobase = float(supra.min())
                                dtc.boolean = True
                                dtc.lookup[float(supra.min())] = len(supra)*units
                        #print(dtc.rheobase)
                        return dtc.rheobase


                if self.verbose >= 2:
                    print("Try %d: SubMax = %s; SupraMin = %s" % \
                    (cnt, sub.max() if len(sub) else None,
                    supra.min() if len(supra) else None))
                cnt += 1
            return dtc
        from neuronunit.optimisation.data_transport_container import DataTC

        dtc = DataTC()
        dtc.attrs = {}
        for k,v in model.attrs.items():
            dtc.attrs[k] = v

        # this is not a perservering assignment, of value,
        # but rather a multi statement assertion that will be checked.
        dtc.backend = model.backend

        dtc = init_dtc(dtc)
        prediction = {}
        temp = find_rheobase(self,dtc)#.rheobase
        if type(temp) is type(pq.pA):
            prediction['value'] =  temp
            return prediction
        if type(temp) is not type(None):
            if hasattr(temp,'rheobase'):
                temp = temp.rheobase
            if type(temp) is type(dict()):
                if temp['value'] is None:
                    prediction['value'] = None
                else:
                    prediction['value'] =  float(temp['value'])* pq.pA
            else:
                prediction['value'] =  temp #float(temp)* pq.pA
        else:
            prediction['value'] = None
        self.prediction = prediction
        return prediction


    def extract_features(self,model):
        prediction = self.generate_prediction(model)
        return prediction
    '''

    def bind_score(self, score, model, observation, prediction):
        super(RheobaseTestP,self).bind_score(score, model,
                                            observation, prediction)
    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        score = None

        score = super(RheobaseTestP,self).\
                     compute_score(observation, prediction)
        return score
    '''
