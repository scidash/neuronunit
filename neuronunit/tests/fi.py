"""F/I neuronunit tests, e.g. investigating firing rates and patterns as a
function of input current"""

import os
import multiprocessing
global cpucount
npartitions = cpucount = multiprocessing.cpu_count()
from .base import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
from .. import optimization

from neuronunit.optimization.data_transport_container import DataTC
import os
import quantities
import neuronunit
from neuronunit.models.reduced import ReducedModel
import dask.bag as db
import quantities as pq
import numpy as np
import copy
import pdb

class RheobaseTest(VmTest):
    """
    Tests the full widths of APs at their half-maximum
    under current injection.
    """
    def _extra(self):
        self.prediction = {}
        self.high = 300*pq.pA
        self.small = 0*pq.pA
        self.rheobase_vm = None

    required_capabilities = (cap.ReceivesSquareCurrent,
                             cap.ProducesSpikes)

    params = {'injected_square_current':
                {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

    name = "Rheobase test"
    description = ("A test of the rheobase, i.e. the minimum injected current "
                   "needed to evoke at least one spike.")
    units = pq.pA
    ephysprop_name = 'Rheobase'
    score_type = scores.RatioScore

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
        import time
        begin_rh = time.time()
        lookup = self.threshold_FI(model, units)
        sub = np.array([x for x in lookup if lookup[x]==0])*units
        supra = np.array([x for x in lookup if lookup[x]>0])*units
        if self.verbose:
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
        if len(sub) and len(supra):
            rheobase = supra.min()
        else:
            rheobase = None
        #prediction['value'] = rheobase
        prediction['value'] = rheobase
        return prediction

    def threshold_FI(self, model, units, guess=None):
        lookup = {} # A lookup table global to the function below.

        def f(ampl):
            if float(ampl) not in lookup:
                current = self.params.copy()['injected_square_current']

                uc = {'amplitude':ampl}
                current.update(uc)
                model.inject_square_current(current)
                n_spikes = model.get_spike_count()
                if self.verbose:
                    print("Injected %s current and got %d spikes" % \
                            (ampl,n_spikes))
                lookup[float(ampl)] = n_spikes
                spike_counts = np.array([n for x,n in lookup.items() if n>0])
                if n_spikes and n_spikes <= spike_counts.min():
                    self.rheobase_vm = model.get_membrane_potential()

        max_iters = 5

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
        if prediction['value'] is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(RheobaseTest,self).\
                        compute_score(observation, prediction)
            #self.bind_score(score,None,observation,prediction)
        return score

    def bind_score(self, score, model, observation, prediction):
        super(RheobaseTest,self).bind_score(score, model,
                                            observation, prediction)
        if self.rheobase_vm is not None:
            score.related_data['vm'] = self.rheobase_vm

class RheobaseTestP(VmTest):
     """
     A parallel version of test Rheobase.
     Tests the full widths of APs at their half-maximum
     under current injection.

     """
     required_capabilities = (cap.ReceivesSquareCurrent,
                              cap.ProducesSpikes)
     DELAY = 100.0*pq.ms
     DURATION = 1000.0*pq.ms
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
        '''
        inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
        outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
        Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
        compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
        If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
        corresponding virtual model objects.
        '''
        #if type(model.rmemory) is not type(None):
            #import pdb; pdb.set_trace()
            #dtc.ampl = model.rmemory
        #    print(model.rmemory)
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

            if 0. in supra and len(sub) == 0:
                dtc.boolean = True
                #dtc.rheobase = None # should be None

                dtc.rheobase = -1 # should be None
                #score = scores.InsufficientDataScore(None)
                return dtc

            if (len(sub) + len(supra)) == 0:
                #this assertion would only be occur if there was a bug
                assert sub.max()<=supra.min()
            elif len(sub) and len(supra):
                if supra.min() - sub.max() < 5: # Less than 5 pA
                    dtc.rheobase = float(supra.min())
                    dtc.boolean = True
                    return dtc
                steps = np.linspace(sub.max(),supra.min(),cpucount-1)
                steps = steps[1:-2]*pq.pA
                #print("Here are the steps: ",steps)
                #print(steps)
            elif len(sub):
                steps = list(np.linspace(sub.max(),2*sub.max(),cpucount-1))
                steps = steps[1::]*pq.pA
                #[ i for i in steps if not i == sub.max() ]
                #print(steps)
                #steps = [ i*pq.pA for i in steps ]
            elif len(supra):
                steps = list(np.linspace(-2*(supra.min()),supra.min(),cpucount-1))
                # There is a misconception that -1 refers to the second last element, and -0 refers to the last element
                # https://stackoverflow.com/questions/39838900/getting-second-to-last-element-in-list#39838913
                steps = steps[::-2]*pq.pA
                #print(steps)
                #steps = [ i for i in steps if not i == supra.min() ]
                #steps = [ i*pq.pA for i in steps ]

            dtc.current_steps = steps

            #dtc.rheobase = None
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
            LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
            dtc.model_path = LEMS_MODEL_PATH

            model = ReducedModel(dtc.model_path,name='vanilla', backend=(dtc.backend, {'DTC':dtc}))

            dtc.current_src_name = model._backend.current_src_name
            assert type(dtc.current_src_name) is not type(None)
            dtc.cell_name = model._backend._cell_name
            #model.set_attrs(dtc.attrs)

            DELAY = 100.0*pq.ms
            DURATION = 1000.0*pq.ms
            params = {'injected_square_current':
                      {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

            ampl = float(dtc.ampl)
            if ampl not in dtc.lookup or len(dtc.lookup) == 0:
                current = params.copy()['injected_square_current']
                uc = {'amplitude':ampl}
                current.update(uc)
                current = {'injected_square_current':current}
                dtc.run_number += 1
                model.set_attrs(dtc.attrs)
                model.inject_square_current(current)
                dtc.previous = ampl
                n_spikes = model.get_spike_count()
                dtc.lookup[float(ampl)] = n_spikes
                if n_spikes == 1:
                    dtc.rheobase = float(ampl)
                    print(dtc.rheobase)
                    dtc.boolean = True
                    return dtc
            return dtc

        def init_dtc(dtc):

            # check for memory and exploit it.
            if dtc.initiated == True:
                #dtc.ampl = dtc.rheobase

                dtc = check_current(dtc)
                if dtc.boolean:
                    return dtc
                else:
                    # expand values in the range to accomodate for mutation.
                    # but otherwise exploit memory of the genes to inform searchable range.

                    if type(dtc.current_steps) is type(float):
                        dtc.current_steps = [ 0.75 * dtc.current_steps, 1.25 * dtc.current_steps ]
                    elif type(dtc.current_steps) is type(list):
                        dtc.current_steps = [ s * 1.25 for s in dtc.current_steps ]
                    dtc.initiated = True # logically unnecessary but included for readibility

            if dtc.initiated == False:

                dtc.boolean = False
                steps = np.linspace(0.01,250,7.0)
                steps_current = [ i*pq.pA for i in steps ]
                dtc.current_steps = steps_current
                dtc.initiated = True
            return dtc

        def find_rheobase(self, dtc):

            assert os.path.isfile(dtc.model_path), "%s is not a file" % dtc.model_path
            # If this it not the first pass/ first generation
            # then assume the rheobase value found before mutation still holds until proven otherwise.
            # dtc = check_current(model.rheobase,dtc)
            # If its not true enter a search, with ranges informed by memory
            cnt = 0
            while dtc.boolean == False and cnt< 10:

                #dtc.current_steps = list(filter(lambda cs: cs !=0.0 , dtc.current_steps))
                dtc_clones = [ copy.copy(dtc) for i in range(0,len(dtc.current_steps)) ]
                for i,s in enumerate(dtc.current_steps):

                    dtc_clones[i].ampl = dtc.current_steps[i]
                dtc_clones = [ d for d in dtc_clones if not np.isnan(d.ampl) ]
                b0 = db.from_sequence(dtc_clones, npartitions=npartitions)
                dtc_clone = list(b0.map(check_current).compute())

                for d in dtc_clone:
                    dtc.lookup.update(d.lookup)
                dtc = check_fix_range(dtc)
                cnt += 1
                sub, supra = get_sub_supra(dtc.lookup)
                print("Try %d: SubMax = %s; SupraMin = %s" % \
                (cnt, sub.max().round(1) if len(sub) else None,
                 supra.min().round(1) if len(supra) else None))
            return dtc

        dtc = DataTC()
        dtc.attrs = {}
        for k,v in model.attrs.items():
            dtc.attrs[k] = v

        # this is not a perservering assignment, of value,
        # but rather a multi statement assertion that will be checked.

        dtc = init_dtc(dtc)
        dtc.model_path = model.orig_lems_file_path
        dtc.backend = model.backend
        assert os.path.isfile(dtc.model_path), "%s is not a file" % dtc.model_path

        #import dask.array as da
        #from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler


        prediction = {}
        prediction['value'] = find_rheobase(self,dtc).rheobase * pq.pA

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
