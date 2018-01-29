"""F/I neuronunit tests, e.g. investigating firing rates and patterns as a
function of input current"""

import os

from .base import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
from .. import optimization

class RheobaseTest(VmTest):
    """
    Tests the full widths of APs at their half-maximum
    under current injection.
    """
    def _extra(self):
        self.prediction = None
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
        begin_rh=time.time()
        lookup = self.threshold_FI(model, units)
        sub = np.array([x for x in lookup if lookup[x]==0])*units
        supra = np.array([x for x in lookup if lookup[x]>0])*units
        #self.verbose=True
        if self.verbose:
            if len(sub):
                print("Highest subthreshold current is %s" \
                      % (float(sub.max().round(2))*units))
            else:
                print("No subthreshold current was tested.")
            if len(supra):
                print("Lowest suprathreshold current is %s" \
                      % supra.min().round(2))
            else:
                print("No suprathreshold current was tested.")

        if len(sub) and len(supra):
            rheobase = supra.min()
        else:
            rheobase = None
        prediction['value'] = rheobase

        self.prediction = prediction
        return self.prediction

    def threshold_FI(self, model, units, guess=None):
        lookup = {} # A lookup table global to the function below.

        def f(ampl):
            if float(ampl) not in lookup:
                current = self.params.copy()['injected_square_current']
                #This does not do what you would expect.
                #due to syntax I don't understand.
                #updating the dictionary keys with new values doesn't work.

                uc = {'amplitude':ampl}
                current.update(uc)
                ampl = float(ampl)

                model.inject_square_current(current)

                n_spikes = model.get_spike_count()
                self.verbose = True
                if self.verbose:
                    print("Injected %s current and got %d spikes" % \
                            (ampl,n_spikes))
                lookup[float(ampl)] = n_spikes
                spike_counts = np.array([n for x,n in lookup.items() if n>0])
                if n_spikes and n_spikes <= spike_counts.min():
                    self.rheobase_vm = model.get_membrane_potential()

        max_iters = 10

        #evaluate once with a current injection at 0pA
        high=self.high
        small=self.small

        f(high)
        i = 0

        while True:

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

from neuronunit.optimization.data_transport_container import DataTC


class RheobaseTestP(VmTest):



     """
     A parallel version of test Rheobase.
     Tests the full widths of APs at their half-maximum
     under current injection.

     """
     def _extra(self,dview=None):
         self.prediction = None
         #self.dview = dview

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

     score_type = scores.RatioScore

     def generate_prediction(self, model):

        '''
        inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
        outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
        Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
        compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
        If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
        corresponding virtual model objects.
        '''
        from ipyparallel import depend, require, dependent
        import ipyparallel as ipp

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
            import pdb
            import copy
            import numpy as np
            import quantities as pq
            from ipyparallel import depend, require, dependent
            print('stuck here')
            sub=[]
            supra=[]
            steps=[]

            dtc.rheobase = 0.0
            for k,v in dtc.lookup.items():
                dtc.searchedd[v]=float(k)

                if v == 1:
                    #A logical flag is returned to indicate that rheobase was found.
                    dtc.rheobase = float(k)
                    #dtc.searched.append(float(k))
                    dtc.steps = 0.0
                    dtc.boolean = True
                    return dtc
                elif v == 0:
                    sub.append(k)
                elif v > 0:
                    supra.append(k)

            sub = np.array(sub)
            supra = np.array(supra)
            print(sub,'sub')
            print(supra,'supra')

            if 0. in supra and len(sub) == 0:
                dtc.boolean = True
                dtc.rheobase = -1
                #score = scores.InsufficientDataScore(None)

                return dtc


            if len(sub)!=0 and len(supra)!=0:
                #this assertion would only be occur if there was a bug
                assert sub.max()<=supra.min()
            if len(sub) and len(supra):
                center = list(np.linspace(sub.max(),supra.min(),9.0))
                center = [ i for i in center if not i == sub.max() ]
                center = [ i for i in center if not i == supra.min() ]
                center[int(len(center)/2)+1]=(sub.max()+supra.min())/2.0
                steps = [ i*pq.pA for i in center ]

            elif len(sub):
                steps = list(np.linspace(sub.max(),2*sub.max(),9.0))
                steps = [ i for i in steps if not i == sub.max() ]
                steps = [ i*pq.pA for i in steps ]

            elif len(supra):
                step = list(np.linspace(-2*(supra.min()),supra.min(),9.0))
                steps = [ i for i in steps if not i == supra.min() ]
                steps = [ i*pq.pA for i in steps ]

            dtc.steps = steps
            dtc.rheobase = None
            return copy.copy(dtc)

        @require('quantities')
        def check_current(ampl,dtc):
            '''
            Inputs are an amplitude to test and a virtual model
            output is an virtual model with an updated dictionary.
            '''
            import os
            from neuronunit.models.reduced import ReducedModel
            import neuronunit
            import copy
            #LEMS_MODEL_PATH = os.path.join(neuronunit.__path__[0],
            LEMS_MODEL_PATH = str(neuronunit.__path__[0])+str('/models/NeuroML2/LEMS_2007One.xml')
            dtc.model_path = LEMS_MODEL_PATH
            model = ReducedModel(dtc.model_path,name='vanilla',backend = dtc.backend)


            DELAY = 100.0*pq.ms
            DURATION = 1000.0*pq.ms
            params = {'injected_square_current':
                      {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

            dtc = copy.copy(dtc)
            ampl = float(ampl)
            if ampl not in dtc.lookup or len(dtc.lookup)==0:
                current = params.copy()['injected_square_current']
                uc = {'amplitude':ampl}
                current.update(uc)
                current = {'injected_square_current':current}

                dtc.run_number += 1
                model.set_attrs(dtc.attrs)
                model.inject_square_current(current)
                dtc.previous = ampl
                n_spikes = model.get_spike_count()
                print(n_spikes, 'n_spikes')
                dtc.lookup[float(ampl)] = n_spikes


                if n_spikes == 1:
                    dtc.rheobase = float(ampl)
                    dtc.boolean = True
                    return dtc

                return dtc
            if float(ampl) in dtc.lookup:
                return dtc


        @require('itertools','numpy','copy')
        def init_dtc(dtc):
            import numpy as np
            import copy

            if dtc.initiated == True:
                # expand values in the range to accomodate for mutation.
                # but otherwise exploit memory of this range.

                if type(dtc.steps) is type(float):
                    dtc.steps = [ 0.75 * dtc.steps, 1.25 * dtc.steps ]
                elif type(dtc.steps) is type(list):
                    dtc.steps = [ s * 1.25 for s in dtc.steps ]
                dtc.initiated = True # logically unnecessary but included for readibility

            if dtc.initiated == False:
                import quantities as pq
                import numpy as np
                dtc.boolean = False
                steps = np.linspace(0,250,7.0)
                steps_current = [ i*pq.pA for i in steps ]
                dtc.steps = steps_current
                dtc.initiated = True
            return dtc

        @require('itertools')
        def find_rheobase(self, dtc):
            import dask.bag as db

            #if not hasattr(self,'dview'):
            #    import ipyparallel as ipp
            #    rc = ipp.Client(profile='default')
            #        self.dview = rc[:]
            cnt = 0
            assert os.path.isfile(dtc.model_path), "%s is not a file" % dtc.model_path
            # If this it not the first pass/ first generation
            # then assume the rheobase value found before mutation still holds until proven otherwise.
            # dtc = check_current(model.rheobase,dtc)
            # If its not true enter a search, with ranges informed by memory


            cnt = 0
            while dtc.boolean == False:
                #dtc.searched.append(dtc.steps)

                dtcs = [ dtc for s in dtc.steps ]

                #from dask.distributed import Client
                #client = Client()
                # create two dask bags for
                # a sequence gathered data type, that is partitioned into eight partition sizes.
                #ncores = client.ncores()
                #partitions = len(list(ncores.values()))
                b0 = db.from_sequence(dtc.steps, npartitions=8)
                b1 = db.from_sequence(dtcs, npartitions=8)


                # By default they rely on the multiprocessing scheduler, which has its own set of known limitations (see Shared Memory)
                # Bags are immutable and so you can not change individual elements
                # Bag operations tend to be slower than array/dataframe computations in the same way that standard Python containers tend to be slower than NumPy arrays and Pandas dataframes.
                # Bag.groupby is slow. You should try to use Bag.foldby if possible. Using Bag.foldby requires more thought.


                dtcpop = list(db.map(check_current,b0,b1).compute())
                for dtc_clone in dtcpop:
                    dtc.lookup.update(dtc_clone.lookup)
                dtc = check_fix_range(dtc)
                cnt += 1
            return dtc

        dtc = DataTC()
        dtc.attrs = {}
        for k,v in model.attrs.items():
            dtc.attrs[k]=v
        dtc = init_dtc(dtc)
        dtc.model_path = model.orig_lems_file_path
        dtc.backend = model.backend
        assert os.path.isfile(dtc.model_path), "%s is not a file" % dtc.model_path
        prediction = {}
        prediction['value'] = find_rheobase(self,dtc).rheobase * pq.pA
        print('weird arse prediction',prediction, type(prediction))
        if type(prediction['value']) is type(None):
            prediction['value'] = -1 * pq.pA
        if type(prediction['value']) is type(float) and prediction['value'] <= 0.0:
            prediction['value'] = -1 * pq.pA
        return prediction

     def bind_score(self, score, model, observation, prediction):
         super(RheobaseTestP,self).bind_score(score, model,
                                            observation, prediction)

     def compute_score(self, observation, prediction):
         """Implementation of sciunit.Test.score_prediction."""

         score = None
         if type(prediction['value']) is type(None):
            prediction['value'] = -1 * pq.pA
            return scores.InsufficientDataScore(None)

         if float(prediction['value']) <= 0.0:
            print('gets here')
            # if rheobase is negative discard the model essentially.
            prediction['value'] = -1 * pq.pA
            return scores.InsufficientDataScore(None)

         score = super(RheobaseTestP,self).\
                     compute_score(observation, prediction)
         return score
