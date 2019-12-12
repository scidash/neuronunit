import numpy as np
#from neuronunit.optimisation.optimisation_management import mint_generic_model
#from sciunit.models.runnable import RunnableModel
import quantities as qt
import copy

try:
    import asciiplotlib as apl
except:
    pass
class DataTC(object):
    '''
    Data Transport Container

    This Object class serves as a data type for storing rheobase search
    attributes and apriori model parameters,
    with the distinction that unlike the LEMS model this class
    can be cheaply transported across HOSTS/CPUs
    '''
    def __init__(self):
        self.lookup = {}
        self.rheobase = None
        self.previous = 0
        self.run_number = 0
        self.attrs = None
        self.steps = None
        self.name = None
        self.results = None
        self.fitness = None
        self.score = None
        self.scores = None

        self.boolean = False
        self.initiated = False
        self.delta = []
        self.evaluated = False
        self.results = {}
        self.searched = []
        self.searchedd = {}
        self.cached_attrs = {}
        self.predictions = {}
        self.observations = {}
        self.backend = None
        self.summed = None
        self.constants = None
        self.scores_ratio = None
        self.from_imputation = False
        self.preds = {}
        self.td = {}

    def get_ss(self):
        # get summed score
        if self.scores is not None:
            if len(self.scores) == 1:
                self.summed = list(self.scores.values())[0]
            else:
                self.summed = np.sum(list(self.scores.values()))
        else:
            self.summed = None
        return self.summed

    def add_constant(self):
        if self.constants is not None:
            self.attrs.update(self.constants)
        return #self.attrs

    def format_test(self):
        from neuronunit.optimisation.optimization_management import switch_logic, active_values, passive_values
        # pre format the current injection dictionary based on pre computed
        # rheobase values of current injection.
        # This is much like the hooked method from the old get neab file.
        self.protocols = {}
        if not hasattr(self,'tests'):
            self.tests = copy.copy(self.tests)
        if hasattr(self.tests,'keys'):# is type(dict):
            tests = [key for key in self.tests.values()]
            self.tests = switch_logic(tests)#,self.tests.use_rheobase_score)
        else:
            self.tests = switch_logic(self.tests)
        for k,v in enumerate(self.tests):
            self.protocols[k] = {}
            if hasattr(v,'passive'):#['protocol']:
                if v.passive == False and v.active == True:
                    keyed = self.protocols[k]#.params
                    self.protocols[k] = active_values(keyed,self.rheobase)
                elif v.passive == True and v.active == False:
                    keyed = self.protocols[k]#.params
                    self.protocols[k] = passive_values(keyed)
                    #tests.protocols[k] = self.protocols
            if v.name in str('RestingPotentialTest'):
                self.protocols[k]['injected_square_current']['amplitude'] = 0.0*qt.pA

        return self.tests
    def dtc_to_model(self):
        from neuronunit.models.very_reduced_sans_lems import VeryReducedModel
        model = VeryReducedModel(backend=self.backend)
        model.backend = self.backend
        model.attrs = self.attrs
        model.rheobase = self.rheobase

        #backend=(self.backend, {'DTC':self}))#, {'DTC':dtc}))
        # If  test taking data, and objects are present (observations etc).
        # Take the rheobase test and store it in the data transport container.
        if not hasattr(self,'scores'):
            self.scores = None
        if type(self.scores) is type(None):
            self.scores = {}
        #model.attrs = self.attrs
        model.scores = self.scores
        #model.rheobase = self.rheobase
        try:
            model.inj = self.params
        except:
            try:
                model.inj = self.vparams
            except:
                model.inj = None
        return model
    def dtc_to_gene(self):
        from neuronunit.optimisation.optimization_management import WSListIndividual
        print('warning translation dictionary should be used, to garuntee correct attribute order from random access dictionaries')
        if hasattr(self,'td'):
            gene = WSListIndividual()
            for i in self.td:
                gene.append(self.attrs[i])

        else:
            gene = WSListIndividual(list(self.attrs.values()))
        return gene

    def judge_test(self,index=0):
        model = self.dtc_to_model()
        if not hasattr(self,'tests'):
            print('warning dtc object does not contain NU-tests yet')
            return dtc

        ts = self.tests
        #this_test = ts[index]
        if not hasattr(self,'preds'):
            self.preds = {}
        for this_test in self.tests:
            this_test.setup_protocol(model)
            pred = this_test.extract_features(model,this_test.get_result(model))
            pred1 = this_test.generate_prediction(model)
            self.preds[this_test.name] = pred
        return self.preds

    def jt_ratio(self,index=0):
        from sciunit import scores
        model = self.dtc_to_model()
        if not hasattr(self,'tests'):
            print('warning dtc object does not contain NU-tests yet')
            return dtc

        ts = self.tests
        #this_test = ts[index]
        if not hasattr(self,'preds'):
            self.preds = {}
        tests = self.format_test()
        for this_test in self.tests:
            if this_test.passive:
                this_test.params['injected_square_current'] = {}
                this_test.params['injected_square_current']['amplitude'] = -10*qt.pA

                this_test.params['injected_square_current']['duration'] = 1000*qt.ms
                this_test.params['injected_square_current']['delay'] = 200*qt.ms

                this_test.setup_protocol(model)
                pred = this_test.extract_features(model,this_test.get_result(model))
            else:
                this_test.params['injected_square_current']['amplitude'] = self.rheobase
                this_test.params['injected_square_current']['duration'] = 1000*qt.ms
                this_test.params['injected_square_current']['delay'] = 200*qt.ms

                pred = this_test.generate_prediction(model)
            #self.preds[this_test.name] = pred
            ratio_type = scores.RatioScore
            temp = copy.copy(this_test.score_type)
            this_test.score_type = ratio_type
            try:
                print(this_test.name)
                self.rscores[rscores.name] = this_test.compute_score(this_test.observation,pred)

            except:
                this_test.score_type = temp
                #self.rscores = this_test.compute_score(model)
                self.rscores[rscores.name] = this_test.compute_score(this_test.observation,pred)

                self.failed = {}
                self.failed['pred'] = pred
                self.failed['observation'] = this_test.observation

        return self.preds

    def check_params(self):
        self.judge_test()
        print(self.rheobase)
        print(self.vparams)
        print(self.params)
        return self.preds
    def plot_obs(self,ow):
        '''
        assuming a waveform exists (observed waved-form) plot to terminal with ascii
        This is useful for debugging new backends, in bash big/fast command line orientated optimization routines.
        '''

        t = [float(f) for f in ow.times]
        v = [float(f) for f in ow.magnitude]
        fig = apl.figure()
        fig.plot(t, v, label=str('observation waveform from inside dtc: '), width=100, height=20)
        fig.show()

    def iap(self):
        '''
        Inject and plot to terminal with ascii
        This is useful for debugging new backends, in bash big/fast command line orientated optimization routines.
        '''

        model = self.dtc_to_model()
        #new_tests['RheobaseTest']
        if type(self.tests) is type({'1':1}):
            if 'RheobaseTest' in self.tests.keys():
                uset_t = self.tests['RheobaseTest']
            else:
                uset_t = self.tests['RheobaseTestP']

        elif type(self.tests) is type(list):
            for t in self.tests:
                if t.name in str('RheobaseTest'):
                    uset_t = t
                    break

        pms = uset_t.params
        pms['injected_square_current']['amplitude'] = self.rheobase
        print(pms)
        model.inject_square_current(pms['injected_square_current'])
        nspike = model.get_spike_train()
        print(nspike)
        vm = model.get_membrane_potential()
        t = [float(f) for f in vm.times]
        v = [float(f) for f in vm.magnitude]
        try:
            fig = apl.figure()
            fig.plot(t, v, label=str('observation waveform from inside dtc: ')+str(nspike), width=100, height=20)
            fig.show()
        except:
            import warnings
            print('ascii plot not installed')
        return vm
