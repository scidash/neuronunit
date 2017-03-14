"""Main Graupner-Brunel STDP example script"""

import numpy

import bluepyopt as bpop
import stdputil


def gbParam(params):
    """Create the parameter set for Graupner-Brunel model from an *individual*.

    :param individual: iterable
    :rtype : dict
    """
    gbparam = dict(
        theta_d=1.0,
        theta_p=1.3,
        rho_star=0.5,
        beta=0.75)  # Fixed params

    for param_name, param_value in params:
        gbparam[param_name] = param_value

    return gbparam

class BBEvaluator(bpop.evaluators.Evaluator):

    """Graupner-Brunel Evaluator"""

    def __init__(self):
        """Constructor"""

        super(BBEvaluator, self).__init__()
        # Graupner-Brunel model parameters and boundaries,
        # from (Graupner and Brunel, 2012)



        NDIM = 4

        self.params=[('vr',-65,-55),('a',0.015,0.045),('b',-0.0010,-0.0035)]
        #self.evaluator.objectives
        #self.param=['vr','a','b']

        #rov=[]
        #parameter.lower_bound=

        #rov0 = np.linspace(-65,-55,2)
        #rov1 = np.linspace(0.015,0.045,2)
        #rov2 = np.linspace(-0.0010,-0.0035,2)

        #rov0 = np.linspace(-65,-55,1000)
        #rov1 = np.linspace(0.015,0.045,1000)
        #rov2 = np.linspace(-0.0010,-0.0035,1000)
        #rov.append(rov0)
        #rov.append(rov1)
        #rov.append(rov2)
        seed_in=1
        #BOUND_LOW=[ np.min(i) for i in rov ]
        #BOUND_UP=[ np.max(i) for i in rov ]

        #self.objectives=[0 for i in range(0,3)]

        '''
        self.graup_params = [('tau_ca', 1e-3, 100e-3),
                             ('C_pre', 0.1, 20.0),
                             ('C_post', 0.1, 50.0),
                             ('gamma_d', 5.0, 5000.0),
                             ('gamma_p', 5.0, 2500.0),
                             ('sigma', 0.35, 70.7),
                             ('tau', 2.5, 2500.0),
                             ('D', 0.0, 50e-3),
                             ('b', 1.0, 100.0)]

        self.params = [bpop.parameters.Parameter
                       (param_name, bounds=(min_bound, max_bound))
                       for param_name, min_bound, max_bound in self.
                       graup_params]

        self.param_names = [param.name for param in self.params]

        self.protocols, self.sg, self.stdev, self.stderr = \
            stdputil.load_neviansakmann()

        self.objectives = [bpop.objectives.Objective(protocol.prot_id)
                           for protocol in self.protocols]

        '''
    def evaluate_with_lists(self, param_values):
        """Evaluate individual

        :param param_values: iterable
            Parameters list
        """
        #param_dict = self.get_param_dict(param_values)

        err = []

        def func2map(ind):

            for i, p in enumerate(param):
                name_value=str(ind[i])
                #reformate values.
                model.name=name_value
                if i==0:
                    attrs={'//izhikevich2007Cell':{p:name_value }}
                else:
                    attrs['//izhikevich2007Cell'][p]=name_value

            ind.attrs=attrs

            model.update_run_params(attrs)

            ind.params=[]
            for i in attrs['//izhikevich2007Cell'].values():
                if hasattr(ind,'params'):
                    ind.params.append(i)

            ind.results=model.results
            score = get_neab.suite.judge(model)
            ind.error = [ i.sort_key for i in score.unstack() ]
            return ind

        def evaluate(individual):#This method must be pickle-able for scoop to work.
            for i, p in enumerate(param):
                name_value=str(individual[i])
                #reformate values.
                model.name=name_value
                if i==0:
                    attrs={'//izhikevich2007Cell':{p:name_value }}
                else:
                    attrs['//izhikevich2007Cell'][p]=name_value

            individual.attrs=attrs

            model.update_run_params(attrs)

            individual.params=[]
            for i in attrs['//izhikevich2007Cell'].values():
                if hasattr(individual,'params'):
                    individual.params.append(i)

            individual.results=model.results
            score = get_neab.suite.judge(model)
            self.objectives= [ i.sort_key for i in score.unstack() ]
            individual.error = [ i.sort_key for i in score.unstack() ]
            #return ind
            #individual=func2map(individual)
            error=individual.error
            assert individual.results
            #print(rc.ids)
            #LOCAL_RESULTS.append(individual.results)
            err.append(error[0],error[1],error[2],error[3],error[4])

        return err

    '''
    def get_param_dict(self, param_values):
        """Build dictionary of parameters for the Graupner-Brunel model from an
        ordered list of values (i.e. an individual).

        :param param_values: iterable
            Parameters list
        """
        return gbParam(zip(self.param_names, param_values))

    def compute_sciunit(self, param_values):
        """Compute synaptic gain for all protocols.

        :param param_values: iterable
            Parameters list
        """
        param_dict = self.get_param_dict(param_values)

        syn_gain = [stdputil.protocol_outcome(protocol, param_dict)
                    for protocol in self.protocols]

        return syn_gain
    '''
