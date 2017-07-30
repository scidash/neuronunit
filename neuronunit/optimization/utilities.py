
class VirtualModel(object):
    '''
    This is a pickable dummy clone
    version of the NEURON simulation model
    It does not contain an actual model, but it can be used to
    wrap the real model.
    This Object class serves as a data type for storing rheobase search
    attributes and other useful parameters,
    with the distinction that unlike the NEURON model this class
    can be transported across HOSTS/CPUs
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
        self.boolean = False
        self.initiated = False
        self.delta = []
        self.evaluated = False

class Utilities:
    def __init__(self):
        self = self
        self.get_neab = None

    def set_attrs(self,get_neab,dview):
        self.get_neab = get_neab

        self.dview = dview


    def model2map(param_dict):#This method must be pickle-able for scoop to work.
        vm=VirtualModel()
        vm.attrs={}
        #print(param_dict)
        for k,v in param_dict.items():
            vm.attrs[k]=v
        return vm


    def pop2map(iter_arg):
        '''
        Just a sanity check an otherwise impotent method
        '''
        vm=VirtualModel()

        vm.attrs=attrs
        model.load_model()
        model.update_run_params(vm.attrs)
        #print(model.params,attrs,vm.attrs)
        return (model, vm)

    def error2map(iter_):#This method must be pickle-able for scoop to work.
        '''
        Inputs an iterable list, a neuron unit test object suite of neuron model
        tests of emperical data reproducibility.
        '''
        iter_arg,value = iter_

        return_list=[]
        try:
            assert iter_arg.attrs is not type(None)
        except:
            print('exception occured {0}'.format(type(iter_arg.attrs)))
        model.update_run_params(iter_arg.attrs)
        import quantities as qt

        score = None
        sane = False
        if type(value) is not type(None):
            assert value >= 0
            sane = self.get_neab.suite.tests[3].sanity_check(value * pq.pA,model)
            uc = {'amplitude':value}
            current = params.copy()['injected_square_current']
            current.update(uc)
            current = {'injected_square_current':current}
            import copy
            model.inject_square_current(current)
            init_vm = model.results['vm']
            n_spikes = model.get_spike_count()
            assert n_spikes == 1 or n_spikes == 0
            self.get_neab.suite.tests[0].prediction={}
            self.get_neab.suite.tests[0].prediction['value'] = value * pq.pA
            return_list = []
            error = []# re-declare error this way to stop it growing.
            if sane == True and n_spikes == 1:
                for i in [3,4,5]:
                    self.get_neab.suite.tests[i].params['injected_square_current']['amplitude']=value*pq.pA
                score = self.get_neab.suite.judge(model)#passing in model, changes model
                skv = list(filter(lambda item: type(item) is not type(None), score.sort_key.values[0]))
                try:
                    assert len(skv) != 0
                    error= [ np.abs(i) for i in skv ]
                except:
                    error = [ 10.0 for i in range(0,7) ]
                model.name=str('rheobase {0} parameters {1}'.format(str(value),str(model.params)))
                import neuronunit.capabilities as cap
                spikes_numbers=[]
                model.run_number+=1
                return_list.append(np.sum(error))
                return_list.append(error)
                return_list.append(iter_arg.attrs)
                return_list.append(value*pq.pA)
            elif sane == False:
                #create a nominally high error
                error = [ 10.0 for i in range(0,7) ]
                return_list.append(np.sum(error))
                return_list.append(error)
                return_list.append(iter_arg.attrs)
                return_list.append(value*pq.pA)


        return return_list


    def check_fix_range2(vms):
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
        sub=[]
        supra=[]
        steps=[]
        vms.rheobase=0.0
        for k,v in vms.lookup.items():
            if v==1:
                #A logical flag is returned to indicate that rheobase was found.
                vms.rheobase=float(k)
                vms.steps = 0.0
                return (True,vms)
            elif v==0:
                sub.append(k)
            elif v>0:
                supra.append(k)

        sub=np.array(sub)
        supra=np.array(supra)

        if len(sub)!=0 and len(supra)!=0:
            #this assertion would only be wrong if there was a bug
            print(str(bool(sub.max()>supra.min())))
            assert not sub.max()>supra.min()
        if len(sub) and len(supra):
            everything=np.concatenate((sub,supra))

            center = np.linspace(sub.max(),supra.min(),7.0)
            centerl = list(center)
            for i,j in enumerate(centerl):
                if i in list(everything):
                    np.delete(center,i)
                    del centerl[i]
            #delete the index
            #np.delete(center,np.where(everything is in center))
            #make sure that element 4 in a seven element vector
            #is exactly half way between sub.max() and supra.min()
            center[int(len(center)/2)+1]=(sub.max()+supra.min())/2.0
            steps = [ i*pq.pA for i in center ]

        elif len(sub):
            steps2 = np.linspace(sub.max(),2*sub.max(),7.0)
            np.delete(steps2,np.array(sub))
            steps = [ i*pq.pA for i in steps2 ]

        elif len(supra):
            steps2 = np.linspace(-2*(supra.min()),supra.min(),7.0)
            np.delete(steps2,np.array(supra))
            steps = [ i*pq.pA for i in steps2 ]

        vms.steps = steps
        vms.rheobase = None
        return (False,copy.copy(vms))

    def check_current2(ampl,vm):
        '''
        Inputs are an amplitude to test and a virtual model
        output is an virtual model with an updated dictionary.
        '''
        global model
        import quantities as pq
        import get_neab
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel

        DELAY = 100.0*pq.ms
        DURATION = 1000.0*pq.ms
        params = {'injected_square_current':
                  {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}


        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str(vm.name),backend='NEURON')
        model.load_model()
        model.update_run_params(vm.attrs)

        #print(type(model),type(self.model))
        print(model)

        if float(ampl) not in vm.lookup or len(vm.lookup)==0:

            current = params.copy()['injected_square_current']

            uc = {'amplitude':ampl}
            current.update(uc)
            current = {'injected_square_current':current}
            vm.run_number += 1
            model.update_run_params(vm.attrs)
            model.inject_square_current(current)
            vm.previous = ampl
            n_spikes = model.get_spike_count()
            vm.lookup[float(ampl)] = n_spikes
            if n_spikes == 1:
                #model.rheobase_memory=float(ampl)
                vm.rheobase=float(ampl)
                print(type(vm.rheobase))
                print('current {0} spikes {1}'.format(vm.rheobase,n_spikes))
                vm.name = str('rheobase {0} parameters {1}'.format(str(current),str(model.params)))
                return vm

            return vm
        if float(ampl) in vm.lookup:
            return vm


    def searcher2(self,vms):
        '''
        inputs f a function to evaluate. rh_param a tuple with element 1 boolean, element 2 float or list
        and a  virtual model object.
        '''
        import numpy as np
        import copy
        import get_neab
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        import quantities as pq
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str(vms.name),backend='NEURON')
        model.load_model()
        #print(type(model),type(self.model))
        print(model)
        from itertools import repeat

        lookuplist = []
        cnt = 0

        # boolean a switch that should turn true if rheobase is
        # found by switching true the search is terminated.
        boolean=False
        model.update_run_params(vms.attrs)


        while boolean == False and cnt < 7:
            # commit to a loop of 6 iterations. To find the rheobase
            if boolean:
                return vms
            else:
                # Basically reset the model
                model.update_run_params(vms.attrs)
                if type(vms.steps) is type(None):
                    steps = np.linspace(50,150,7.0)
                    steps_current = [ i*pq.pA for i in steps ]
                    vms.steps = steps_current
                    assert type(vms.steps) is not type(None)
                # vms.steps and range should be the same thing
                # Basically reset the model
                model.update_run_params(vms.attrs)
                vml = list(map(self.check_current2,copy.copy(vms.steps),repeat(vms)))
                #vml = list(self.dview.map(self.check_current2,copy.copy(vms.steps),repeat(vms)))
                for v in vml:
                    vms.lookup.update(v.lookup)
                    print()
                boolean,vms = self.check_fix_range2(copy.copy(vms))
            cnt+=1 #update an iterator
        return vms

    def check_fix_range(vms):
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
        import numpy as np
        import quantities as pq
        sub=[]
        supra=[]
        steps=[]
        vms.rheobase=0.0
        for k,v in vms.lookup.items():
            if v==1:
                #A logical flag is returned to indicate that rheobase was found.
                vms.rheobase=float(k)
                vms.steps=0.0
                return (True,vms)
            elif v==0:
                sub.append(k)
            elif v>0:
                supra.append(k)

        sub=np.array(sub)
        supra=np.array(supra)

        if len(sub)!=0 and len(supra)!=0:
            #this assertion would only be wrong if there was a bug
            print(str(bool(sub.max()>supra.min())))
            assert not sub.max()>supra.min()
        if len(sub) and len(supra):
            everything=np.concatenate((sub,supra))

            center = np.linspace(sub.max(),supra.min(),7.0)
            centerl = list(center)
            for i,j in enumerate(centerl):
                if i in list(everything):
                    np.delete(center,i)
                    del centerl[i]
            #delete the index
            #np.delete(center,np.where(everything is in center))
            #make sure that element 4 in a seven element vector
            #is exactly half way between sub.max() and supra.min()
            center[int(len(center)/2)+1]=(sub.max()+supra.min())/2.0
            steps = [ i*pq.pA for i in center ]

        elif len(sub):
            steps2 = np.linspace(sub.max(),2*sub.max(),7.0)
            np.delete(steps2,np.array(sub))
            steps = [ i*pq.pA for i in steps2 ]

        elif len(supra):
            steps2 = np.linspace(-2*(supra.min()),supra.min(),7.0)
            np.delete(steps2,np.array(supra))
            steps = [ i*pq.pA for i in steps2 ]

        vms.steps=steps
        vms.rheobase=None
        return (False,vms)

    def check_current(ampl,vm):
        '''
        Inputs are an amplitude to test and a virtual model
        output is an virtual model with an updated dictionary.
        '''
        global model
        import quantities as pq

        DELAY = 100.0*pq.ms
        DURATION = 1000.0*pq.ms
        params = {'injected_square_current':
                  {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

        if float(ampl) not in vm.lookup or len(vm.lookup)==0:

            current = params.copy()['injected_square_current']

            uc = {'amplitude':ampl}
            current.update(uc)
            current = {'injected_square_current':current}
            vm.run_number += 1
            model.update_run_params(vm.attrs)
            model.inject_square_current(current)
            vm.previous = ampl
            n_spikes = model.get_spike_count()
            vm.lookup[float(ampl)] = n_spikes
            if n_spikes == 1:
                model.rheobase_memory=float(ampl)
                vm.rheobase=float(ampl)
                print(type(vm.rheobase))
                print('current {0} spikes {1}'.format(vm.rheobase,n_spikes))
                vm.name = str('rheobase {0} parameters {1}'.format(str(current),str(model.params)))
                return vm

            return vm
        if float(ampl) in vm.lookup:
            return vm


    def searcher(self,vms1):
        '''
        inputs f a function to evaluate. rh_param a tuple with element 1 boolean, element 2 float or list
        and a  virtual model object.
        '''
        import numpy as np
        import get_neab
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        import quantities as pq
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='place_holder',backend='NEURON')
        model.load_model()
        print(model)
        from itertools import repeat
        (vms,rh_param) = vms1
        #name_str = 'parameters {0}'.format(str(vms.attrs))
        #print('{0}'.format(vms1))
        #model.name = name_str
        if rh_param[0] == True:
            return rh_param[1]
        lookuplist = []
        cnt = 0
        boolean=False

        model.update_run_params(vms.attrs)
        while boolean == False and cnt < 12:
            if len(model.params)==0:
                model.update_run_params(vms.attrs)
            if type(rh_param[1]) is float:
                #if its a single value educated guess
                if model.rheobase_memory == None:
                    model.rheobase_memory = rh_param[1]
                vms = self.check_current(model.rheobase_memory , vms)
                #print(vms)
                model.update_run_params(vms.attrs)
                boolean,vms = self.check_fix_range(vms)
                if boolean:
                    return vms
                else:
                    #else search returned none type, effectively false
                    rh_param = (None,None)

            elif len(vms.lookup) == 0 and type(rh_param[1]) is list:
                #If the educated guess failed, or if the first attempt is parallel vector of samples
                assert vms is not None
                returned_list = list(map(self.check_current,rh_param[1],repeat(vms)))
                for v in returned_list:
                    vms.lookup.update(v.lookup)
                boolean,vms = self.check_fix_range(vms)
                assert vms != None
                if boolean:
                    return vms

            else:
                #Finally if a parallel vector of samples failed zoom into the
                #smallest relevant interval and re-sample at a higher resolution
                returned_list=[]
                if type(vms.steps) is type(None):
                    steps = np.linspace(50,150,7.0)
                    steps_current = [ i*pq.pA for i in steps ]
                    vms.steps = steps_current
                    assert type(vms.steps) is not type(None)
                returned_list = list(map(self.check_current,vms.steps,repeat(vms)))
                for v in returned_list:
                    vms.lookup.update(v.lookup)
                boolean,vms = self.check_fix_range(vms)
                if boolean:
                    return vms
            cnt+=1
        return vms
    '''
    def rheobase_checking(vmpop, rh_value=None):
        #
        #This method needs to be checked carefully in case it duplicates work
        #
        from itertools import repeat
        import pdb
        def bulk_process(vm,rh_value):
            #
            #package arguments and call the parallel searcher
            #
            if type(vm) is not type(None):
                rh_param = (False,rh_value)
                vm = searcher(rh_param,vm)
                return vm

        if type(vmpop) is not type(list):
            return bulk_process(vmpop,0)

        elif type(vmpop) is type(list):
            vmtemp = []
            if type(rh_value) is type(None):
                vmtemp = bulk_process(copy.copy(vmpop),0)
                #vmtemp = list(self.map(bulk_process,vmpop,repeat(0)))
            elif type(rh_value) is not type(None):
                vmtemp = bulk_process(vmpop,rh_value)
                #vmtemp = list(self.map(bulk_process,vmpop,rh_value))
            return vmtemp
    '''
    def hypervolume_contrib(front, **kargs):
        """Returns the hypervolume contribution of each individual. The provided
        *front* should be a set of non-dominated individuals having each a
        :attr:`fitness` attribute.
        """
        import numpy
        # Must use wvalues * -1 since hypervolume use implicit minimization
        # And minimization in deap use max on -obj
        wobj = numpy.array([ind.fitness.wvalues for ind in front]) * -1
        ref = kargs.get("ref", None)
        if ref is None:
            ref = numpy.max(wobj, axis=0) + 1

        total_hv = hv.hypervolume(wobj, ref)

        def contribution(i):
            # The contribution of point p_i in point set P
            # is the hypervolume of P without p_i
            return total_hv - hv.hypervolume(numpy.concatenate((wobj[:i], wobj[i+1:])), ref)

        # Parallelization note: Cannot pickle local function
        return map(contribution, range(len(front)))



    def plot_stats(pop,logbook):
        evals, gen ,std, avg, max_, min_ = logbook.select("evals","gen","avg", "max", "min", "std")
        x = list(range(0,len(avg)))

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.semilogy(x, avg, "--b")
        plt.semilogy(x, max_, "--b")
        plt.semilogy(x, min_, "-b")
        plt.semilogy(x, fbest, "-c")
        #plt.semilogy(x, sigma, "-g")
        #plt.semilogy(x, axis_ratio, "-r")
        plt.grid(True)
        plt.title("blue: f-values, green: sigma, red: axis ratio")

        plt.subplot(2, 2, 2)
        plt.plot(x, best)
        plt.grid(True)
        plt.title("Object Variables")

        plt.subplot(2, 2, 3)
        plt.semilogy(x, std)
        plt.grid(True)
        plt.title("Standard Deviations in All Coordinates")
        plt.savefig('GA_stats_vs_generation.png')
        f=open('worst_candidate.txt','w')
        if len(vmpop)!=0:
            f.write(str(vmpop[-1].attrs))
            f.write(str(vmpop[-1].rheobase))

        f.write(logbook.stream)
        f.close()
        score_matrixt=[]
        if len(vmpop)!=0:

            score_matrixt.append((vmpop[0].error,vmpop[0].attrs,vmpop[0].rheobase))
            score_matrixt.append((vmpop[1].error,vmpop[1].attrs,vmpop[1].rheobase))

            score_matrixt.append((vmpop[-1].error,vmpop[-1].attrs,vmpop[-1].rheobase))
        import pickle
        import pickle
        with open('score_matrixt.pickle', 'wb') as handle:
            pickle.dump(score_matrixt, handle)

        with open('vmpop.pickle', 'wb') as handle:
            pickle.dump(vmpop, handle)



    def test_to_model_plot(vms,local_test_methods):
        #from neuronunit.tests import get_neab
        tests = self.get_neab.suite.tests
        import matplotlib.pyplot as plt
        import copy
        global model
        global nsga_matrix
        model.local_run()
        model.update_run_params(vms.attrs)
        #model.re_init(vms.attrs)
        tests = None
        tests = self.get_neab.suite.tests
        tests[0].prediction={}
        tests[0].prediction['value']=vms.rheobase * qt.pA
        tests[0].params['injected_square_current']['amplitude']=vms.rheobase * qt.pA
        #score = get_neab.suite.judge(model)#pass
        if local_test_methods in [4,5,6]:
            tests[local_test_methods].params['injected_square_current']['amplitude']=vms.rheobase * qt.pA
        #model.results['vm'] = [ 0 ]
        model.re_init(vms.attrs)
        tests[local_test_methods].generate_prediction(model)
        injection_trace = np.zeros(len(model.results['t']))

        trace_size = int(len(model.results['t']))
        injection_trace = np.zeros(trace_size)

        end = len(model.results['t'])#/delta
        delay = int((float(self.get_neab.suite.tests[0].params['injected_square_current']['delay'])/1600.0 ) * end )
        #delay = get_neab.suite.tests[0].params['injected_square_current']['delay']['value']/delta
        duration = int((float(1100.0)/1600.0) * end ) # delta
        injection_trace[0:int(delay)] = 0.0
        injection_trace[int(delay):int(duration)] = vms.rheobase
        injection_trace[int(duration):int(end)] = 0.0
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(model.results['t'],model.results['vm'],label='$V_{m}$ (mV)')
        axarr[0].set_xlabel(r'$V_{m} (mV)$')
        axarr[0].set_xlabel(r'$time (ms)$')

        axarr[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        axarr[1].plot(model.results['t'],injection_trace,label='$I_{i}$(pA)')
        if vms.rheobase > 0:
            axarr[1].set_ylim(0, 2*vms.rheobase)
        if vms.rheobase < 0:
            axarr[1].set_ylim(2*vms.rheobase,0)
        axarr[1].set_xlabel(r'$current injection (pA)$')
        axarr[1].set_xlabel(r'$time (ms)$')

        axarr[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

        model.re_init(vms.attrs)
        tests = None
        tests = self.get_neab.suite.tests
        #tests[0].prediction={}
        tests[0].prediction={}
        tests[0].prediction['value']=vms.attrs * qt.pA
        tests[0].params['injected_square_current']['amplitude']=vms.rheobase * qt.pA


        if local_test_methods in [4,5,6]:
            tests[local_test_methods].params['injected_square_current']['amplitude']=vms.rheobase * qt.pA

        #model.results['vm'] = [ 0 ]
        model.re_init(vms.attrs)
        #tests[local_test_methods].judge(model)
        tests[local_test_methods].generate_prediction(model)
        injection_trace = np.zeros(len(model.results['t']))
        delta = model.results['t'][1]-model.results['t'][0]

        trace_size = int(len(model.results['t']))
        injection_trace = np.zeros(trace_size)

        end = len(model.results['t'])#/delta
        delay = int((float(self.get_neab.suite.tests[0].params['injected_square_current']['delay'])/1600.0 ) * end )
        #delay = get_neab.suite.tests[0].params['injected_square_current']['delay']['value']/delta
        duration = int((float(1100.0)/1600.0) * end ) # delta
        injection_trace[0:int(delay)] = 0.0
        injection_trace[int(delay):int(duration)] = vms.rheobase
        injection_trace[int(duration):int(end)] = 0.0
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(model.results['t'],model.results['vm'],label='$V_{m}$ (mV)')
        axarr[0].set_xlabel(r'$V_{m} (mV)$')
        axarr[0].set_xlabel(r'$time (ms)$')

        axarr[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        axarr[1].plot(model.results['t'],injection_trace,label='$I_{i}$(pA)')
        if vms.rheobase > 0:
            axarr[1].set_ylim(0, 2*vms.rheobase)
        if vms.rheobase < 0:
            axarr[1].set_ylim(2*vms.rheobase,0)
        axarr[1].set_xlabel(r'$current injection (pA)$')
        axarr[1].set_xlabel(r'$time (ms)$')

        axarr[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

        plt.title(str(tests[local_test_methods]))
        plt.savefig(str('best_solution')+str('.png'))
        #plt.clf()
        model.results['vm']=None
        model.results['t']=None
        tests[local_test_methods].related_data=None
        local_test_methods=None
        return 0
