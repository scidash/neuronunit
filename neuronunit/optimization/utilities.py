
class DataTV(object):
    '''
    Data Transport Vessel

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
        self.results = {}
        self.searched = []
        self.searchedd = {}
'''
import unittest
tc = unittest.TestCase('__init__')
tc.assertEqual(var1, var2, msg=None)
tc.assertNotEqual(var1, var2, msg=None)
tc.assertTrue(expr, msg=None)
tc.assertRaises(exception, func, para, meters, ...)
'''

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


    def pop2map(attrs):
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
##
# Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &
##
import ipyparallel as ipp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
inv_pid_map = {}
dview = rc[:]

import os

#def p_imports():
from neuronunit.models import backends
from neuronunit.models.reduced import ReducedModel
import get_neab
print(get_neab.LEMS_MODEL_PATH)
new_file_path = '{0}{1}'.format(str(get_neab.LEMS_MODEL_PATH),int(os.getpid()))
print(new_file_path)

os.system('cp ' + str(get_neab.LEMS_MODEL_PATH)+str(' ') + new_file_path)
model = ReducedModel(new_file_path,name='vanilla',backend='NEURON')
model.load_model()
#    return

#dview.apply_sync(p_imports)
#p_imports()

def difference(v): # v is a tesst
    '''
    This method does not do what you would think
    from reading it.

    rescaling is the culprit. I suspect I do not
    understand how to rescale one unit with another
    compatible unit.
    '''

    import numpy as np
    print(v.prediction.keys())
    print(v.prediction.values())

    # The trick is.
    # prediction always has value. but observation 7 out of 8 times has mean.

    if 'value' in v.prediction.keys():
        unit_predictions = v.prediction['value']
        if 'mean' in v.observation.keys():
            unit_observations = v.observation['mean']
        elif 'value' in v.observation.keys():
            unit_observations = v.observation['value']



    if 'mean' in v.prediction.keys():
        unit_predictions = v.prediction['mean']
        if 'mean' in v.observation.keys():
            unit_observations = v.observation['mean']
        elif 'value' in v.observation.keys():
            unit_observations = v.observation['value']

    to_r_s = unit_observations.units
    unit_predictions = unit_predictions.rescale(to_r_s)
    unit_observations = unit_observations.rescale(to_r_s)
    unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )
    print(unit_delta)
    return float(unit_delta)

def pre_format(vms):
    import quantities as pq
    import copy
    vtest = {}
    import get_neab
    tests = get_neab.tests

    for k,v in enumerate(tests):
        vtest[k] = {}
        if k == 0:
            prediction = {}
            prediction['value'] = vms.rheobase * pq.pA

        if k != 0:
            prediction = None

        if k == 1 or k == 2 or k == 3:
            # Negative square pulse current.
            vtest[k]['duration'] = 100 * pq.ms
            vtest[k]['amplitude'] = -10 *pq.pA
            vtest[k]['delay'] = 30 * pq.ms

        if k == 0 or k == 4 or k == 5 or k == 6 or k == 7:
            # Threshold current.
            vtest[k]['duration'] = 1000 * pq.ms
            vtest[k]['amplitude'] = vms.rheobase * pq.pA
            vtest[k]['delay'] = 100 * pq.ms

        v = None
    return vtest

def evaluate(vms,weight_matrix = None):#This method must be pickle-able for ipyparallel to work.
    '''
    Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    value, that was found in a previous rheobase search.

    outputs: a tuple that is a compound error function that NSGA can act on.

    Assumes rheobase for each individual virtual model object (vms) has already been found
    there should be a check for vms.rheobase, and if not then error.
    Inputs a gene and a virtual model object.
    outputs are error components.
    '''

    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    import get_neab
    from itertools import repeat
    #import unittest
    #tc = unittest.TestCase('__init__')


    new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
    model = ReducedModel(new_file_path,name=str('vanilla'),backend='NEURON')
    model.load_model()
    assert type(vms.rheobase) is not type(None)
    #tests = get_neab.suite.tests
    model.set_attrs(attrs = vms.attrs)

    model.rheobase = vms.rheobase * pq.pA

    import copy
    tests = copy.copy(get_neab.tests)
    pre_fitness = []
    fitness = []
    differences = []
    fitness1 = []
    if float(vms.rheobase) <= 0.0:
        fitness1 = [ 125.0 for i in tests ]

    elif float(vms.rheobase) > 0.0:
        for k,v in enumerate(tests):
            if k == 0:
                v.prediction = {}
                v.prediction['value'] = vms.rheobase * pq.pA

            if k != 0:
                v.prediction = None


            score = v.judge(model,stop_on_error = False, deep_error = True)
            differences.append(difference(v))
            pre_fitness.append(float(score.sort_key))
            model.run_number += 1
            #vms.results[t]
    # outside of the test iteration block.
    if float(vms.rheobase) > 0.0:# and type(score) is not scores.InsufficientDataScore(None):
        for k,f in enumerate(copy.copy(pre_fitness)):

            fitness1.append(difference(v))

            if k == 5:
                from neuronunit import capabilities
                ans = model.get_membrane_potential()
                sw = capabilities.spikes2widths(ans)
                unit_observations = tests[5].observation['mean']

                #unit_observations = v.observation['value']
                to_r_s = unit_observations.units
                unit_predictions  = float(sw.rescale(to_r_s))
                fitness1[5] = float(np.abs( np.abs(float(unit_observations))-np.abs(float(unit_predictions))))
                #fitness1[5] = unit_delta
            if k == 0:
                fitness1.append(differences[0])
            if differences[0] > 10.0:
                if k != 0:
                    #fitness1.append(pre_fitness[k])
                    fitness1.append(pre_fitness[k] + 1.5 * differences[0] ) # add the rheobase error to all the errors.
                    assert fitness1[k] != pre_fitness[k]
            else:
                fitness1.append(pre_fitness[k])
            if k == 1:
                fitness1.append(differences[1])
            if differences[1] > 10.0 :
                if k != 1 and len(fitness1)>1 :
                    #fitness1.append(pre_fitness[k])
                    fitness1.append(pre_fitness[k] + 1.25 * differences[1] ) # add the rheobase error to all the errors.
                    assert fitness1[k] != pre_fitness[k]
        print(fitness1, fitness)
    pre_fitness = []
    return fitness1[0],fitness1[1],\
           fitness1[2],fitness1[3],\
           fitness1[4],fitness1[5],\
           fitness1[6],fitness1[7],



def pre_evaluate(vms):
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    import get_neab

    import copy
    # copying here is critical for get_neab
    tests = copy.copy(get_neab.tests)
    from itertools import repeat
    vtests = pre_format(copy.copy(vms))

    tests[0].prediction = {}
    tests[0].prediction['value'] = vms.rheobase * pq.pA

    if float(vms.rheobase) > 0.0:
        for k,t in enumerate(tests):
            '''
            can tests be re written such that it is more closure compatible?
            '''
            t.params = {}
            t.params['injected_square_current'] = {}
            t.params['injected_square_current']['duration'] = None
            t.params['injected_square_current']['amplitude'] = None
            t.params['injected_square_current']['delay'] = None

            for key, value in vtests[k].items():
                t.params['injected_square_current'][key] = value


            if k == 0:
                tests[k].prediction = {}
                tests[k].prediction['value'] = vms.rheobase * pq.pA

            new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
            model = ReducedModel(new_file_path,name=str('vanilla'),backend='NEURON')
            model.load_model()
            model.set_attrs(attrs = vms.attrs)
            print(t,model)
            score = t.judge(model,stop_on_error = False, deep_error = True)
            print(model.get_spike_count())
            v_m = model.get_membrane_potential()
            if 't' not in vms.results.keys():
                vms.results[t] = {}
                vms.results[t]['v_m'] = v_m
            elif 't' in vms.results.keys():
                vms.results[t]['v_m'] = v_m
    return vms

#from scoop import futures


def get_trans_dict(param_dict):
    trans_dict = {}
    for i,k in enumerate(list(param_dict.keys())):
        trans_dict[i]=k
    return trans_dict
import model_parameters
param_dict = model_parameters.model_params

def vm_to_ind(vm,td):
    '''
    Re instanting Virtual Model at every update vmpop
    is Noneifying its score attribute, and possibly causing a
    performance bottle neck.
    '''

    ind =[]
    for k in td.keys():
        ind.append(vm.attrs[td[k]])
    ind.append(vm.rheobase)
    return ind



def update_vm_pop(pop, trans_dict):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
    from itertools import repeat
    import numpy as np
    import copy
    pop = [toolbox.clone(i) for i in pop ]
    #import utilities
    def transform(ind):
        '''
        Re instanting Virtual Model at every update vmpop
        is Noneifying its score attribute, and possibly causing a
        performance bottle neck.
        '''
        vm = utilities.VirtualModel()

        param_dict = {}
        for i,j in enumerate(ind):
            param_dict[trans_dict[i]] = str(j)
        vm.attrs = param_dict
        vm.name = vm.attrs
        vm.evaluated = False
        return vm


    if len(pop) > 0:
        vmpop = dview.map_sync(transform, pop)
        vmpop = list(copy.copy(vmpop))
    else:
        # In this case pop is not really a population but an individual
        # but parsimony of naming variables
        # suggests not to change the variable name to reflect this.
        vmpop = transform(pop)
    return vmpop



def check_rheobase(vmpop,pop=None):
    '''
    inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
    outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
    Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
    compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
    If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
    corresponding virtual model objects.
    '''
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
        import copy
        import numpy as np
        import quantities as pq
        sub=[]
        supra=[]
        steps=[]
        vms.rheobase = 0.0
        for k,v in vms.lookup.items():
            vm.searchedd[v]=float(k)

            if v == 1:
                #A logical flag is returned to indicate that rheobase was found.
                vms.rheobase=float(k)
                vm.searched.append(float(k))
                vms.steps = 0.0
                vms.boolean = True
                return vms
            elif v == 0:
                sub.append(k)
            elif v > 0:
                supra.append(k)

        sub=np.array(sub)
        supra=np.array(supra)

        if len(sub)!=0 and len(supra)!=0:
            #this assertion would only be occur if there was a bug
            assert sub.max()<=supra.min()
        if len(sub) and len(supra):
            everything = np.concatenate((sub,supra))

            center = np.linspace(sub.max(),supra.min(),7.0)
            # The following code block probably looks counter intuitive.
            # Its job is to delete duplicated search values.
            # Ie everything is a list of 'everything' already explored.
            # It then inserts a bias corrected center position.
            for i,j in enumerate(centerl):
                if j in list(everything):

                    np.delete(center,i)
                    # delete the duplicated elements element, and replace it with a corrected
                    # center below.
            #delete the index
            #np.delete(center,np.where(everything is in center))
            #make sure that element 4 in a seven element vector
            #is exactly half way between sub.max() and supra.min()
            center[int(len(center)/2)+1]=(sub.max()+supra.min())/2.0
            steps = [ i*pq.pA for i in center ]

        elif len(sub):
            steps = np.linspace(sub.max(),2*sub.max(),7.0)
            np.delete(steps,np.array(sub))
            steps = [ i*pq.pA for i in steps ]

        elif len(supra):
            steps = np.linspace(-2*(supra.min()),supra.min(),7.0)
            np.delete(steps,np.array(supra))
            steps = [ i*pq.pA for i in steps ]

        vms.steps = steps
        vms.rheobase = None
        return copy.copy(vms)


    def check_current(ampl,vm):
        '''
        Inputs are an amplitude to test and a virtual model
        output is an virtual model with an updated dictionary.
        '''

        global model
        import quantities as pq
        import get_neab
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel

        new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(int(os.getpid()))
        model = ReducedModel(new_file_path,name=str('vanilla'),backend='NEURON')
        model.load_model()
        model.set_attrs(attrs = vms.attrs)


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
            model.set_attrs(attrs = vm.attrs)

            model.inject_square_current(current)
            vm.previous = ampl
            n_spikes = model.get_spike_count()
            vm.lookup[float(ampl)] = n_spikes
            if n_spikes == 1:
                vm.rheobase = float(ampl)

                vm.name = str('rheobase {0} parameters {1}'.format(str(current),str(model.params)))
                vm.boolean = True
                return vm

            return vm
        if float(ampl) in vm.lookup:
            return vm

    from itertools import repeat
    import numpy as np
    import copy
    import pdb
    import get_neab

    def init_vm(vm):
        if vm.initiated == True:
            # expand values in the range to accomodate for mutation.
            # but otherwise exploit memory of this range.

            if type(vm.steps) is type(float):
                vm.steps = [ 0.75 * vm.steps, 1.25 * vm.steps ]
            elif type(vm.steps) is type(list):
                vm.steps = [ s * 1.25 for s in vm.steps ]
            #assert len(vm.steps) > 1
            vm.initiated = True # logically unnecessary but included for readibility

        if vm.initiated == False:
            import quantities as pq
            import numpy as np
            vm.boolean = False
            steps = np.linspace(0,250,7.0)
            steps_current = [ i*pq.pA for i in steps ]
            vm.steps = steps_current
            vm.initiated = True
        return vm

    def find_rheobase(vm):
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        import get_neab

        new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
        #os.system('cp '+str(get_neab.LEMS_MODEL_PATH)+' '+new_file_path)
        model = ReducedModel(new_file_path,name=str('vanilla'),backend='NEURON')
        model.load_model()
        model.set_attrs(attrs = vm.attrs)

        cnt = 0
        # If this it not the first pass/ first generation
        # then assume the rheobase value found before mutation still holds until proven otherwise.
        if type(vm.rheobase) is not type(None):
            vm = check_current(vm.rheobase,vm)
        # If its not true enter a search, with ranges informed by memory
        cnt = 0
        while vm.boolean == False:
            for step in vm.steps:
                vm = check_current(step, vm)
                vm = check_fix_range(vm)
                cnt+=1
                print(cnt)
        return vm

    ## initialize where necessary.
    #import time
    vmpop = list(dview.map_sync(init_vm,vmpop))

    # if a population has already been evaluated it may be faster to let it
    # keep its previous rheobase searching range where this
    # memory of a previous range as acts as a guess as the next mutations range.

    vmpop = list(dview.map_sync(find_rheobase,vmpop))

    return vmpop, pop
