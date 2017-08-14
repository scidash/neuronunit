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
    import numpy as np
    print(v.prediction.keys())
    print(v.prediction.values())
    #for k,v in v.prediction.items():
    #    print(k,v,'debugging key value in difference')
    #import pdb; pdb.set_trace()

    if 'value' in v.prediction.keys():
        unit_predictions = v.prediction['value']
        unit_observations = v.observation['value']

    if 'mean' in v.prediction.keys():
        unit_predictions = v.prediction['mean']
        unit_observations = v.observation['mean']

    #unit_observations = v.observation['value']
    to_r_s = unit_observations.units
    unit_predictions = unit_predictions.rescale(to_r_s)
    unit_observations = unit_observations.rescale(to_r_s)
    unit_delta = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )
    #import pdb; pdb.set_trace()
    print(unit_delta)
    v = None
    return float(unit_delta)

def evaluate(vms):#This method must be pickle-able for ipyparallel to work.
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
    model.update_run_params(vms.attrs)
    model.rheobase = vms.rheobase * pq.pA

    import copy
    tests = copy.copy(get_neab.tests)
    pre_fitness = []
    fitness = []
    differences = []

    if float(vms.rheobase) <= 0.0:
        fitness = [ 125.0 for i in tests ]

    elif float(vms.rheobase) > 0.0:
        for k,v in enumerate(tests):
            if k == 0:
                v.prediction = {}
                v.prediction['value'] = vms.rheobase * pq.pA

            if k != 0:
                v.prediction = None

            if k == 1 or k == 2 or k == 3:
                # Negative square pulse current.
                v.params['injected_square_current']['duration'] = 100 * pq.ms
                v.params['injected_square_current']['amplitude'] = -10 *pq.pA
                v.params['injected_square_current']['delay'] = 30 * pq.ms
            if k == 0 or k == 4 or k == 5 or k == 6 or k == 7:
                # Threshold current.
                v.params['injected_square_current']['duration'] = 1000 * pq.ms
                v.params['injected_square_current']['amplitude'] = vms.rheobase * pq.pA
                v.params['injected_square_current']['delay'] = 100 * pq.ms

            score = v.judge(model,stop_on_error = False, deep_error = True)
            differences.append(difference(v))
            pre_fitness.append(float(score.sort_key))
            model.run_number += 1

    # outside of the test iteration block.
    if float(vms.rheobase) > 0.0:# and type(score) is not scores.InsufficientDataScore(None):
        for k,f in enumerate(copy.copy(pre_fitness)):

            if k == 5:
                from neuronunit import capabilities
                ans = model.get_membrane_potential()
                sw = capabilities.spikes2widths(ans)
                unit_observations = tests[5].observation['mean']

                #unit_observations = v.observation['value']
                to_r_s = unit_observations.units
                unit_predictions = unit_predictions.rescale(sw)
                unit_predictions  = sw.rescale(to_r_s)
                fitness1[5] = np.abs( np.abs(unit_observations)-np.abs(unit_predictions) )
                #fitness1[5] = unit_delta
            if k == 0:
                fitness1.append(differences[0])
            if differences[0] > 10.0:
                if k != 0:
                    fitness1.append(pre_fitness[k] + 1.5 * differences[0] ) # add the rheobase error to all the errors.
                    assert fitness1[k] != pre_fitness[k]
            else:
                fitness1.append(pre_fitness[k])
            if k == 1:
                fitness1.append(differences[1])
            if differences[1] > 10.0 :
                if k != 1 and len(fitness1)>1 :
                    fitness1.append(pre_fitness[k] + 1.25 * differences[1] ) # add the rheobase error to all the errors.
                    assert fitness1[k] != pre_fitness[k]
        print(fitness1, fitness)
    pre_fitness = []
    return fitness1[0],fitness1[1],\
           fitness1[2],fitness1[3],\
           fitness1[4],fitness1[5],\
           fitness1[6],fitness1[7],



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
        vms.rheobase=0.0
        for k,v in vms.lookup.items():
            if v==1:
                #A logical flag is returned to indicate that rheobase was found.
                vms.rheobase=float(k)
                vms.steps = 0.0
                vms.boolean = True
                return vms
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
            everything = np.concatenate((sub,supra))

            center = np.linspace(sub.max(),supra.min(),7.0)
            centerl = list(center)
            # The following code block probably looks counter intuitive.
            # Its job is to delete duplicated search values.
            # Ie everything is a list of everything already explored.
            # It then makes a corrected center position.
            for i,j in enumerate(centerl):
                if i in list(everything):

                    np.delete(center,i)
                    del centerl[i]
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
        model.update_run_params(vm.attrs)

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
        model.update_run_params(vm.attrs)
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
