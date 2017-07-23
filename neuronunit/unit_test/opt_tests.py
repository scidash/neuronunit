"""Unit tests for the core of NeuronUnit"""
#Assumption that this file was executed after first executing the bash: ipcluster start -n 8 --profile=default &

# Run with any of:
# python core_tests.py
# python -m unittest core_tests.py
# coverage run --source . core_tests.py

%load_ext autoreload
#%reload 2
#!pip install -e '/home/jovyan/mnt/neuronunit'

import unittest
import sys
import os
os.system('sudo /opt/conda/bin/pip install -e . --process-dependency-links .')'
os.system('ipcluster start -n 8 --profile=default &')
#os.system('sleep 5')

import ipyparallel as ipp
#from ipyparallel.apps import iploggerapp
rc = ipp.Client(profile='default')
rc[:].use_cloudpickle()
dview = rc[:]

#THIS_DIR = os.path.dirname(os.getcwd())
#this_nu = os.path.join(THIS_DIR,'../../')
#sys.path.insert(0,this_nu)
#from neuronunit.optimization import nsga_p_for_nb



with dview.sync_imports():
    import os
out='/home/jovyan/mnt/neuronunit'
print(out)
def modify_path():
    sys.path.insert(0,'/home/jovyan/mnt/neuronunit')

dview.apply(modify_path)#,THIS_DIR)
modify_path()
print(sys.path)

with dview.sync_imports():
    from neuronunit.optimization import utilities, get_neab
    #from neuronunit.optimization import get_neab
    #from neuronunit import tests

utilities.VirtualModel
dview.push({'VirtualModel':VirtualModel})

with dview.sync_imports():
    import nsga_p_for_nb

class OptimizationTestCase(unittest.TestCase):
    """Rheobase finder """

    def test_path_mod_and_import(self):
        with dview.sync_imports():
            import os

        def modify_path():
            sys.path.insert(0,'/home/jovyan/mnt/neuronunit')

        dview.apply(modify_path)#,THIS_DIR)
        modify_path()
        with dview.sync_imports():
            from neuronunit.optimization import get_neab, utitilies
            #from neuronunit import tests


        utilities.VirtualModel
        dview.push({'VirtualModel':VirtualModel})

        with dview.sync_imports():
            import nsga_p_for_nb



    #scores = dview.map_sync(test.judge, models) # The map function, returning scores.
    #sciunit.ScoreMatrix([test],models,scores=scores)

    def test_rheobase(self):
        with open('../optimization/complete_dump.p','rb') as handle:
            variables = pickle.load(handle)
            vmpop,pop,pf,history=variables[0],variables[1],variables[2],variables[3]
            vmpop,pop,pf,history=variables[0],variables[1],variables[2],variables[3]

        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        import quantities as pq
        import numpy as np
        import get_neab

        import matplotlib.pyplot as plt

        matplotlib.use('Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work
        #matplotlib.style.use('ggplot')
        plt.style.use('ggplot')
        fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')


        for v in vmpop:
            new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
            model = ReducedModel(new_file_path,name=str(v.attrs),backend='NEURON')
            model.load_model()
            model.update_run_params(v.attrs)
            assert type(v.rheobase) is not type(None)
            params={}
            params['injected_square_current']={}

            params['injected_square_current']['duration'] = 1000 * pq.ms
            params['injected_square_current']['amplitude'] = v.rheobase * pq.pA
            params['injected_square_current']['delay'] = 100 * pq.ms
            model.inject_square_current(params['injected_square_current'])
            print(type(model.results['vm']),type(model.results['t']))
            # holding is default behaviorlabel=r'$\sin (x)$'
            #plt.plot(model.results['t'],model.results['vm'], label=str(v.rheobase)+str('pA'))
            axes.fill_between(model.results['t'],model.results['vm'], label=str(v.rheobase)+str('pA'),
            color='lightgray',
            linewidth=2,
            label=r'population standard deviation')

            plt.xlabel(str(' (ms)'))
            plt.ylabel(str(' (mV)'))
        #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        plt.title(str('Check of Rheobase Values'))#, fontsize=16, color='gray')
        plt.legend()
        plt.tight_layout()
        plt.savefig('rheobase_tests.png')

def test_tests(self):
    with open('../optimization/complete_dump.p','rb') as handle:
        variables = pickle.load(handle)
        vmpop,pop,pf,history=variables[0],variables[1],variables[2],variables[3]
        vmpop,pop,pf,history=variables[0],variables[1],variables[2],variables[3]

    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    import get_neab

    import matplotlib.pyplot as plt
    from itertools import repeat

    matplotlib.use('Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work
    matplotlib.style.use('ggplot')

    tests = get_neab.suite.tests
    for k,v in enumerate(tests):
        if k == 0:
            v.prediction = {}
            v.prediction['value'] = vms.rheobase * pq.pA
        if k == 1 or k == 2 or k == 3:
            v.params['injected_square_current']['duration'] = 100 * pq.ms
            v.params['injected_square_current']['amplitude'] = -10 *pq.pA
            v.params['injected_square_current']['delay'] = 30 * pq.ms
        if k == 5 or k == 6 or k == 7:
            v.params['injected_square_current']['duration'] = 1000 * pq.ms
            v.params['injected_square_current']['amplitude'] = vms.rheobase * pq.pA
            v.params['injected_square_current']['delay'] = 100 * pq.ms
            v.prediction = {}
            v.prediction['value'] = vms.rheobase * pq.pA
    local_test_methods = [ i.judge for i in tests ]


    def test_to_model(local_test_methods,model):
        import matplotlib.pyplot as plt
        matplotlib.use('Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work
        matplotlib.style.use('ggplot')
        local_test_methods(model)
        local_test_methods.tests.related_data['vm'].rescale('mV')
        plt.plot(local_test_methods.tests.related_data['t'], local_test_methods.tests.related_data['vm'])
        #plt.clf()

        return plt


    for v in vmpop:
        new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
        model = ReducedModel(new_file_path,name=str(v.attrs),backend='NEURON')
        model.load_model()
        model.update_run_params(v.attrs)
        plt = list(map(test_to_model,local_test_methods,repeat(model)))
        for p in plt:
            p.savefig('voltage_tests.png')
            #score = t.judge(model)

            assert type(v.rheobase) is not type(None)
        params={}
        params['injected_square_current']={}

        params['injected_square_current']['duration'] = 1000 * pq.ms
        params['injected_square_current']['amplitude'] = v.rheobase * pq.pA
        params['injected_square_current']['delay'] = 100 * pq.ms
        model.inject_square_current(params['injected_square_current'])
        print(type(model.results['vm']),type(model.results['t']))
        # holding is default behaviorlabel=r'$\sin (x)$'
        plt.plot(model.results['t'],model.results['vm'])#, label=str(v.rheobase)+str('pA'))
        #plt.xlabel(str(' (ms)'))
        #plt.ylabel(str(' (mV)'))
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #plt.title(str('Check of Rheobase Values'))#, fontsize=16, color='gray')
    #plt.legend()
    #plt.tight_layout()
    plt.savefig('rheobase_tests.png')


    def view_scores(self):

        import nsga_p_for_nb

        with open('../optimization/complete_dump.p','rb') as handle:
            variables = pickle.load(handle)
            vmpop,pop,pf,history=variables[0],variables[1],variables[2],variables[3]

        def test_pevaluation(self,vmpop,pop):
            fitnesses = list(dview.map(nsga_p_for_nb.evaluate, vmpop))
            for v in vmpop:
                assert type(v.score) is not type(None)

    def test_evaluation(self):
        '''
        A big test do it last.
        '''


        import nsga_p_for_nb

        with open('../optimization/complete_dump.p','rb') as handle:
            variables = pickle.load(handle)
            vmpop,pop,pf,history=variables[0],variables[1],variables[2],variables[3]

        def test_pevaluation(self,vmpop,pop):
            fitnesses = list(dview.map(nsga_p_for_nb.evaluate, vmpop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            return fitnesses

        def test_sevaluation(self,vmpop,pop):
            serial_fitnesses = []
            for vm in vmpop:
                serial_fitnesses.append(nsga_p_for_nb.evaluate(vm))
            return serial_fitnesses

        parallel_fitnesses = test_pevaluation()
        serial_fitnesses = test_pevaluation()
        assert serial_fitnesses == parallel_fitnesses




'''
from sciunit.utils import NotebookTools

# CONVERT_NOTEBOOKS environment variable controls whether notebooks are
# executed as notebooks or converted to regular python files first.

class DocumentationTestCase(NotebookTools,unittest.TestCase):
    """Testing documentation notebooks"""
    #@unittest.skip("Skipping chapter 1")
    def test_chapter1(self):
        self.do_notebook('explorter_voyager')
'''

if __name__ == '__main__':
    unittest.main()
