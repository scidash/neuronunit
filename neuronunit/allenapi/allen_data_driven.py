import pickle
import pandas as pd
import seaborn as sns

import bluepyopt as bpop
import bluepyopt.ephys as ephys
import pickle

import quantities as pq
PASSIVE_DURATION = 500.0*pq.ms
PASSIVE_DELAY = 200.0*pq.ms
import matplotlib.pyplot as plt
import numpy
import copy
import numpy as np
from sciunit.scores import ZScore
from collections.abc import Iterable
from bluepyopt.parameters import Parameter
from neuronunit.allenapi import make_allen_tests_from_id
from neuronunit.allenapi.make_allen_tests_from_id import *
from neuronunit.allenapi.make_allen_tests import AllenTest
from neuronunit.optimization.optimization_management import dtc_to_rheo
from neuronunit.optimization.optimization_management import inject_model30,check_bin_vm30,check_bin_vm15
from sciunit.scores import RelativeDifferenceScore
from sciunit import TestSuite
from sciunit.scores.collections import ScoreArray
from neuronunit.tests.base import AMPL, DELAY, DURATION
from neuronunit.optimization.optimization_management import test_all_objective_test
from neuronunit.optimization.optimization_management import check_binary_match, three_step_protocol,inject_and_plot_passive_model
from neuronunit.optimization.model_parameters import MODEL_PARAMS, BPO_PARAMS
from bluepyopt.allenapi.utils import dask_map_function

def opt_setup(specimen_id,cellmodel,target_num,provided_model = None,cached=None,fixed_current=False):
    if cached is not None:
        with open(str(specimen_id)+'later_allen_NU_tests.p','rb') as f:
            suite = pickle.load(f)

    else:

        sweep_numbers,data_set,sweeps = make_allen_tests_from_id.allen_id_to_sweeps(specimen_id)
        vmm,stimulus,sn,spike_times = make_allen_tests_from_id.get_model_parts_sweep_from_spk_cnt(target_num,data_set,sweep_numbers,specimen_id)
        suite,specimen_id = make_allen_tests_from_id.make_suite_known_sweep_from_static_models(vmm,stimulus,specimen_id)
        with open(str(specimen_id)+'later_allen_NU_tests.p','wb') as f:
            pickle.dump(suite,f)

    target = StaticModel(vm=suite.traces['vm15'])
    target.vm15 = suite.traces['vm15']
    nu_tests = suite.tests;
    check_bin_vm15(target,target)
    attrs = {k:np.mean(v) for k,v in MODEL_PARAMS[cellmodel].items()}
    dtc = DataTC(backend=cellmodel,attrs=attrs)
    for t in nu_tests:
        if t.name == 'Spikecount_1.5x':
            spk_count = float(t.observation['mean'])
            break
    observation_range={}
    observation_range['value'] = spk_count
    provided_model.backend = cellmodel
    provided_model.allen = None
    provided_model.allen = True
    model = provided_model
    if fixed_current:
        uc = {'amplitude':fixed_current,'duration':ALLEN_DURATION,'delay':ALLEN_DELAY}
        target_current = None
    else:
        scs = SpikeCountSearch(observation_range)
        target_current = scs.generate_prediction(provided_model)
        ALLEN_DELAY = 1000.0*qt.s
        ALLEN_DURATION = 2000.0*qt.s
        uc = {'amplitude':target_current['value'],'duration':ALLEN_DURATION,'delay':ALLEN_DELAY}
    model = dtc.dtc_to_model()
    model._backend.inject_square_current(**uc)
    return model, suite, nu_tests, target_current, spk_count

class NUFeatureAllenMultiSpike(object):
    def __init__(self,test,model,cnt,target,spike_obs,print_stuff=False):
        self.test = test
        self.model = model
        self.spike_obs = spike_obs
        self.cnt = cnt
        self.target = target
    def calculate_score(self,responses):

        if not 'features' in responses.keys():# or not 'model' in responses.keys():
            return 1000.0
        features = responses['features']
        if features is None:
            return 1000.0
        self.test.score_type = RelativeDifferenceScore
        feature_name = self.test.name
        if feature_name not in features.keys():
            return 1000.0

        if features[feature_name] is None:
            return 1000.0
        if type(features[self.test.name]) is type(Iterable):
            features[self.test.name] = np.mean(features[self.test.name])

        self.test.observation['mean'] = np.mean(self.test.observation['mean'])
        self.test.set_prediction(np.mean(features[self.test.name]))

        if 'Spikecount_1.5x'==feature_name:
            delta = np.abs(features[self.test.name]-np.mean(self.test.observation['mean']))
            if np.nan==delta or delta==np.inf:
                delta = 1000.0
            return delta
        else:
            if features[feature_name] is None:
                return 1000.0
            #self.test.score_type = RelativeDifferenceScore
            prediction = {'value':np.mean(features[self.test.name])}
            try:
                score_gene = self.test.feature_judge()
            except:
                score_gene = self.test.feature_judge()
                score_gene2 = self.test.judge(self.model,prediction=prediction)
                print(score_gene.raw,score_gene2.raw)
            #assert score_gene2.raw == score_gene.raw
            #score_gene = self.test.feature_judge()
            if score_gene is not None:
                if score_gene.raw is not None:
                    delta = np.abs(float(score_gene.raw))
                else:
                    delta = None
            else:
                delta = None
            if delta is None:
                delta = np.abs(features[self.test.name]-np.mean(self.test.observation['mean']))
            if np.nan==delta or delta==np.inf:
                delta = np.abs(features[self.test.name]-np.mean(self.test.observation['mean']))
            if np.nan==delta or delta==np.inf:
                delta = 1000.0
            return delta
def opt_setup_two(model, cellmodel, suite, nu_tests, target_current, spk_count,provided_model = None):
    objectives = []
    spike_obs = []
    for tt in nu_tests:
        if 'Spikecount_1.5x' == tt.name:
            spike_obs.append(tt.observation)
    spike_obs = sorted(spike_obs, key=lambda k: k['mean'],reverse=True)

    for cnt,tt in enumerate(nu_tests):
        feature_name = '%s' % (tt.name)
        if 'Spikecount_1.5x' == tt.name:
            ft = NUFeatureAllenMultiSpike(tt,model,cnt,target_current,spike_obs)
            objective = ephys.objectives.SingletonObjective(
                feature_name,
                ft)
            objectives.append(objective)


    score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives)

    if provided_model is None:
        print('depriciated')
        assert 1==2

        simple_cell = ephys.models.ReducedCellModel(
                name='simple_cell',
                params=BPO_PARAMS[cellmodel],backend=cellmodel)
    else:
        simple_cell = provided_model
    simple_cell.backend = cellmodel
    simple_cell.params = BPO_PARAMS[cellmodel]
    sweep_protocols = []
    for protocol_name, amplitude in [('step1', 0.05)]:

        protocol = ephys.protocols.SweepProtocol(protocol_name, [None], [None])
        sweep_protocols.append(protocol)
    onestep_protocol = ephys.protocols.SequenceProtocol('onestep', protocols=sweep_protocols)

    simple_cell.params_by_names(BPO_PARAMS[cellmodel].keys())
    simple_cell.params;

    cell_evaluator = ephys.evaluators.CellEvaluator(
            cell_model=simple_cell,
            param_names=BPO_PARAMS[cellmodel].keys(),
            fitness_protocols={onestep_protocol.name: onestep_protocol},
            fitness_calculator=score_calc,
            sim='euler')

    simple_cell.params_by_names(BPO_PARAMS[cellmodel].keys())
    simple_cell.params;
    simple_cell.seeded_current = target_current['value']
    simple_cell.spk_count = spk_count


    objectives2 = []
    for cnt,tt in enumerate(nu_tests):
        feature_name = '%s' % (tt.name)
        ft = NUFeatureAllenMultiSpike(tt,model,cnt,target_current,spike_obs,print_stuff=True)
        objective = ephys.objectives.SingletonObjective(
            feature_name,
            ft)
        objectives2.append(objective)
    score_calc2 = ephys.objectivescalculators.ObjectivesCalculator(objectives2)


    simple_cell.params_by_names(BPO_PARAMS[cellmodel].keys())
    simple_cell.params;


    cell_evaluator2 = ephys.evaluators.CellEvaluator(
            cell_model=simple_cell,
            param_names=list(BPO_PARAMS[cellmodel].keys()),
            fitness_protocols={onestep_protocol.name: onestep_protocol},
            fitness_calculator=score_calc2,
            sim='euler')
    return cell_evaluator2,simple_cell


def multi_layered(MU,NGEN,mapping_funct,cell_evaluator2):
    optimisation = bpop.optimisations.DEAPOptimisation(
            evaluator=cell_evaluator2,
            offspring_size = MU,
            map_function = map,
            selector_name='IBEA',mutpb=0.05,cxpb=0.6,current_fixed=from_outer)
    final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=NGEN)
    return final_pop, hall_of_fame, logs, hist


def opt_exec(MU,NGEN,mapping_funct,cell_evaluator2,mutpb=0.05,cxpb=0.6):
    optimisation = bpop.optimisations.DEAPOptimisation(
            evaluator=cell_evaluator2,
            offspring_size = MU,
            map_function = map,
            selector_name='IBEA',
            mutpb=mutpb,
            cxpb=cxpb)
    final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=NGEN)
    return final_pop, hall_of_fame, logs, hist

def opt_to_model(hall_of_fame,cell_evaluator2,suite, target_current, spk_count):
    best_ind = hall_of_fame[0]
    best_ind_dict = cell_evaluator2.param_dict(best_ind)
    model = cell_evaluator2.cell_model
    cell_evaluator2.param_dict(best_ind)

    model.attrs = {str(k):float(v) for k,v in cell_evaluator2.param_dict(best_ind).items()}
    model._backend.attrs = model.attrs

    opt = model.model_to_dtc()
    opt.attrs = {str(k):float(v) for k,v in cell_evaluator2.param_dict(best_ind).items()}
    model._backend.attrs = opt.attrs
    target = copy.copy(opt)
    target.vm15 = suite.traces['vm15']
    opt.seeded_current = target_current['value']
    opt.spk_count = spk_count

    target.seeded_current = target_current['value']
    target.spk_count = spk_count


    vm301,vm151,_,target = inject_model30(target,solve_for_current=target_current['value'])
    vm302,vm152,_,opt = inject_model30(opt,solve_for_current=target_current['value'])
    return opt,target
    '''
    #check_bin_vm30(opt,opt)
    check_bin_vm15(opt,opt)


    #check_bin_vm30(target,target)


    check_bin_vm15(target,target)



    gen_numbers = logs.select('gen')
    min_fitness = logs.select('min')
    max_fitness = logs.select('max')
    avg_fitness = logs.select('avg')
    plt.plot(gen_numbers, max_fitness, label='max fitness')
    plt.plot(gen_numbers, avg_fitness, label='avg fitness')
    plt.plot(gen_numbers, min_fitness, label='min fitness')

    plt.plot(gen_numbers, min_fitness, label='min fitness')
    plt.semilogy()
    plt.xlabel('generation #')
    plt.ylabel('score (# std)')
    plt.legend()
    plt.xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    plt.show()
    '''
'''
cp = {}
cp['final_pop'] = final_pop
cp['hall_of_fame'] = hall_of_fame


#with open('allen_opt.p','wb') as f:
#    pickle.dump(f,[final_pop, hall_of_fame, logs, hist])









optimisation = bpop.optimisations.DEAPOptimisation(
        evaluator=cell_evaluator2,
        offspring_size = MU,
        map_function = dask_map_function,
        selector_name='IBEA',mutpb=0.1,cxpb=0.35,seeded_pop=[cp['final_pop'],cp['hall_of_fame']])#,seeded_current=target_current)
final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=50)
'''
