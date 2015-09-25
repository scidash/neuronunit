import inspect
from quantities.quantity import Quantity
import sciunit
from sciunit import Test,Score,ObservationError
from neuronunit.capabilities import ProducesMembranePotential
from neuronunit.capabilities import ReceivesCurrent
from sciunit.comparators import zscore, dimensionless# Converters.
from sciunit.scores import ErrorScore,BooleanScore,ZScore # Scores.

try:
	import numpy
except:
	print("NumPy not loaded.")

import matplotlib.pyplot as plt
import matplotlib
from neuron import h
import collections
import efel
import os
import multiprocessing
import functools
import math
from scipy import stats

import json
from hippounit import plottools
import collections


try:
    import cPickle as pickle
except:
    import pickle
import gzip

import copy_reg
from types import MethodType

from quantities import mV, nA, ms, V, s


def zscore2(observation, prediction):
    """Computes a z-score from an observation and a prediction."""

    p_value_Ith = prediction['model_Ith']
    o_mean_Ith = observation['mean_Ith']
    o_std_Ith = observation['Ith_std']
    p_value_Veq = prediction['model_Veq']
    o_mean_Veq = observation['mean_Veq']
    o_std_Veq = observation['Veq_std']

    try:
        result_Ith = (p_value_Ith - o_mean_Ith)/o_std_Ith
        result_Ith = dimensionless(result_Ith)
        result_Veq = (p_value_Veq - o_mean_Veq)/o_std_Veq
        result_Veq = dimensionless(result_Veq)

    except (TypeError,AssertionError) as e:
        result_Ith = e
        result_Veq = e

    return [result_Ith, result_Veq]

score_l=[]

class ZScore2(Score):
    """
    A Z score. A float indicating standardized difference
    from a reference mean.
    """

    def __init__(self, score, related_data={}):

        for i in range(0, len(score)):
	        if not isinstance(score[i], Exception) and not isinstance(score[i], float):
	            raise InvalidScoreError("Score must be a float.")
	        else:
	            super(ZScore2,self).__init__(score[i], related_data=related_data)
	            score_l.append(score[i])

    def __str__(self):

		return 'Z_Ith = %.2f, Z_Veq = %.2f' % (score_l[0], score_l[1])


def ttest(exp_mean, model_mean, exp_sd, model_sd, exp_n, model_n):
    m1 = exp_mean
    m2 = model_mean
    v1 = exp_sd**2
    v2 = model_sd**2
    n1 = exp_n
    n2 = model_n

    vn1 = v1 / n1
    vn2 = v2 / n2

    df = ((vn1 + vn2)**2) / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

    denom = numpy.sqrt(vn1 + vn2)
    d = m1 - m2
    t = numpy.divide(d, denom)

    prob = stats.t.sf(numpy.abs(t), df) * 2  # use np.abs to get upper tail

    return prob

def ttest_calc(observation, prediction):

    exp_means=[observation['mean_threshold'], observation['mean_prox_threshold'], observation['mean_dist_threshold'], observation['mean_peak_deriv'], observation['mean_nonlin_at_th'], observation['mean_nonlin_suprath'],  observation['mean_amp_at_th'], observation['mean_time_to_peak'], observation['mean_async_nonlin']]
    exp_SDs=[observation['threshold_std'], observation['prox_threshold_std'], observation['dist_threshold_std'], observation['peak_deriv_std'], observation['nonlin_at_th_std'], observation['nonlin_suprath_std'],  observation['amp_at_th_std'], observation['time_to_peak_std'], observation['async_nonlin_std']]
    exp_Ns=[observation['exp_n'], observation['prox_n'], observation['dist_n'], observation['exp_n'], observation['exp_n'], observation['exp_n'], observation['exp_n'], observation['exp_n'], observation['async_n']]

    model_means = [prediction['model_mean_threshold'], prediction['model_mean_prox_threshold'], prediction['model_mean_dist_threshold'], prediction['model_mean_peak_deriv'], prediction['model_mean_nonlin_at_th'], prediction['model_mean_nonlin_suprath'],  prediction['model_mean_amp_at_th'], prediction['model_mean_time_to_peak'], prediction['model_mean_async_nonlin']]
    model_SDs = [prediction['model_threshold_std'], prediction['model_prox_threshold_std'], prediction['model_dist_threshold_std'], prediction['model_peak_deriv_std'], prediction['model_nonlin_at_th_std'], prediction['model_nonlin_suprath_std'], prediction['model_amp_at_th_std'], prediction['model_time_to_peak_std'], prediction['model_async_nonlin_std']]
    model_N= prediction['model_n']

    p_values=[]

    for i in range (0, len(exp_means)):

	    try:
	        ttest_result = ttest(exp_means[i], model_means[i], exp_SDs[i], model_SDs[i], exp_Ns[i], model_N)
	        ttest_result = dimensionless(ttest_result)
	        p_values.append(ttest_result)

	    except (TypeError,AssertionError) as e:
	        ttest_result = e

    return p_values


class P_Value(Score):
    """
    A p value from t-test.
    """

    def __init__(self, score, related_data={}):

        for i in range(0, len(score)):
	        if not isinstance(score[i], Exception) and not isinstance(score[i], float):
	            raise InvalidScoreError("Score must be a float.")
	        else:
	            super(P_Value,self).__init__(score[i], related_data=related_data)
	            score_l.append(score[i])

    def __str__(self):

		return '\n p_value_threshold = %.2f,\n p_value_prox_threshold  = %.2f,\n p_value_dist_threshold = %.2f,\n p_value_peak_dV/dt_at_threshold = %.2f,\n p_value_nonlin_at_th = %.2f,\n p_value_suprath_nonlin = %.2f,\n p_value_amplitude_at_th = %.2f,\n p_value_time_to_peak_at = %.2f,\n p_value_nonlin_at_th_asynch = %.2f\n' % (score_l[0], score_l[1],score_l[2], score_l[3],score_l[4], score_l[5],score_l[6], score_l[7],score_l[8])

def zscore3(observation, prediction):
    """Computes sum of z-scores from observation and prediction."""

    feature_error_means=numpy.array([])
    feature_error_stds=numpy.array([])
    features_names=(observation.keys())
    feature_results_dict={}

    for i in range (0, len(features_names)):
        if "Apic" not in features_names[i]:

            p_value = prediction[features_names[i]]['feature values']
            o_mean = float(observation[features_names[i]]['Mean'])
            o_std = float(observation[features_names[i]]['Std'])

            p_std = prediction[features_names[i]]['feature sd']


            try:
                feature_error = abs(p_value - o_mean)/o_std
                feature_error = dimensionless(feature_error)
                feature_error_mean=numpy.mean(feature_error)
                feature_error_sd=numpy.std(feature_error)

            except (TypeError,AssertionError) as e:
                feature_error = e
            feature_error_means=numpy.append(feature_error_means,feature_error_mean)
            feature_result={features_names[i]:{'mean feature error':feature_error_mean,
                                            'feature error sd':feature_error_sd}}

        feature_results_dict.update(feature_result)


    score_sum=numpy.sum(feature_error_means)

    return score_sum, feature_results_dict, features_names



class ZScore3(Score):
    """
    A Z score. A float indicating standardized difference
    from a reference mean.
    """

    def __init__(self, score, related_data={}):

	    if not isinstance(score, Exception) and not isinstance(score, float):
	        raise InvalidScoreError("Score must be a float.")
	    else:
	        super(ZScore3,self).__init__(score, related_data=related_data)

    def __str__(self):

		return 'Z_sum = %.2f' % self.score


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


class DepolarizationBlockTest(Test):
	"""Tests if the model enters depolarization block under current injection of increasing amplitudes."""

	def __init__(self,
			     observation = {'mean_Ith':None, 'Ith_std':None, 'mean_Veq': None, 'Veq_std': None}  ,
			     name="Depolarization block test" ,
				 force_run=False,
				show_plot=True):

		Test.__init__(self,observation,name)

		self.required_capabilities += (ProducesMembranePotential,
							  		   ReceivesCurrent,)

		self.force_run = force_run
		self.directory='./temp_data/'
		self.directory_figs='./figs/'

		description = "Tests if the model enters depolarization block under current injection of increasing amplitudes."

	score_type = ZScore2


	def cclamp(self, model, amp):

		path = self.directory + model.name + '/'

		if not os.path.exists(path):
			os.makedirs(path)

		file_name = path + 'cclamp_' + str(amp) + '.p'

		if self.force_run or (os.path.isfile(file_name) is False):

			trace = {}
			traces=[]

			print "- running amplitude:", amp

			# load cell
			model.initialise()
			model.set_cclamp(amp)
			t, v = model.run_cclamp()

			trace['T'] = t
			trace['V'] = v
			trace['stim_start'] = [500]
			trace['stim_end'] = [500+1000]
			traces.append(trace)

			traces_results = efel.getFeatureValues(traces,
										['Spikecount'])

			traces.append(traces_results)
			pickle.dump(traces, gzip.GzipFile(file_name, "wb"))

		else:
		    traces = pickle.load(gzip.GzipFile(file_name, "rb"))

		return traces

	def find_Ith_Veq(self, model, results, amps):


		path_figs = self.directory_figs + 'depol_block/' + model.name + '/'

		if not os.path.exists(path_figs):
			os.makedirs(path_figs)

		print "The figures are saved in the directory: ", path_figs

		spikecount_array=numpy.array([])

		for i, amp in enumerate(amps):

		    spikecount_array=numpy.append(spikecount_array, results[i][1][0]['Spikecount'])


		max=numpy.amax(spikecount_array)

		Ith_index = numpy.where(spikecount_array==max)[0]

		if Ith_index== spikecount_array.size-1:     # If the max num AP is the last element, it didn`t enter depol. block
		    Ith=float('NaN')                                 # If Ith == None, it means it didn`t enter depol block!!!
		    Veq=float('NaN')
		    Veq_index=Ith_index
		    plt.figure(1)
		    plt.plot(results[spikecount_array.size-1][0]['T'],results[spikecount_array.size-1][0]['V'])
		    plt.title("somatic response to the highest current intensity\n (The model did not enter depol. block.)")
		    print " the model did not enter depolarization block"
		    plt.savefig(path_figs + 'somatic_resp_at_depol_block' + '.pdf', dpi=300)

		else:
		    Ith=amps[Ith_index]
		    Ith=Ith[0]
		    Veq_index=Ith_index+1
		    plt.figure(1)
		    plt.plot(results[Veq_index][0]['T'],results[Veq_index][0]['V'])
		    plt.title("somatic response at Ith + 0.05 nA")
		    plt.xlabel("time (ms)")
		    plt.ylabel("Somatic voltage (mV)")
		    plt.savefig(path_figs + 'somatic_resp_at_depol_block' + '.pdf', dpi=300)

		    Veq_trace=results[Veq_index][0]['V']
		    time=numpy.array(results[Veq_index][0]['T'])
		    indices1 = numpy.where(1400<=time)[0]           # we need the last 100ms of the current pulse
		    indices2 = numpy.where(1500>=time)[0]
		    trace_end_index_beginning = numpy.amin(indices1)
		    trace_end_index_end=numpy.amax(indices2)
		    trace_end=Veq_trace[trace_end_index_beginning:trace_end_index_end]      # THIS CONTAINS the last 100ms of the current pulse

		    Veq=numpy.average(trace_end)

		print "Ith (the current intensity for which the model exhibited the maximum number of APs):", Ith
		print "Veq (the equilibrium value during the depolarization block):", Veq

		plt.figure(2)
		plt.plot(amps,spikecount_array,'.-')
		plt.xlabel("I (nA)")
		plt.ylabel("number of APs")
		plt.savefig(path_figs + 'number_of_APs' + '.pdf', dpi=600)

		plt.figure(3)
		plt.plot(results[Ith_index][0]['T'],results[Ith_index][0]['V'])
		plt.title("somatic response at Ith")
		plt.xlabel("time (ms)")
		plt.ylabel("Somatic voltage (mV)")
		plt.savefig(path_figs + 'somatic_resp_at_Ith' + '.pdf', dpi=600)

		x =numpy.array([1, 2])
		Ith_array = numpy.array([self.observation['mean_Ith'], Ith])
		labels = ['Target Ith with SD', model.name]
		e = numpy.array([self.observation['Ith_std'], 0.0])

		x2 =numpy.array([1])
		y2 = numpy.array([self.observation['mean_Ith']])
		e = numpy.array([self.observation['Ith_std']])

		plt.figure(4)
		plt.plot(x, Ith_array, 'o')
		plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
		plt.xticks(x, labels, rotation=10)
		plt.margins(0.2)
		plt.ylabel("Ith (nA)")
		plt.savefig(path_figs + 'Ith' + '.pdf', dpi=600)

		x =numpy.array([1, 2])
		Veq_array = numpy.array([self.observation['mean_Veq'], Veq])
		labels = ['Target Veq with SD',model.name]
		e = numpy.array([self.observation['Veq_std'], 0.0])

		x2 =numpy.array([1])
		y2 = numpy.array([self.observation['mean_Veq']])
		e = numpy.array([self.observation['Veq_std']])

		plt.figure(5)
		plt.plot(x, Veq_array, 'o')
		plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
		plt.xticks(x, labels, rotation=10)
		plt.margins(0.2)
		plt.ylabel("Veq (mV)")
		plt.savefig(path_figs + 'Veq' + '.pdf', dpi=600)

    	#errors

		if not math.isnan(Veq):
			Veq_error=abs(Veq*mV-self.observation['mean_Veq'])/self.observation['Veq_std']
			print "The error of Veq in units of the experimental SD: ", Veq_error


		if not math.isnan(Ith):
			Ith_error=abs(Ith*nA-self.observation['mean_Ith'])/self.observation['Ith_std']
			print "The error of Ith in units of the experimental SD: ", Ith_error


		num_AP_min=13
		num_AP_max=82
		labels2 = [model.name]

		plt.figure(7)
		plt.axhline(y=num_AP_min, label='Min. num.of AP\n observed\n experimentally')
		plt.axhline(y=num_AP_max, label='Max. num.of AP\n observed\n experimentally')
		plt.legend()

		plt.plot([1], spikecount_array[Ith_index], 'o')
		plt.title("For the models that doesn't enter depol block,\n the plot shows the num of AP-s for the highest current intensity")
		plt.xticks([1], labels2)
		plt.margins(0.2)
		plt.ylabel("Number of AP at Ith")

		return Ith, Veq

	def validate_observation(self, observation):

		try:
			assert type(observation['mean_Ith']) is Quantity
			assert type(observation['Ith_std']) is Quantity
			assert type(observation['mean_Veq']) is Quantity
			assert type(observation['Veq_std']) is Quantity
		except Exception as e:
			raise ObservationError(("Observation must be of the form "
									"{'mean':float*mV,'std':float*mV}"))

	def generate_prediction(self, model):
		"""Implementation of sciunit.Test.generate_prediction."""

		npool = 4
		pool = multiprocessing.Pool(npool)
		amps = numpy.arange(0,1.65,0.05)

		cclamp_ = functools.partial(self.cclamp, model)
		result = pool.map_async(cclamp_, amps)
		results = result.get()

		Ith, Veq = self.find_Ith_Veq(model, results, amps)
		prediction = {'model_Ith':float(Ith)*nA,'model_Veq': float(Veq)*mV}

		return prediction

	def compute_score(self, observation, prediction):
		"""Implementation of sciunit.Test.score_prediction."""
		score0 = zscore2(observation,prediction)
		score=ZScore2(score0)
		return score




class ObliqueIntegrationTest(Test):
	"""Tests the signal integration in oblique dendrites for increasing number of synchronous and asynchronous inputs"""

	def __init__(self,
			     observation = {'mean_threshold':None,'threshold_sem':None, 'threshold_std': None,
				                'mean_prox_threshold':None,'prox_threshold_sem':None, 'prox_threshold_std': None,
				                'mean_dist_threshold':None,'dist_threshold_sem':None, 'dist_threshold_std': None,
				                'mean_nonlin_at_th':None,'nonlin_at_th_sem':None, 'nonlin_at_th_std': None,
				                'mean_nonlin_suprath':None,'nonlin_suprath_sem':None, 'nonlin_suprath_std': None,
				                'mean_peak_deriv':None,'peak_deriv_sem':None, 'peak_deriv_std': None,
				                'mean_amp_at_th':None,'amp_at_th_sem':None, 'amp_at_th_std': None,
				                'mean_time_to_peak':None,'time_to_peak_sem':None, 'time_to_peak_std': None,
				                'mean_async_nonlin':None,'async_nonlin_sem':None, 'async_nonlin_std': None}  ,
			     name="Oblique integration test" ,
				 force_run_synapse=False,
				 force_run_bin_search=False,
				show_plot=True):

		Test.__init__(self, observation, name)

		self.required_capabilities += (ProducesMembranePotential,
							  		   ReceivesCurrent,)

		self.force_run_synapse = force_run_synapse
		self.force_run_bin_search = force_run_bin_search
		self.directory='./temp_data/'
		self.directory_figs='./figs/'

		description = "Tests the signal integration in oblique dendrites for increasing number of synchronous and asynchronous inputs"

	score_type = P_Value


	def analyse_syn_traces(self, t, v, v_dend, threshold):

	    trace = {}
	    trace['T'] = t
	    trace['V'] = v
	    trace['stim_start'] = [150]
	    trace['stim_end'] = [150+500]  # should be changed
	    traces = [trace]

	    trace_dend = {}
	    trace_dend['T'] = t
	    trace_dend['V'] = v_dend
	    trace_dend['stim_start'] = [150]
	    trace_dend['stim_end'] = [150+500]
	    traces_dend = [trace_dend]

	    efel.setThreshold(threshold)
	    traces_results_dend = efel.getFeatureValues(traces_dend,['Spikecount'])
	    traces_results = efel.getFeatureValues(traces,['Spikecount'])
	    spikecount_dend=traces_results_dend[0]['Spikecount']
	    spikecount=traces_results[0]['Spikecount']

	    result = [traces, traces_dend, spikecount, spikecount_dend]

	    return result


	def run_synapse(self,model, dend_loc_num_weight, interval):


	    ndend, xloc, num, weight = dend_loc_num_weight

	    path = self.directory + model.name + '/synapse/'

	    if not os.path.exists(path):
	        os.makedirs(path)

	    if interval>0.1:
	        file_name = path + 'synapse_' + str(num)+ '_' + str(ndend)+ '_' + str(xloc) + '.p'
	    else:
	        file_name = path + 'synapse_async_' + str(num)+ '_' + str(ndend)+ '_' + str(xloc) + '.p'

	    if self.force_run_synapse or (os.path.isfile(file_name) is False):

	        print "- number of inputs:", num, "dendrite:", ndend, "xloc", xloc

	        model.initialise()
	        model.set_ampa_nmda([ndend,xloc])
	        model.set_netstim_netcon(interval)
	        model.set_num_weight(num, weight)

	        t, v, v_dend = model.run_syn()

	        result = self.analyse_syn_traces(t, v, v_dend, model.threshold)

	        pickle.dump(result, gzip.GzipFile(file_name, "wb"))

	    else:
	        result = pickle.load(gzip.GzipFile(file_name, "rb"))

	    return result

	def binsearch(self, model, dend_loc0):

	    path_bin_search = self.directory + model.name + '/bin_search/'

	    if not os.path.exists(path_bin_search):
	        os.makedirs(path_bin_search)

	    interval=0.1

	    file_name = path_bin_search + 'weight_' +str(dend_loc0[0])+ '_' + str(dend_loc0[1]) + '.p'


	    if self.force_run_bin_search or (os.path.isfile(file_name) is False):

	        c_minmax = model.c_minmax
	        c_step_start = model.c_step_start
	        c_step_stop= model.c_step_stop
	        #c_stim=numpy.arange()

	        model.initialise()
	        model.set_ampa_nmda(dend_loc0)
	        model.set_netstim_netcon(interval)

	        found = False
	        while c_step_start >= c_step_stop and not found:

	            c_stim = numpy.arange(c_minmax[0], c_minmax[1], c_step_start)

	            #print c_stim
	            first = 0
	            last = numpy.size(c_stim, axis=0)-1
	            num = [3,4]


	            while first<=last and not found:

	                midpoint = (first + last)//2
	                result=[]

	                for n in num:
	                    model.set_num_weight(n, c_stim[midpoint])

	                    t, v, v_dend = model.run_syn()

	                    result.append(self.analyse_syn_traces(t, v, v_dend, model.threshold))
	                    #print result

	                if result[0][3]==0 and result[1][3]>=1:
	                    found = True
	                else:
	                    if result[0][3]>=1 and result[1][3]>=1:
	                        last = midpoint-1

	                    elif result[0][3]==0 and result[1][3]==0:
	                        first = midpoint+1

	            c_step_start=c_step_start/2

	            if found:
	                if result[1][2]>=1 :
	                    found = None
	                    break

	        binsearch_result=[found, c_stim[midpoint]]

	        pickle.dump(binsearch_result, gzip.GzipFile(file_name, "wb"))

	    else:
	        binsearch_result = pickle.load(gzip.GzipFile(file_name, "rb"))


	    return binsearch_result

	def calcs_plots(self, model, results, dend_loc000, dend_loc_num_weight):

	    experimental_mean_threshold=self.observation['mean_threshold']
	    threshold_SEM=self.observation['threshold_sem']
	    threshold_SD=self.observation['threshold_std']

	    threshold_prox=self.observation['mean_prox_threshold']
	    threshold_prox_SEM=self.observation['prox_threshold_sem']
	    threshold_prox_SD=self.observation['prox_threshold_std']

	    threshold_dist=self.observation['mean_dist_threshold']
	    threshold_dist_SEM=self.observation['dist_threshold_sem']
	    threshold_dist_SD=self.observation['dist_threshold_std']

	    exp_mean_nonlin=self.observation['mean_nonlin_at_th']
	    nonlin_SEM=self.observation['nonlin_at_th_sem']
	    nonlin_SD=self.observation['nonlin_at_th_std']

	    suprath_exp_mean_nonlin=self.observation['mean_nonlin_suprath']
	    suprath_nonlin_SEM=self.observation['nonlin_suprath_sem']
	    suprath_nonlin_SD=self.observation['nonlin_suprath_std']

	    exp_mean_peak_deriv=self.observation['mean_peak_deriv']
	    deriv_SEM=self.observation['peak_deriv_sem']
	    deriv_SD=self.observation['peak_deriv_std']

	    exp_mean_amp=self.observation['mean_amp_at_th']
	    amp_SEM= self.observation['amp_at_th_sem']
	    amp_SD=self.observation['amp_at_th_std']

	    exp_mean_time_to_peak=self.observation['mean_time_to_peak']
	    exp_mean_time_to_peak_SEM=self.observation['time_to_peak_sem']
	    exp_mean_time_to_peak_SD=self.observation['time_to_peak_std']

	    path_figs = self.directory_figs + 'oblique/' + model.name + '/'

	    if not os.path.exists(path_figs):
	        os.makedirs(path_figs)

	    print "The figures are saved in the directory: ", path_figs

	    stop=len(dend_loc_num_weight)+1
	    sep=numpy.arange(0,stop,11)
	    sep_results=[]

	    max_num_syn=10
	    num = numpy.arange(0,max_num_syn+1)

	    for i in range (0,len(dend_loc000)):
	        sep_results.append(results[sep[i]:sep[i+1]])             # a list that contains the results of the 10 locations seperately (in lists)

	    # sep_results[0]-- the first location
	    # sep_results[0][5] -- the first location at 5 input
	    # sep_results[0][1][0] -- the first location at 1 input, SOMA
	    # sep_results[0][1][1] -- the first location at 1 input, dendrite
	    # sep_results[0][1][1][0] -- just needed

	    fig0, axes0 = plt.subplots(nrows=2, ncols=1)
	    fig0.tight_layout()
	    fig0.suptitle('Synchronous inputs (red: dendritic trace, black: somatic trace)')
	    for i in range (0,len(dend_loc000)):
	        plt.subplot(5,2,i+1)
	        plt.subplots_adjust(hspace = 0.5)
	        for j, number in enumerate(num):
	            plt.plot(sep_results[i][j][0][0]['T'],sep_results[i][j][0][0]['V'], 'k')       # somatic traces
	            plt.plot(sep_results[i][j][1][0]['T'],sep_results[i][j][1][0]['V'], 'r')        # dendritic traces
	        plt.title('Input in dendrite '+str(dend_loc000[i][0])+ ' at location: ' +str(dend_loc000[i][1]))

	        plt.xlabel("time (ms)")
	        plt.ylabel("Voltage (mV)")
	    fig0 = plt.gcf()
	    fig0.set_size_inches(12, 18)
	    plt.savefig(path_figs + 'traces_sync' + '.pdf', dpi=600,)

	    soma_depol=numpy.array([])
	    soma_depols=[]
	    sep_soma_depols=[]
	    dV_dt=[]
	    sep_dV_dt=[]
	    soma_max_depols=numpy.array([])
	    soma_expected=numpy.array([])
	    sep_soma_max_depols=[]
	    sep_soma_expected=[]
	    max_dV_dt=numpy.array([])
	    sep_max_dV_dt=[]
	    max_dV_dt_index=numpy.array([],dtype=numpy.int64)
	    sep_threshold=numpy.array([])
	    prox_thresholds=numpy.array([])
	    dist_thresholds=numpy.array([])
	    peak_dV_dt_at_threshold=numpy.array([])
	    nonlin=numpy.array([])
	    suprath_nonlin=numpy.array([])
	    amp_at_threshold=[]
	    sep_time_to_peak=[]
	    time_to_peak_at_threshold=numpy.array([])
	    time_to_peak=numpy.array([])
	    threshold_index=numpy.array([])

	    for i in range (0, len(sep_results)):
	        for j in range (0,max_num_syn+1):

	    # calculating somatic depolarization and first derivative
	            soma_depol=sep_results[i][j][0][0]['V'] - sep_results[i][0][0][0]['V']
	            soma_depols.append(soma_depol)

	            soma_max_depols=numpy.append(soma_max_depols,numpy.amax(soma_depol))

	            dt=numpy.diff(sep_results[i][j][0][0]['T'] )
	            dV=numpy.diff(sep_results[i][j][0][0]['V'] )
	            deriv=dV/dt
	            dV_dt.append(deriv)

	            max_dV_dt=numpy.append(max_dV_dt, numpy.amax(dV_dt))

	            diff_max_dV_dt=numpy.diff(max_dV_dt)

	            if j==0:
	                soma_expected=numpy.append(soma_expected,0)
	            else:
	                soma_expected=numpy.append(soma_expected,soma_max_depols[1]*j)

	            if j!=0:
	                peak=numpy.amax(soma_depol)

	                peak_index=numpy.where(soma_depol==peak)[0]
	                peak_time=sep_results[i][j][0][0]['T'][peak_index]
	                t_to_peak=peak_time-150
	                time_to_peak = numpy.append(time_to_peak, t_to_peak)
	            else:
	                time_to_peak = numpy.append(time_to_peak, 0)

	            #print time_to_peak

	        threshold_index0=numpy.where(diff_max_dV_dt==numpy.amax(diff_max_dV_dt[1:]))[0]
	        threshold_index0=numpy.add(threshold_index0,1)

	        if sep_results[i][threshold_index0][3] > 1 and sep_results[i][threshold_index0-1][3]==1:    #double spikes can cause bigger jump in dV?dt than the first single spike, to find the threshol, we want to eliminate this, but we also need the previous input level to generate spike
	            threshold_index=numpy.where(diff_max_dV_dt==numpy.amax(diff_max_dV_dt[1:threshold_index0-1]))
	            threshold_index=numpy.add(threshold_index,1)
	        else:
	            threshold_index=threshold_index0

	        threshold=soma_expected[threshold_index]


	        sep_soma_depols.append(soma_depols)
	        sep_dV_dt.append(dV_dt)
	        sep_soma_max_depols.append(soma_max_depols)
	        sep_soma_expected.append(soma_expected)
	        sep_max_dV_dt.append(max_dV_dt)
	        sep_threshold=numpy.append(sep_threshold, threshold)
	        peak_dV_dt_at_threshold=numpy.append(peak_dV_dt_at_threshold,max_dV_dt[threshold_index])
	        nonlin=numpy.append(nonlin, soma_max_depols[threshold_index]/ soma_expected[threshold_index]*100)  #degree of nonlinearity
	        suprath_nonlin=numpy.append(suprath_nonlin, soma_max_depols[threshold_index+1]/ soma_expected[threshold_index+1]*100)  #degree of nonlinearity
	        amp_at_threshold=numpy.append(amp_at_threshold, soma_max_depols[threshold_index])
	        sep_time_to_peak.append(time_to_peak)
	        time_to_peak_at_threshold=numpy.append(time_to_peak_at_threshold, time_to_peak[threshold_index])

	        soma_depols=[]
	        dV_dt=[]
	        soma_max_depols=numpy.array([])
	        soma_expected=numpy.array([])
	        max_dV_dt=numpy.array([])
	        threshold_index=numpy.array([])
	        threshold_index0=numpy.array([])
	        time_to_peak=numpy.array([])


	    prox_thresholds=sep_threshold[0::2]
	    dist_thresholds=sep_threshold[1::2]

	    threshold_errors = numpy.array([abs(experimental_mean_threshold - threshold_errors*mV)/threshold_SD  for threshold_errors in sep_threshold])     # does the same calculation on every element of a list  #x = [1,3,4,5,6,7,8] t = [ t**2 for t in x ]
	    prox_threshold_errors=numpy.array([abs(threshold_prox - prox_threshold_errors*mV)/threshold_prox_SD  for prox_threshold_errors in prox_thresholds])        # and I could easily make it a numpy array : t = numpy.array([ t**2 for t in x ])
	    dist_threshold_errors=numpy.array([abs(threshold_dist - dist_threshold_errors*mV)/threshold_dist_SD  for dist_threshold_errors in dist_thresholds])
	    peak_deriv_errors=numpy.array([abs(exp_mean_peak_deriv - peak_deriv_errors*mV /ms )/deriv_SD  for peak_deriv_errors in peak_dV_dt_at_threshold])
	    nonlin_errors=numpy.array([abs(exp_mean_nonlin- nonlin_errors)/nonlin_SD  for nonlin_errors in nonlin])
	    suprath_nonlin_errors=numpy.array([abs(suprath_exp_mean_nonlin- suprath_nonlin_errors)/suprath_nonlin_SD  for suprath_nonlin_errors in suprath_nonlin])
	    amplitude_errors=numpy.array([abs(exp_mean_amp- amplitude_errors*mV)/amp_SD  for amplitude_errors in amp_at_threshold])
	    time_to_peak_errors=numpy.array([abs(exp_mean_time_to_peak- time_to_peak_errors*ms)/exp_mean_time_to_peak_SD  for time_to_peak_errors in time_to_peak_at_threshold])


	    # means and SDs
	    mean_threshold_errors=numpy.mean(threshold_errors)
	    mean_prox_threshold_errors=numpy.mean(prox_threshold_errors)
	    mean_dist_threshold_errors=numpy.mean(dist_threshold_errors)
	    mean_peak_deriv_errors=numpy.mean(peak_deriv_errors)
	    mean_nonlin_errors=numpy.mean(nonlin_errors)
	    suprath_mean_nonlin_errors=numpy.mean(suprath_nonlin_errors)
	    mean_amplitude_errors=numpy.mean(amplitude_errors)
	    mean_time_to_peak_errors=numpy.mean(time_to_peak_errors)

	    sd_threshold_errors=numpy.std(threshold_errors)
	    sd_prox_threshold_errors=numpy.std(prox_threshold_errors)
	    sd_dist_threshold_errors=numpy.std(dist_threshold_errors)
	    sd_peak_deriv_errors=numpy.std(peak_deriv_errors)
	    sd_nonlin_errors=numpy.std(nonlin_errors)
	    suprath_sd_nonlin_errors=numpy.std(suprath_nonlin_errors)
	    sd_amplitude_errors=numpy.std(amplitude_errors)
	    sd_time_to_peak_errors=numpy.std(time_to_peak_errors)


	    mean_sep_threshold=float(numpy.mean(sep_threshold)) *mV
	    mean_prox_thresholds=float(numpy.mean(prox_thresholds)) *mV
	    mean_dist_thresholds=numpy.mean(dist_thresholds) *mV
	    mean_peak_dV_dt_at_threshold=numpy.mean(peak_dV_dt_at_threshold) *V /s
	    mean_nonlin=numpy.mean(nonlin)
	    suprath_mean_nonlin=numpy.mean(suprath_nonlin)
	    mean_amp_at_threshold=numpy.mean(amp_at_threshold) *mV
	    mean_time_to_peak_at_threshold=numpy.mean(time_to_peak_at_threshold) *ms

	    sd_sep_threshold=float(numpy.std(sep_threshold)) *mV
	    sd_prox_thresholds=float(numpy.std(prox_thresholds)) *mV
	    sd_dist_thresholds=numpy.std(dist_thresholds) *mV
	    sd_peak_dV_dt_at_threshold=numpy.std(peak_dV_dt_at_threshold) *V /s
	    sd_nonlin=numpy.std(nonlin)
	    suprath_sd_nonlin=numpy.std(suprath_nonlin)
	    sd_amp_at_threshold=numpy.std(amp_at_threshold) *mV
	    sd_time_to_peak_at_threshold=numpy.std(time_to_peak_at_threshold) *ms


	    depol_input=numpy.array([])
	    mean_depol_input=[]
	    SD_depol_input=[]
	    SEM_depol_input=[]

	    expected_depol_input=numpy.array([])
	    expected_mean_depol_input=[]

	    prox_depol_input=numpy.array([])
	    prox_mean_depol_input=[]
	    prox_SD_depol_input=[]
	    prox_SEM_depol_input=[]

	    prox_expected_depol_input=numpy.array([])
	    prox_expected_mean_depol_input=[]

	    dist_depol_input=numpy.array([])
	    dist_mean_depol_input=[]
	    dist_SD_depol_input=[]
	    dist_SEM_depol_input=[]

	    dist_expected_depol_input=numpy.array([])
	    dist_expected_mean_depol_input=[]

	    peak_deriv_input=numpy.array([])
	    mean_peak_deriv_input=[]
	    SD_peak_deriv_input=[]
	    SEM_peak_deriv_input=[]
	    n=len(sep_soma_max_depols)

	    prox_sep_soma_max_depols=sep_soma_max_depols[0::2]
	    dist_sep_soma_max_depols=sep_soma_max_depols[1::2]
	    prox_n=len(prox_sep_soma_max_depols)
	    dist_n=len(dist_sep_soma_max_depols)

	    prox_sep_soma_expected=sep_soma_expected[0::2]
	    dist_sep_soma_expected=sep_soma_expected[1::2]


	    for j in range (0, max_num_syn+1):
	        for i in range (0, len(sep_soma_max_depols)):
	            depol_input=numpy.append(depol_input,sep_soma_max_depols[i][j])
	            expected_depol_input=numpy.append(expected_depol_input,sep_soma_expected[i][j])
	            peak_deriv_input=numpy.append(peak_deriv_input,sep_max_dV_dt[i][j])
	        mean_depol_input.append(numpy.mean(depol_input))
	        expected_mean_depol_input.append(numpy.mean(expected_depol_input))
	        mean_peak_deriv_input.append(numpy.mean(peak_deriv_input))
	        SD_depol_input.append(numpy.std(depol_input))
	        SEM_depol_input.append(numpy.std(depol_input)/math.sqrt(n))
	        SD_peak_deriv_input.append(numpy.std(peak_deriv_input))
	        SEM_peak_deriv_input.append(numpy.std(peak_deriv_input)/math.sqrt(n))
	        depol_input=numpy.array([])
	        expected_depol_input=numpy.array([])
	        peak_deriv_input=numpy.array([])

	    for j in range (0, max_num_syn+1):
	        for i in range (0, len(prox_sep_soma_max_depols)):
	            prox_depol_input=numpy.append(prox_depol_input,prox_sep_soma_max_depols[i][j])
	            prox_expected_depol_input=numpy.append(prox_expected_depol_input,prox_sep_soma_expected[i][j])
	        prox_mean_depol_input.append(numpy.mean(prox_depol_input))
	        prox_expected_mean_depol_input.append(numpy.mean(prox_expected_depol_input))
	        prox_SD_depol_input.append(numpy.std(prox_depol_input))
	        prox_SEM_depol_input.append(numpy.std(prox_depol_input)/math.sqrt(prox_n))
	        prox_depol_input=numpy.array([])
	        prox_expected_depol_input=numpy.array([])

	    for j in range (0, max_num_syn+1):
	        for i in range (0, len(dist_sep_soma_max_depols)):
	            dist_depol_input=numpy.append(dist_depol_input,dist_sep_soma_max_depols[i][j])
	            dist_expected_depol_input=numpy.append(dist_expected_depol_input,dist_sep_soma_expected[i][j])
	        dist_mean_depol_input.append(numpy.mean(dist_depol_input))
	        dist_expected_mean_depol_input.append(numpy.mean(dist_expected_depol_input))
	        dist_SD_depol_input.append(numpy.std(dist_depol_input))
	        dist_SEM_depol_input.append(numpy.std(dist_depol_input)/math.sqrt(dist_n))
	        dist_depol_input=numpy.array([])
	        dist_expected_depol_input=numpy.array([])

	    plt.figure(2)


	    plt.title('Synchronous inputs')

	    # Expected EPSP - Measured EPSP plot
	    colormap = plt.cm.spectral      #http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
	    plt.gca().set_color_cycle([colormap(j) for j in numpy.linspace(0, 0.9, len(sep_results))])
	    for i in range (0, len(sep_results)):

	        plt.plot(sep_soma_expected[i],sep_soma_max_depols[i], '-o', label='dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	        plt.plot(sep_soma_expected[i],sep_soma_expected[i], 'k--')         # this gives the linear line
	        plt.xlabel("expected EPSP (mV)")
	        plt.ylabel("measured EPSP (mV)")
	        plt.legend(loc=2, prop={'size':10})
	    fig = plt.gcf()
	    fig.set_size_inches(12, 12)
	    plt.savefig(path_figs + 'input_output_curve_sync' + '.pdf', dpi=600,)

	    plt.figure(3)
	    plt.suptitle('Synchronous inputs')

	    plt.subplot(3,1,1)
	    plt.errorbar(expected_mean_depol_input, mean_depol_input, yerr=SD_depol_input, linestyle='-', marker='o', color='red', label='SD')
	    plt.errorbar(expected_mean_depol_input, mean_depol_input, yerr=SEM_depol_input, linestyle='-', marker='o', color='blue', label='SEM')
	    plt.plot(expected_mean_depol_input,expected_mean_depol_input, 'k--')         # this gives the linear line
	    plt.margins(0.1)
	    plt.legend(loc=2)
	    plt.title("Summary plot of mean input-output curve for every locations")
	    plt.xlabel("expected EPSP (mV)")
	    plt.ylabel("measured EPSP (mV)")

	    plt.subplot(3,1,2)
	    plt.errorbar(prox_expected_mean_depol_input, prox_mean_depol_input, yerr=prox_SD_depol_input, linestyle='-', marker='o', color='red', label='SD')
	    plt.errorbar(prox_expected_mean_depol_input, prox_mean_depol_input, yerr=prox_SEM_depol_input, linestyle='-', marker='o', color='blue', label='SEM')
	    plt.plot(prox_expected_mean_depol_input,prox_expected_mean_depol_input, 'k--')         # this gives the linear line
	    plt.margins(0.1)
	    plt.legend(loc=2)
	    plt.title("Summary plot of mean input-output curve for proximal locations")
	    plt.xlabel("expected EPSP (mV)")
	    plt.ylabel("measured EPSP (mV)")

	    plt.subplot(3,1,3)
	    plt.errorbar(dist_expected_mean_depol_input, dist_mean_depol_input, yerr=dist_SD_depol_input, linestyle='-', marker='o', color='red', label='SD')
	    plt.errorbar(dist_expected_mean_depol_input, dist_mean_depol_input, yerr=dist_SEM_depol_input, linestyle='-', marker='o', color='blue', label='SEM')
	    plt.plot(dist_expected_mean_depol_input,dist_expected_mean_depol_input, 'k--')         # this gives the linear line

	    plt.margins(0.1)
	    plt.legend(loc=2)
	    plt.title("Summary plot of mean input-output curve for distal locations")
	    plt.xlabel("expected EPSP (mV)")
	    plt.ylabel("measured EPSP (mV)")

	    fig = plt.gcf()
	    fig.set_size_inches(12, 15)
	    plt.savefig(path_figs + 'summary_input_output_curve_sync' + '.pdf', dpi=600,)

	    plt.figure(4)

	    plt.subplot(2,1,1)
	    plt.title('Synchronous inputs')
	#Derivative plot
	    colormap = plt.cm.spectral
	    plt.gca().set_color_cycle([colormap(j) for j in numpy.linspace(0, 0.9, len(sep_results))])
	    for i in range (0, len(sep_results)):

	        plt.plot(num,sep_max_dV_dt[i], '-o', label='dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	        plt.xlabel("# of inputs")
	        plt.ylabel("dV/dt (V/s)")
	        plt.legend(loc=2, prop={'size':10})

	    plt.subplot(2,1,2)

	    plt.errorbar(num, mean_peak_deriv_input, yerr=SD_peak_deriv_input, linestyle='-', marker='o', color='red', label='SD')
	    plt.errorbar(num, mean_peak_deriv_input, yerr=SEM_peak_deriv_input, linestyle='-', marker='o', color='blue', label='SEM')
	    plt.margins(0.1)
	    plt.legend(loc=2)
	    plt.title("Summary plot of mean peak dV/dt amplitude")
	    plt.xlabel("# of inputs")
	    plt.ylabel("dV/dt (V/s)")

	    fig = plt.gcf()
	    fig.set_size_inches(12, 12)
	    plt.savefig(path_figs + 'peak_derivative_plots_sync' + '.pdf', dpi=600,)

	    #VALUES PLOT
	    fig, axes = plt.subplots(nrows=4, ncols=2)
	    fig.tight_layout()
	    fig.suptitle('Synchronous inputs')

	    plt.subplot(4, 2, 1)
	# plot of thresholds
	    x =numpy.array([])
	    labels = ['exp mean with SD']
	    e = numpy.array([threshold_SD])
	    x2 =numpy.array([1])
	    y2 = numpy.array([experimental_mean_threshold])
	    for i in range (0, len(sep_results)+1):
	        x=numpy.append(x, i+1)
	    for i in range (0, len(sep_results)):
	        labels.append('dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	        plt.plot(x[i+1], sep_threshold[i], 'o')

	    plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	    plt.xticks(x, labels, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("Threshold (mV)")

	    plt.subplot(4, 2, 2)

	# plot of proximal thresholds
	    x_prox =numpy.array([])
	    labels_prox = ['exp mean with SD']
	    e = numpy.array([threshold_prox_SD])
	    x2 =numpy.array([1])
	    y2 = numpy.array([threshold_prox])
	    for i in range (0, len(dend_loc000),2):
	        labels_prox.append('dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	    for i in range (0, len(prox_thresholds)+1):
	        x_prox=numpy.append(x_prox, i+1)
	    for i in range (0, len(prox_thresholds)):
	        plt.plot(x_prox[i+1], prox_thresholds[i], 'o')

	    plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	    plt.xticks(x_prox, labels_prox, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("Proximal threshold (mV)")

	    plt.subplot(4, 2, 3)

	# plot of distal thresholds
	    x_dist =numpy.array([])
	    labels_dist = ['exp mean with SD']
	    e = numpy.array([threshold_dist_SD])
	    x2 =numpy.array([1])
	    y2 = numpy.array([threshold_dist])
	    for i in range (1, len(dend_loc000),2):
	        labels_dist.append('dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	    for i in range (0, len(dist_thresholds)+1):
	        x_dist=numpy.append(x_dist, i+1)
	    for i in range (0, len(dist_thresholds)):
	        plt.plot(x_dist[i+1], dist_thresholds[i], 'o')

	    plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	    plt.xticks(x_dist, labels_dist, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("Distal threshold (mV)")

	    plt.subplot(4, 2, 4)

	# plot of peak derivateives at threshold
	    e = numpy.array([deriv_SD])
	    x2 =numpy.array([1])
	    y2 = numpy.array([exp_mean_peak_deriv])
	    for i in range (0, len(sep_results)):
	        plt.plot(x[i+1], peak_dV_dt_at_threshold[i], 'o')

	    plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	    plt.xticks(x, labels, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("peak derivative at threshold (V/s)")

	    plt.subplot(4, 2, 5)

	# plot of degree of nonlinearity at threshold

	    e = numpy.array([nonlin_SD])
	    x2 =numpy.array([1])
	    y2 = numpy.array([exp_mean_nonlin])
	    for i in range (0, len(sep_results)):
	        plt.plot(x[i+1], nonlin[i], 'o')

	    plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	    plt.xticks(x, labels, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("degree of nonlinearity (%)")

	    plt.subplot(4, 2, 6)

	# plot of suprathreshold degree of nonlinearity

	    e = numpy.array([suprath_nonlin_SD])
	    x2 =numpy.array([1])
	    y2 = numpy.array([suprath_exp_mean_nonlin])
	    for i in range (0, len(sep_results)):
	        plt.plot(x[i+1], suprath_nonlin[i], 'o')

	    plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	    plt.xticks(x, labels, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("supth. degree of nonlinearity (%)")

	    plt.subplot(4, 2, 7)

	# plot of amplitude at threshold
	    e = numpy.array([amp_SD])
	    x2 =numpy.array([1])
	    y2 = numpy.array([exp_mean_amp])
	    for i in range (0, len(sep_results)):
	        plt.plot(x[i+1], amp_at_threshold[i], 'o')

	    plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	    plt.xticks(x, labels, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("Amplitude at threshold (mV)")


	    plt.subplot(4, 2, 8)

	# plot of time to peak at threshold
	    e = numpy.array([exp_mean_time_to_peak_SD])
	    x2 =numpy.array([1])
	    y2 = numpy.array([exp_mean_time_to_peak])
	    for i in range (0, len(sep_results)):
	        plt.plot(x[i+1], time_to_peak_at_threshold[i], 'o')

	    plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	    plt.xticks(x, labels, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("time to peak at threshold (ms)")

	    fig = plt.gcf()
	    fig.set_size_inches(14, 14)
	    plt.savefig(path_figs + 'values_sync' + '.pdf', dpi=600,)



	    # ERROR PLOTS


	    fig2, axes2 = plt.subplots(nrows=3, ncols=2)
	    fig2.tight_layout()
	    fig2.suptitle(' Errors in units of the experimental SD of the feature (synchronous inputs)')
	    plt.subplot(4, 2, 1)

	#threshold error plot
	    x_error =numpy.array([])
	    labels_error = []
	    for i in range (0, len(sep_results)):
	        labels_error.append('dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	        x_error=numpy.append(x_error, i+1)

	        plt.plot(x_error[i], threshold_errors[i], 'o')
	    plt.xticks(x_error, labels_error, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("Threshold error")

	    plt.subplot(4, 2, 2)

	# proximal threshold error plot

	    x_prox_err =numpy.array([])
	    labels_prox_err = []
	    for i in range (0, len(dend_loc000),2):
	        labels_prox_err.append('dend'+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	    for i in range (0, len(prox_threshold_errors)):
	        x_prox_err=numpy.append(x, i+1)

	        plt.plot(x_prox_err[i], prox_threshold_errors[i], 'o')
	    plt.xticks(x_prox_err, labels_prox_err, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("Proximal threshold error")

	    plt.subplot(4, 2, 3)

	# distal threshold error plot

	    x_dist_err =numpy.array([])
	    labels_dist_err = []
	    for i in range (1, len(dend_loc000),2):
	        labels_dist_err.append('dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	    for i in range (0, len(dist_threshold_errors)):
	        x_dist_err=numpy.append(x_dist_err, i+1)

	        plt.plot(x_dist_err[i], dist_threshold_errors[i], 'o')
	    plt.xticks(x_dist_err, labels_dist_err, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("Distal threshold error")

	    plt.subplot(4, 2, 4)

	#peak deriv error plot

	    for i in range (0, len(sep_results)):

	        plt.plot(x_error[i], peak_deriv_errors[i], 'o')
	    plt.xticks(x_error, labels_error, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("Peak derivative error")

	    plt.subplot(4, 2, 5)

	#  degree of nonlin. error plot

	    for i in range (0, len(sep_results)):
	        plt.plot(x_error[i], nonlin_errors[i], 'o')
	    plt.xticks(x_error, labels_error, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("Degree of nonlinearity error")


	    plt.subplot(4, 2, 6)

	# suprathreshold degree of nonlin. error plot

	    for i in range (0, len(sep_results)):
	        plt.plot(x_error[i], suprath_nonlin_errors[i], 'o')
	    plt.xticks(x_error, labels_error, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("Suprath. degree of nonlinearity error")

	    plt.subplot(4, 2, 7)

	# amplitude error plot

	    for i in range (0, len(sep_results)):
	        plt.plot(x_error[i], amplitude_errors[i], 'o')
	    plt.xticks(x_error, labels_error, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("Amplitude error")

	    plt.subplot(4, 2, 8)

	# time to peak error plot

	    for i in range (0, len(sep_results)):
	        plt.plot(x_error[i], time_to_peak_errors[i], 'o')
	    plt.xticks(x_error, labels_error, rotation=20)
	    plt.tick_params(labelsize=10)
	    plt.margins(0.1)
	    plt.ylabel("Time to peak error")

	    fig = plt.gcf()
	    fig.set_size_inches(14, 18)
	    plt.savefig(path_figs + 'errors_sync' + '.pdf', dpi=600,)

	# mean values plot
	    fig3, axes3 = plt.subplots(nrows=2, ncols=1)
	    fig3.tight_layout()
	    fig3.suptitle('Synchronous inputs')
	    plt.subplot(2,2,1)

	    e_values = numpy.array([threshold_SD, sd_sep_threshold, threshold_prox_SD, sd_prox_thresholds, threshold_dist_SD, sd_dist_thresholds, amp_SD, sd_amp_at_threshold])
	    x_values =numpy.array([1,2,3,4,5,6,7,8])
	    y_values = numpy.array([experimental_mean_threshold, mean_sep_threshold, threshold_prox, mean_prox_thresholds, threshold_dist, mean_dist_thresholds, exp_mean_amp, mean_amp_at_threshold])
	    labels_values=['experimental mean threshold', 'model mean threshold', 'exp mean proximal threshold', 'model mean proximal threshold', 'exp mean distal threshold', 'model mean distal threshold','exp mean amplitude at th.', 'model mean amplitude at th.']
	    plt.errorbar(x_values, y_values, e_values, linestyle='None', marker='o')
	    plt.xticks(x_values, labels_values, rotation=10)
	    plt.margins(0.1)
	    plt.ylabel("experimental and model mean values with SD (mV)")

	    plt.subplot(2,2,2)

	    e_values = numpy.array([deriv_SD, sd_peak_dV_dt_at_threshold])
	    x_values =numpy.array([1,2])
	    y_values = numpy.array([exp_mean_peak_deriv, mean_peak_dV_dt_at_threshold])
	    labels_values=['exp mean peak dV/dt at th.', 'model mean peak dV/dt at th.']
	    plt.errorbar(x_values, y_values, e_values, linestyle='None', marker='o')
	    plt.xticks(x_values, labels_values, rotation=10)
	    plt.margins(0.1)
	    plt.ylabel("experimental and model mean max dV/dt with SD (V/s)")

	    plt.subplot(2,2,3)

	    e_values = numpy.array([nonlin_SD, sd_nonlin, suprath_nonlin_SD, suprath_sd_nonlin])
	    x_values =numpy.array([1,2,3,4])
	    y_values = numpy.array([exp_mean_nonlin, mean_nonlin, suprath_exp_mean_nonlin, suprath_mean_nonlin])
	    labels_values=['exp mean degree of nonlinearity at th.', 'model mean degree of nonlinearity at th.', 'exp mean supth. degree of nonlinearity', 'model mean supth. degree of nonlinearity']
	    plt.errorbar(x_values, y_values, e_values, linestyle='None', marker='o')
	    plt.xticks(x_values, labels_values, rotation=10)
	    plt.margins(0.1)
	    plt.ylabel("experimental and model mean degree of nonlinearity with SD (%)")

	    plt.subplot(2,2,4)

	    e_values = numpy.array([exp_mean_time_to_peak_SD, sd_time_to_peak_at_threshold])
	    x_values =numpy.array([1,2])
	    y_values = numpy.array([exp_mean_time_to_peak, mean_time_to_peak_at_threshold])
	    labels_values=['exp mean time to peak at th.', 'model mean time to peak at th.']
	    plt.errorbar(x_values, y_values, e_values, linestyle='None', marker='o')
	    plt.xticks(x_values, labels_values, rotation=10)
	    plt.margins(0.1)
	    plt.ylabel("experimental and model mean time to peak with SD (ms)")

	    fig = plt.gcf()
	    fig.set_size_inches(22, 16)
	    plt.savefig(path_figs + 'mean_values_sync' + '.pdf', dpi=600,)

	    plt.figure()
	# mean errors plot
	    plt.title('Synchronous inputs')
	    e_errors = numpy.array([sd_threshold_errors, sd_prox_threshold_errors, sd_dist_threshold_errors, sd_peak_deriv_errors, sd_nonlin_errors, suprath_sd_nonlin_errors, sd_amplitude_errors, sd_time_to_peak_errors])
	    x_errors =numpy.array([1,2,3,4,5,6,7,8])
	    y_errors = numpy.array([mean_threshold_errors,  mean_prox_threshold_errors, mean_dist_threshold_errors, mean_peak_deriv_errors, mean_nonlin_errors, suprath_mean_nonlin_errors, mean_amplitude_errors, mean_time_to_peak_errors ])
	    labels_errors=['mean threshold error', 'mean proximal threshold error', 'mean distal threshold error', 'mean peak dV/dt at th. error', 'mean degree of nonlinearity at th.error', 'mean supth. degree of nonlinearity error','mean amplitude at th. error', 'mean time to peak at th. error']
	    plt.errorbar(x_errors, y_errors, e_errors, linestyle='None', marker='o')
	    plt.xticks(x_errors, labels_errors, rotation=10)
	    plt.margins(0.1)
	    plt.ylabel("model mean errors in unit of the experimental SD (with SD)")

	    fig = plt.gcf()
	    fig.set_size_inches(16, 16)
	    plt.savefig(path_figs + 'mean_errors_sync' + '.pdf', dpi=600,)

	    exp_n=92
	    n_prox=33
	    n_dist=44

	    exp_means=[experimental_mean_threshold, threshold_prox, threshold_dist, exp_mean_peak_deriv, exp_mean_nonlin, suprath_exp_mean_nonlin,  exp_mean_amp, exp_mean_time_to_peak]
	    exp_SDs=[threshold_SD, threshold_prox_SD, threshold_dist_SD, deriv_SD, nonlin_SD, suprath_nonlin_SD,  amp_SD, exp_mean_time_to_peak_SD ]
	    exp_Ns=[exp_n, n_prox, n_dist, exp_n, exp_n, exp_n, exp_n, exp_n]

	    model_means = [mean_sep_threshold, mean_prox_thresholds, mean_dist_thresholds, mean_peak_dV_dt_at_threshold , mean_nonlin, suprath_mean_nonlin,  mean_amp_at_threshold, mean_time_to_peak_at_threshold]
	    model_SDs = [sd_sep_threshold, sd_prox_thresholds, sd_dist_thresholds, sd_peak_dV_dt_at_threshold , sd_nonlin, suprath_sd_nonlin, sd_amp_at_threshold, sd_time_to_peak_at_threshold]
	    model_N= len(sep_results)


	    return model_means, model_SDs, model_N


	def calcs_plots_async(self, model, results, dend_loc000, dend_loc_num_weight):

	    async_nonlin=self.observation['mean_async_nonlin']
	    async_nonlin_SEM=self.observation['async_nonlin_sem']
	    async_nonlin_SD=self.observation['async_nonlin_std']

	    path_figs = self.directory_figs + 'oblique/' + model.name + '/'

	    if not os.path.exists(path_figs):
	        os.makedirs(path_figs)

	    stop=len(dend_loc_num_weight)+1
	    sep=numpy.arange(0,stop,11)
	    sep_results=[]

	    max_num_syn=10
	    num = numpy.arange(0,max_num_syn+1)

	    for i in range (0,len(dend_loc000)):
	        sep_results.append(results[sep[i]:sep[i+1]])             # a list that contains the results of the 10 locations seperately (in lists)

	    # sep_results[0]-- the first location
	    # sep_results[0][5] -- the first location at 5 input
	    # sep_results[0][1][0] -- the first location at 1 input, SOMA
	    # sep_results[0][1][1] -- the first location at 1 input, dendrite
	    # sep_results[0][1][1][0] -- just needed


	    fig0, axes0 = plt.subplots(nrows=2, ncols=1)
	    fig0.tight_layout()
	    fig0.suptitle('Asynchronous inputs (red: dendritic trace, black: somatic trace)')
	    for i in range (0,len(dend_loc000)):
	        plt.subplot(5,2,i+1)
	        plt.subplots_adjust(hspace = 0.5)
	        for j, number in enumerate(num):
	            plt.plot(sep_results[i][j][0][0]['T'],sep_results[i][j][0][0]['V'], 'k')       # somatic traces
	            plt.plot(sep_results[i][j][1][0]['T'],sep_results[i][j][1][0]['V'], 'r')        # dendritic traces
	        plt.title('Input in dendrite '+str(dend_loc000[i][0])+ ' at location: ' +str(dend_loc000[i][1]))

	        plt.xlabel("time (ms)")
	        plt.ylabel("Voltage (mV)")
	    fig = plt.gcf()
	    fig.set_size_inches(12, 18)
	    plt.savefig(path_figs + 'traces_async' + '.pdf', dpi=600,)

	    soma_depol=numpy.array([])
	    soma_depols=[]
	    sep_soma_depols=[]
	    dV_dt=[]
	    sep_dV_dt=[]
	    soma_max_depols=numpy.array([])
	    soma_expected=numpy.array([])
	    sep_soma_max_depols=[]
	    sep_soma_expected=[]
	    max_dV_dt=numpy.array([])
	    sep_max_dV_dt=[]
	    nonlin=numpy.array([])
	    sep_nonlin=[]

	    nonlins_at_th=numpy.array([])


	    for i in range (0, len(sep_results)):
	        for j in range (0,max_num_syn+1):

	    # calculating somatic depolarization and first derivative
	            soma_depol=sep_results[i][j][0][0]['V'] - sep_results[i][0][0][0]['V']
	            soma_depols.append(soma_depol)

	            soma_max_depols=numpy.append(soma_max_depols,numpy.amax(soma_depol))

	            dt=numpy.diff(sep_results[i][j][0][0]['T'] )
	            dV=numpy.diff(sep_results[i][j][0][0]['V'] )
	            deriv=dV/dt
	            dV_dt.append(deriv)

	            max_dV_dt=numpy.append(max_dV_dt, numpy.amax(dV_dt))

	            if j==0:
	                soma_expected=numpy.append(soma_expected,0)
	                nonlin=numpy.append(nonlin,100)
	            else:
	                soma_expected=numpy.append(soma_expected,soma_max_depols[1]*j)
	                nonlin=numpy.append(nonlin, soma_max_depols[j]/ soma_expected[j]*100)

	        sep_soma_depols.append(soma_depols)
	        sep_dV_dt.append(dV_dt)
	        sep_soma_max_depols.append(soma_max_depols)
	        sep_soma_expected.append(soma_expected)
	        sep_max_dV_dt.append(max_dV_dt)
	        sep_nonlin.append(nonlin)
	        nonlins_at_th=numpy.append(nonlins_at_th, nonlin[4])      #degree of nonlin at 4 inputs, that is the threshold in the synchronous case

	        soma_depols=[]
	        dV_dt=[]
	        soma_max_depols=numpy.array([])
	        soma_expected=numpy.array([])
	        max_dV_dt=numpy.array([])
	        nonlin=numpy.array([])

	    expected_depol_input=numpy.array([])
	    expected_mean_depol_input=[]
	    depol_input=numpy.array([])
	    mean_depol_input=[]
	    SD_depol_input=[]
	    SEM_depol_input=[]

	    peak_deriv_input=numpy.array([])
	    mean_peak_deriv_input=[]
	    SD_peak_deriv_input=[]
	    SEM_peak_deriv_input=[]
	    n=len(sep_soma_max_depols)


	    for j in range (0, max_num_syn+1):
	        for i in range (0, len(sep_soma_max_depols)):
	            depol_input=numpy.append(depol_input,sep_soma_max_depols[i][j])
	            expected_depol_input=numpy.append(expected_depol_input,sep_soma_expected[i][j])
	            peak_deriv_input=numpy.append(peak_deriv_input,sep_max_dV_dt[i][j])
	        mean_depol_input.append(numpy.mean(depol_input))
	        expected_mean_depol_input.append(numpy.mean(expected_depol_input))
	        mean_peak_deriv_input.append(numpy.mean(peak_deriv_input))
	        SD_depol_input.append(numpy.std(depol_input))
	        SEM_depol_input.append(numpy.std(depol_input)/math.sqrt(n))
	        SD_peak_deriv_input.append(numpy.std(peak_deriv_input))
	        SEM_peak_deriv_input.append(numpy.std(peak_deriv_input)/math.sqrt(n))
	        depol_input=numpy.array([])
	        expected_depol_input=numpy.array([])
	        peak_deriv_input=numpy.array([])

	    mean_nonlin_at_th=numpy.mean(nonlins_at_th)
	    SD_nonlin_at_th=numpy.std(nonlins_at_th)
	    SEM_nonlin_at_th=SD_nonlin_at_th/math.sqrt(n)

	    plt.figure()
	    plt.suptitle('Asynchronous inputs')
	    plt.subplot(2,1,1)
	    # Expected EPSP - Measured EPSP plot
	    colormap = plt.cm.spectral      #http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
	    plt.gca().set_color_cycle([colormap(j) for j in numpy.linspace(0, 0.9, len(sep_results))])
	    for i in range (0, len(sep_results)):

	        plt.plot(sep_soma_expected[i],sep_soma_max_depols[i], '-o', label='dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	        plt.plot(sep_soma_expected[i],sep_soma_expected[i], 'k--')         # this gives the linear line
	    plt.xlabel("expected EPSP (mV)")
	    plt.ylabel("measured EPSP (mV)")
	    plt.legend(loc=2, prop={'size':10})

	    plt.subplot(2,1,2)

	    plt.errorbar(expected_mean_depol_input, mean_depol_input, yerr=SD_depol_input, linestyle='-', marker='o', color='red', label='SD')
	    plt.errorbar(expected_mean_depol_input, mean_depol_input, yerr=SEM_depol_input, linestyle='-', marker='o', color='blue', label='SEM')
	    plt.plot(expected_mean_depol_input,expected_mean_depol_input, 'k--')         # this gives the linear line
	    plt.margins(0.1)
	    plt.legend(loc=2)
	    plt.title("Summary plot of mean input-output curve")
	    plt.xlabel("expected EPSP (mV)")
	    plt.ylabel("measured EPSP (mV)")

	    fig = plt.gcf()
	    fig.set_size_inches(12, 12)
	    plt.savefig(path_figs + 'input_output_curves_async' + '.pdf', dpi=600,)


	    plt.figure()

	    plt.subplot(2,1,1)
	    plt.title('Asynchronous inputs')
	#Derivative plot
	    colormap = plt.cm.spectral
	    plt.gca().set_color_cycle([colormap(j) for j in numpy.linspace(0, 0.9, len(sep_results))])
	    for i in range (0, len(sep_results)):

	        plt.plot(num,sep_max_dV_dt[i], '-o', label='dend '+str(dend_loc000[i][0])+ ' loc ' +str(dend_loc000[i][1]))
	        plt.xlabel("# of inputs")
	        plt.ylabel("dV/dt (V/s)")
	        plt.legend(loc=2, prop={'size':10})

	    plt.subplot(2,1,2)

	    plt.errorbar(num, mean_peak_deriv_input, yerr=SD_peak_deriv_input, linestyle='-', marker='o', color='red', label='SD')
	    plt.errorbar(num, mean_peak_deriv_input, yerr=SEM_peak_deriv_input, linestyle='-', marker='o', color='blue', label='SEM')
	    plt.margins(0.01)
	    plt.legend(loc=2)
	    plt.title("Summary plot of mean peak dV/dt amplitude")
	    plt.xlabel("# of inputs")
	    plt.ylabel("dV/dt (V/s)")

	    fig = plt.gcf()
	    fig.set_size_inches(12, 12)
	    plt.savefig(path_figs + 'peak_derivative_plots_async' + '.pdf', dpi=600,)


	    fig0, axes0 = plt.subplots(nrows=5, ncols=2)
	    fig0.tight_layout()
	    fig0.suptitle('Asynchronous inputs')
	    for j in range (0,len(dend_loc000)):
	        plt.subplot(5,2,j+1)
	        x =numpy.array([])
	        labels = ['exp mean\n with SD']
	        e = numpy.array([async_nonlin_SD])
	        x2 =numpy.array([1])
	        y2 = numpy.array([async_nonlin])
	        for i in range (0, len(sep_nonlin[0])+1):
	            x=numpy.append(x, i+1)
	            labels.append(str(i)+ ' inputs')
	        for i in range (0, len(sep_nonlin[j])):
	            plt.plot(x[i+1], sep_nonlin[j][i], 'o')

	        plt.errorbar(x2, y2, e, linestyle='None', marker='o', color='blue')
	        plt.xticks(x, labels, rotation=30)
	        plt.margins(0.1)
	        plt.ylabel("Degree of nonlinearity (%)")
	        plt.title('dendrite '+str(dend_loc000[j][0])+ ' location: ' +str(dend_loc000[j][1]))

	    fig = plt.gcf()
	    fig.set_size_inches(20, 20)
	    plt.savefig(path_figs + 'nonlin_values_async' + '.pdf', dpi=600,)

	    async_nonlin_errors=[]
	    asynch_nonlin_error_at_th=numpy.array([])

	    for i in range (0, len(sep_nonlin)):
	        async_nonlin_err = numpy.array([abs(async_nonlin- async_nonlin_err)/async_nonlin_SD  for async_nonlin_err in sep_nonlin[i]])     # does the same calculation on every element of a list  #x = [1,3,4,5,6,7,8] t = [ t**2 for t in x ]
	        async_nonlin_errors.append(async_nonlin_err)
	        asynch_nonlin_error_at_th=numpy.append(asynch_nonlin_error_at_th, async_nonlin_err[4])

	    mean_nonlin_error_at_th=numpy.mean(asynch_nonlin_error_at_th)
	    SD_nonlin_error_at_th=numpy.std(asynch_nonlin_error_at_th)
	    SEM_nonlin_error_at_th=SD_nonlin_error_at_th/math.sqrt(n)


	    fig0, axes0 = plt.subplots(nrows=5, ncols=2)
	    fig0.tight_layout()
	    fig0.suptitle('Asynchronous inputs')
	    for j in range (0,len(dend_loc000)):
	        plt.subplot(5,2,j+1)
	        for i in range (0, len(async_nonlin_errors[j])):
	            plt.plot(x[i], async_nonlin_errors[j][i], 'o')

	        plt.xticks(x, labels[1:-1], rotation=20)
	        plt.margins(0.1)
	        plt.ylabel("Degree of nonlin. error (%)")
	        plt.title('dendrite '+str(dend_loc000[j][0])+ ' location: ' +str(dend_loc000[j][1]))
	    fig = plt.gcf()
	    fig.set_size_inches(18, 20)
	    plt.savefig(path_figs + 'nonlin_errors_async' + '.pdf', dpi=600,)

	    return mean_nonlin_at_th, SD_nonlin_at_th



	def validate_observation(self, observation):

		try:
			assert type(observation['mean_threshold']) is Quantity
			assert type(observation['threshold_std']) is Quantity
			assert type(observation['mean_prox_threshold']) is Quantity
			assert type(observation['prox_threshold_std']) is Quantity
			assert type(observation['mean_dist_threshold']) is Quantity
			assert type(observation['dist_threshold_std']) is Quantity
			assert type(observation['mean_peak_deriv']) is Quantity
			assert type(observation['peak_deriv_std']) is Quantity
			assert type(observation['mean_amp_at_th']) is Quantity
			assert type(observation['amp_at_th_std']) is Quantity
			assert type(observation['mean_time_to_peak']) is Quantity
			assert type(observation['time_to_peak_std']) is Quantity

		except Exception as e:
			raise ObservationError(("Observation must be of the form "
									"{'mean':float*mV,'std':float*mV}"))


	def generate_prediction(self, model):
		"""Implementation of sciunit.Test.generate_prediction."""

		traces = []

		global model_name
		model_name = model.name


		npool0 = 4
		pool0 = multiprocessing.Pool(npool0)

		binsearch_ = functools.partial(self.binsearch, model)
		result0 = pool0.map_async(binsearch_, model.dend_loc)
		results0 = result0.get()


		npool = 4
		pool = multiprocessing.Pool(npool)
		max_num_syn=10

		num = numpy.arange(0,max_num_syn+1)
		dend_loc_num_weight=[]

		dend0=[]
		dend_loc00=[]
		for i in range(0, len(model.dend_loc)):
		    if results0[i][0]==None or results0[i][0]==False :
		        if model.dend_loc[i][0] not in dend0:
		            dend0.append(model.dend_loc[i][0])
		    if results0[i][0]==None :
		        print 'the dendritic spike on at least one of the locations of dendrite ',  model.dend_loc[i][0], 'generated somatic AP'
		    if results0[i][0]==False :
		        print 'No dendritic spike could be generated on at least one of the locations of dendrite',  model.dend_loc[i][0]

		for k in range(0, len(dend0)):
		    for i in range(0, len(model.dend_loc)):
		        if model.dend_loc[i][0]==dend0[k]:
		            dend_loc00.append(model.dend_loc[i])

		dend_loc000=list(model.dend_loc)

		for i in range(0, len(dend_loc00)):
		    dend_loc000.remove(dend_loc00[i])       #dend_loc000 does not contain the dendrites in which spike causes somatic AP

		for i in range(0, len(dend_loc000)):

		    for j in num:

		        e=list(dend_loc000[i])
		        e.append(num[j])
		        e.append(results0[i][1])

		        dend_loc_num_weight.append(e)

		interval_sync=0.1

		run_synapse_ = functools.partial(self.run_synapse, model, interval=interval_sync)
		result = pool.map_async(run_synapse_,dend_loc_num_weight)
		results = result.get()


		npool1 = 4
		pool1 = multiprocessing.Pool(npool1)

		interval_async=2
		#
		run_synapse_ = functools.partial(self.run_synapse, model, interval=interval_async)
		result_async = pool1.map_async(run_synapse_,dend_loc_num_weight)
		results_async = result_async.get()

		model_means, model_SDs, model_N = self.calcs_plots(model, results, dend_loc000, dend_loc_num_weight)

		mean_nonlin_at_th_asynch, SD_nonlin_at_th_asynch = self.calcs_plots_async(model, results_async, dend_loc000, dend_loc_num_weight)


		prediction = {'model_mean_threshold':model_means[0], 'model_threshold_std': model_SDs[0],
		                'model_mean_prox_threshold':model_means[1], 'model_prox_threshold_std': model_SDs[1],
		                'model_mean_dist_threshold':model_means[2], 'model_dist_threshold_std': model_SDs[2],
						'model_mean_peak_deriv':model_means[3],'model_peak_deriv_std': model_SDs[3],
		                'model_mean_nonlin_at_th':model_means[4], 'model_nonlin_at_th_std': model_SDs[4],
		                'model_mean_nonlin_suprath':model_means[5], 'model_nonlin_suprath_std': model_SDs[5],
		                'model_mean_amp_at_th':model_means[6],'model_amp_at_th_std': model_SDs[6],
		                'model_mean_time_to_peak':model_means[7], 'model_time_to_peak_std': model_SDs[7],
		                'model_mean_async_nonlin':mean_nonlin_at_th_asynch, 'model_async_nonlin_std': SD_nonlin_at_th_asynch,
		                'model_n': model_N }
		return prediction

	def compute_score(self, observation, prediction):
		"""Implementation of sciunit.Test.score_prediction."""
		score0 = ttest_calc(observation,prediction)

		score=P_Value(score0)

		path_figs = self.directory_figs + 'oblique/' + model_name + '/'

		if not os.path.exists(path_figs):
		    os.makedirs(path_figs)

		plt.figure()
		x =numpy.arange(1,10)
		labels=['threshold', 'proximal threshold', 'distal threshold', 'peak dV/dt at th.','degree of nonlinearity at th.', 'suprath. degree of nonlinearity', 'amplitude at th.', 'time to peak at th.', 'asynch. degree of nonlin. at th.']
		plt.plot(x, score0, linestyle='None', marker='o')
		plt.xticks(x, labels, rotation=20)
		plt.tick_params(labelsize=11)
		plt.axhline(y=0.05, label='0.05', color='red')
		plt.legend()
		plt.margins(0.1)
		plt.ylabel("p values")
		fig = plt.gcf()
		fig.set_size_inches(12, 10)
		plt.savefig(path_figs + 'p_values' + '.pdf', dpi=600,)

		return score


class SomaticFeaturesTest(Test):
	"""Tests if the model enters depolarization block under current injection of increasing amplitudes."""

	def __init__(self,
			     observation = {}  ,
			     name="Somatic features test" ,
				 force_run=False,
				show_plot=True):

		Test.__init__(self,observation,name)

		self.required_capabilities += (ProducesMembranePotential,
							  		   ReceivesCurrent,)

		self.force_run = force_run
		self.directory='./temp_data/'
		self.directory_figs='./figs/'

		with open('./stimfeat/PC_newfeat_No14112401_15012303-m990803_stimfeat.json') as f:
		    self.config = json.load(f, object_pairs_hook=collections.OrderedDict)

		description = "Tests if the model enters depolarization block under current injection of increasing amplitudes."

	score_type = ZScore3

	def create_stimuli_list(self):

	    #with open('./stimfeat/PC_newfeat_No14112401_15012303-m990803_stimfeat.json') as f:
	        #config = json.load(f, object_pairs_hook=collections.OrderedDict)

	    stimulus_list=[]
	    stimuli_list=[]
	    stimuli_names=self.config['stimuli'].keys()

	    for i in range (0, len(stimuli_names)):
	        if "Apic" not in stimuli_names[i]:
	            stimulus_list.append(stimuli_names[i])
	            stimulus_list.append(self.config['stimuli'][stimuli_names[i]]['Amplitude'])
	            stimulus_list.append(self.config['stimuli'][stimuli_names[i]]['Delay'])
	            stimulus_list.append(self.config['stimuli'][stimuli_names[i]]['Duration'])
	            stimulus_list.append(self.config['stimuli'][stimuli_names[i]]['StimSectionName'])
	            stimulus_list.append(self.config['stimuli'][stimuli_names[i]]['StimLocationX'])
	            stimulus_list.append(self.config['stimuli'][stimuli_names[i]]['Type'])
	            stimulus_list.append(self.config['stimuli'][stimuli_names[i]]['RecSectionName'])
	            stimulus_list.append(self.config['stimuli'][stimuli_names[i]]['RecLocationX'])
	            stimuli_list.append(stimulus_list)
	            stimulus_list=[]

	    return stimuli_list

	def create_features_list(self, observation):

	    feature_list=[]
	    features_list=[]
	    features_names=(observation.keys())


	    for i in range (0, len(features_names)):
	        if "Apic" not in features_names[i]:
	            feature_list.append(features_names[i])
	            feature_list.append(observation[features_names[i]]['Std'])
	            feature_list.append(observation[features_names[i]]['Mean'])
	            feature_list.append(observation[features_names[i]]['Stimulus'])
	            feature_list.append(observation[features_names[i]]['Type'])
	            features_list.append(feature_list)
	            feature_list=[]

	    return features_names, features_list

	def run_stim(self, model, stimuli_list):


		stimulus_name, amplitude, delay, duration, stim_section_name, stim_location_x, stim_type, rec_section_name, rec_location_x = stimuli_list

		traces_result={}

		path = self.directory + model.name + '/soma/'

		if not os.path.exists(path):
		    os.makedirs(path)


		if stim_type == "SquarePulse":
		    file_name = path + stimulus_name + '.p'

		    if self.force_run or (os.path.isfile(file_name) is False):

		        print "running stimulus: " + stimulus_name

		        model.initialise()
		        stim_section_name = model.translate(stim_section_name, distance=0)
		        rec_section_name = model.translate(rec_section_name, distance=0)
		        model.set_cclamp_somatic_feature(float(amplitude), float(delay), float(duration), stim_section_name, stim_location_x)
		        t,v = model.run_cclamp_somatic_feature(rec_section_name, rec_location_x)

		        traces_result[stimulus_name]=[t,v]

		        pickle.dump(traces_result, gzip.GzipFile(file_name, "wb"))

		    else:
		        traces_result = pickle.load(gzip.GzipFile(file_name, "rb"))

		else:
		    traces_result=None

		return traces_result

	def analyse_traces(self, stimuli_list, traces_results, features_list):

	    feature_name, target_sd, target_mean, stimulus, feature_type = features_list
	    target_sd=float(target_sd)
	    target_mean=float(target_mean)

	    feature_result={}
	    trace = {}
	    for i in range (0, len(traces_results)):
	        for key, value in traces_results[i].iteritems():
	            stim_name=key
	        if stim_name == stimulus:

	            trace['T'] = traces_results[i][key][0]
	            trace['V'] = traces_results[i][key][1]

	    for i in range (0, len(stimuli_list)):
	        if stimuli_list[i][0]==stimulus:

	            trace['stim_start'] = [float(stimuli_list[i][2])]
	            trace['stim_end'] = [float(stimuli_list[i][2])+float(stimuli_list[i][3])]

	    traces = [trace]
	    #print traces

	    efel_results = efel.getFeatureValues(traces,[feature_type])

	    feature_values=efel_results[0][feature_type]


	    #feature_error=abs(feature_values-target_mean)/target_sd
	    feature_mean=numpy.mean(feature_values)
	    feature_sd=numpy.std(feature_values)
	    #feature_error_mean=numpy.mean(feature_error)
	    #feature_error_sd=numpy.std(feature_error)


	    feature_result={feature_name:{'traces':traces,
	                                  'feature values': feature_values,
	                                  #'feature error':feature_error,
	                                  'feature mean': feature_mean,
	                                  'feature sd': feature_sd}}
	                                  #'mean feature error':feature_error_mean,
	                                  #'feature error sd':feature_error_sd,
	                                  #'target mean': target_mean,
	                                  #'target sd': target_sd}}

	    #feature_result[feature_name]=[traces, efel_results]
	    return feature_result

	def create_figs(self, model, traces_results, features_names, feature_results_dict, observation):

	    path_figs = self.directory_figs + 'soma/' + model.name + '/'

	    if not os.path.exists(path_figs):
	        os.makedirs(path_figs)

	    print "The figures are saved in the directory: ", path_figs

	    plt.figure(1)
	    #key=sorted()
	    for i in range (0, len(traces_results)):
	        for key, value in traces_results[i].iteritems():
	            plt.plot(traces_results[i][key][0], traces_results[i][key][1], label=key)
	    plt.legend(loc=2)
	    plt.savefig(path_figs + 'traces' + '.pdf', dpi=600,)

	    fig, axes = plt.subplots(nrows=4, ncols=2)
	    fig.tight_layout()
	    for i in range (0, len(traces_results)):

	        for key, value in traces_results[i].iteritems():

	            plt.subplot(4,2,i+1)
	            plt.plot(traces_results[i][key][0], traces_results[i][key][1])
	            plt.title(key)
	            plt.xlabel("ms")
	            plt.ylabel("mV")
	    fig = plt.gcf()
	    fig.set_size_inches(12, 10)
	    plt.savefig(path_figs + 'traces_subplots' + '.pdf', dpi=600,)

	    axs = plottools.tiled_figure("absolute features", figs={}, frames=1, columns=1, orientation='page',
	                            height_ratios=[0.9,0.1], top=0.97, bottom=0.05, left=0.25, right=0.97, hspace=0.1, wspace=0.2)

	    for i in range (len(features_names)):
	        if "Apic" not in features_names[i]:
	            feature_name=features_names[i]
	            y=i
	            axs[0].errorbar(feature_results_dict[feature_name]['feature mean'], y, xerr=feature_results_dict[feature_name]['feature sd'], marker='o', color='blue', clip_on=False)
	            axs[0].errorbar(float(observation[feature_name]['Mean']), y, xerr=float(observation[feature_name]['Std']), marker='o', color='red', clip_on=False)
	    axs[0].yaxis.set_ticks(range(len(features_names)))
	    axs[0].set_yticklabels(features_names)
	    axs[0].set_ylim(-1, len(features_names))
	    axs[0].set_title('Absolute Features')
	    plt.savefig(path_figs + 'absolute_features' + '.pdf', dpi=600,)

	'''
	def validate_observation(self, observation):

		try:
			assert type(observation['mean_Ith']) is Quantity
			assert type(observation['Ith_std']) is Quantity
			assert type(observation['mean_Veq']) is Quantity
			assert type(observation['Veq_std']) is Quantity
		except Exception as e:
			raise ObservationError(("Observation must be of the form "
									"{'mean':float*mV,'std':float*mV}"))
	'''

	def generate_prediction(self, model):
		"""Implementation of sciunit.Test.generate_prediction."""

		global model_name
		model_name = model.name

		npool = 4
		pool = multiprocessing.Pool(npool)

		stimuli_list=self.create_stimuli_list()

		run_stim_ = functools.partial(self.run_stim, model)
		traces_result = pool.map_async(run_stim_, stimuli_list)
		traces_results = traces_result.get()

		features_names, features_list = self.create_features_list(self.observation)

		analyse_traces_ = functools.partial(self.analyse_traces, stimuli_list, traces_results)
		feature_result = pool.map_async(analyse_traces_, features_list)
		feature_results = feature_result.get()

		feature_results_dict={}
		for i in range (0,len(feature_results)):
		    feature_results_dict.update(feature_results[i])  #concatenate dictionaries

		self.create_figs(model, traces_results, features_names, feature_results_dict, self.observation)

		prediction = feature_results_dict

		return prediction

	def compute_score(self, observation, prediction):
		"""Implementation of sciunit.Test.score_prediction."""

		path_figs = self.directory_figs + 'soma/' + model_name + '/'

		if not os.path.exists(path_figs):
		    os.makedirs(path_figs)

		score_sum, feature_results_dict, features_names  = zscore3(observation,prediction)

		axs2 = plottools.tiled_figure("features", figs={}, frames=1, columns=1, orientation='page',
		                              height_ratios=[0.9,0.1], top=0.97, bottom=0.05, left=0.25, right=0.97, hspace=0.1, wspace=0.2)

		for i in range (len(features_names)):
		    if "Apic" not in features_names[i]:
		        feature_name=features_names[i]
		        y=i
		        axs2[0].errorbar(feature_results_dict[feature_name]['mean feature error'], y, xerr=feature_results_dict[feature_name]['feature error sd'], marker='o', color='blue', clip_on=False)
		axs2[0].yaxis.set_ticks(range(len(features_names)))
		axs2[0].set_yticklabels(features_names)
		axs2[0].set_ylim(-1, len(features_names))
		axs2[0].set_title('Features')
		plt.savefig(path_figs + 'Feature_errors' + '.pdf', dpi=600,)

		score=ZScore3(score_sum)
		return score

copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)
