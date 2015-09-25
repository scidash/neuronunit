import os

from quantities import mV, nA, ms, V, s
import sciunit
from hippounit import models
from hippounit import tests
from hippounit import capabilities

import matplotlib.pyplot as plt
import math


experimental_mean_threshold=3.4*mV
threshold_SEM=0.2*mV
exp_n=92
threshold_SD=float(threshold_SEM*math.sqrt(exp_n))*mV

threshold_prox=4.5*mV
threshold_prox_SEM=0.5*mV
n_prox=33
threshold_prox_SD=float(threshold_prox_SEM*math.sqrt(n_prox))*mV

threshold_dist=2.6*mV
threshold_dist_SEM=0.2*mV
n_dist=44
threshold_dist_SD=float(threshold_dist_SEM*math.sqrt(n_dist))*mV

exp_mean_nonlin=142    # %
nonlin_SEM=9
nonlin_SD=nonlin_SEM*math.sqrt(exp_n)

suprath_exp_mean_nonlin=129.0     # %
suprath_nonlin_SEM=6.0
suprath_nonlin_SD=suprath_nonlin_SEM*math.sqrt(exp_n)

exp_mean_peak_deriv=2.6*V /s     #V/s
deriv_SEM=0.4*V /s
deriv_SD=float(deriv_SEM*math.sqrt(exp_n))*V /s

exp_mean_amp= 4.828*mV  # calculated from threshold (expected value for linearity) and from degree of nonlin
                    #4.45 using digitizeIt on fig 1 M
amp_SEM= 0.28*mV  #using digitizeIt on fig 1 M
amp_SD=float(amp_SEM*math.sqrt(exp_n))*mV

exp_mean_time_to_peak=13.5*ms
exp_mean_time_to_peak_SEM=0.2*ms
exp_mean_time_to_peak_SD=float(exp_mean_time_to_peak_SEM*math.sqrt(exp_n))*ms

async_nonlin=104.0
async_nonlin_SEM=8.0
async_n=23
async_nonlin_SD=async_nonlin_SEM*math.sqrt(async_n)


observation = {'mean_threshold':experimental_mean_threshold,'threshold_sem':threshold_SEM, 'threshold_std': threshold_SD,
                'mean_prox_threshold':threshold_prox,'prox_threshold_sem':threshold_prox_SEM, 'prox_threshold_std': threshold_prox_SD,
                'mean_dist_threshold':threshold_dist,'dist_threshold_sem':threshold_dist_SEM, 'dist_threshold_std': threshold_dist_SD,
                'mean_nonlin_at_th':exp_mean_nonlin,'nonlin_at_th_sem':nonlin_SEM, 'nonlin_at_th_std': nonlin_SD,
                'mean_nonlin_suprath':suprath_exp_mean_nonlin,'nonlin_suprath_sem':suprath_nonlin_SEM, 'nonlin_suprath_std': suprath_nonlin_SD,
                'mean_peak_deriv':exp_mean_peak_deriv,'peak_deriv_sem':deriv_SEM, 'peak_deriv_std': deriv_SD,
                'mean_amp_at_th':exp_mean_amp,'amp_at_th_sem':amp_SEM, 'amp_at_th_std': amp_SD,
                'mean_time_to_peak':exp_mean_time_to_peak,'time_to_peak_sem':exp_mean_time_to_peak_SEM, 'time_to_peak_std': exp_mean_time_to_peak_SD,
                'mean_async_nonlin':async_nonlin,'async_nonlin_sem':async_nonlin_SEM, 'async_nonlin_std': async_nonlin_SD,
                'exp_n':exp_n, 'prox_n':n_prox, 'dist_n':n_dist, 'async_n': async_n}

show_plot=False
test = tests.ObliqueIntegrationTest(observation, force_run_synapse=False, force_run_bin_search=False, show_plot=show_plot)

model = models.Bianchi()

score = test.judge(model)

score.summarize()

if show_plot: plt.show()
