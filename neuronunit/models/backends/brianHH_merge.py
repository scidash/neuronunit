#!/usr/bin/env python
# coding: utf-8

import unittest
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from neuronunit.optimisation.optimization_management import inject_and_plot_model, dtc_to_rheo
from neuronunit.optimisation.optimization_management import inject_and_plot_passive_model
import numpy as np
from neuronunit.optimisation.data_transport_container import DataTC
from neuronunit.optimisation import model_parameters
from elephant.spike_train_generation import threshold_detection
import quantities as pq
import matplotlib.pyplot as plt
plt.plot([0,1],[1,0])
plt.show()
import matplotlib.pyplot as plt
from brian2 import *
import brian2 as b2
from neo import AnalogSignal
import quantities as pq
import copy
num_neurons = 1
#duration = 2*second

# Parameters
area = 20000*umetre**2
Cm = 1*ufarad*cm**-2 * area
gl = 5e-5*siemens*cm**-2 * area
El = -65*mV
EK = -90*mV
ENa = 50*mV
g_na = 100*msiemens*cm**-2 * area
g_kd = 30*msiemens*cm**-2 * area
VT = -63*mV

# Parameters
area = 20000*umetre**2
Cm = 1*ufarad*cm**-2 * area
gl = 5e-5*siemens*cm**-2 * area
El = -65*mV
EK = -90*mV
ENa = 50*mV
g_na = 100*msiemens*cm**-2 * area
g_kd = 30*msiemens*cm**-2 * area
VT = -63*mV

# The model
eqs = Equations('''
dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
dm/dt = 0.32*(mV**-1)*4*mV/exprel((13.*mV-v+VT)/(4*mV))/ms*(1-m)-0.28*(mV**-1)*5*mV/exprel((v-VT-40.*mV)/(5*mV))/ms*m : 1
dn/dt = 0.032*(mV**-1)*5*mV/exprel((15.*mV-v+VT)/(5*mV))/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
I : amp
''')
# Threshold and refractoriness are only used for spike counting
neuron = NeuronGroup(1, eqs,
                    threshold='v > -40*mV',
                    refractory='v > -40*mV',
                    method='exponential_euler')
neuron.v = El


neuron.v = El
neuron.I = '0.0183*nA'# * i'# / num_neurons'

monitor = SpikeMonitor(neuron)
state_monitor = b2.StateMonitor(neuron, ["v"], record=True)
neuron.I = '0.0*nA'# * i'# / num_neurons'
dur0 = 0.1*second
run(dur0)
neuron.I = '0.0185*nA'# * i'# / num_neurons'
dur1 = 1.0*second
run(dur1)
dur2 = 0.2*second
neuron.I = '0.0*nA'# * i'# / num_neurons'
run(dur2)

print(monitor.count)
state_dic = state_monitor.get_states()

from neo import AnalogSignal
import quantities as pq
vm_b = state_dic['v']
vm_b = [ float(i) for i in vm_b ]
vm_b = AnalogSignal(vm_b,units = pq.V,sampling_period = float(0.001) * pq.s)
plt.plot(vm_b.times,vm_b.magnitude)
