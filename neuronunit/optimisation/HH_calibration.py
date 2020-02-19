#!/usr/bin/env python
# coding: utf-8

# In[ ]:


monitor = None
neuron = None


# In[38]:


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
#neuron.I = '100*uA'# * i'# / num_neurons'

#monitor = SpikeMonitor(neuron)
#state_monitor = b2.StateMonitor(neuron, ["v"], record=True)
#state_dic = state_monitor.get_states()
#run(duration)


# In[39]:


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
#print(vm)

vm_b = AnalogSignal(vm_b,units = pq.V,sampling_period = float(0.001) * pq.s)
plt.plot(vm_b.times,vm_b.magnitude)


plt.xlabel('t (s)')
plt.ylabel('Vm (mV)')
plt.show()


# In[ ]:





# In[40]:


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


class testCrucialBackendsSucceed(unittest.TestCase):
    def setUp(self):
        model_parameters.MODEL_PARAMS.keys()
        #self.backends =  ["HH"]

        #raw_attrs = {k:np.mean(v) for k,v in model_parameters.MODEL_PARAMS[backend].items()}
        #self.backends = backends
        self.model_parameters = model_parameters
        
    def test_must_pass_0(self,vm_b=None,attrs_=None):
        fig, axs = plt.subplots(2,1,figsize=(40, 40))
        cnt=0
        b = str("HH")
        attrs = {k:np.mean(v) for k,v in self.model_parameters.MODEL_PARAMS[b].items()}
        if attrs_ is not None:
            for k,v in attrs_.items():
                attrs[k] = attrs_[k]
            
        pre_model = DataTC()
        if str("V_REST") in attrs.keys():
            attrs["V_REST"] = -75.0
        pre_model.attrs = attrs
        pre_model.backend = b
        vm,_ = inject_and_plot_model(pre_model.attrs,b)
        vm3 = [v/1000.0 for v in vm]
        print('suggesting should be in model','vm3 = [v/1000.0 for v in vm]')
        if vm_b is not None:
            axs[cnt].plot(vm_b.times,vm_b.magnitude)
        axs[cnt].plot(vm.times,vm3)
        axs[cnt].set_title(b)
        cnt+=1
        thresh = threshold_detection(vm,0.0*pq.mV)

        if len(thresh)>0 and vm is not None:
            boolean = True
        else:
            boolean = False
        self.assertTrue(boolean)
        vm,_ = inject_and_plot_passive_model(pre_model.attrs,b)
        axs[cnt].plot(vm.times,vm.magnitude)
        axs[cnt].set_title(b)
        cnt+=1

        if len(vm)>0 and vm is not None:
            boolean = True
        else:
            boolean = False
        self.assertTrue(boolean)
        return attrs

    #return True


# In[41]:


#a = testCrucialBackendsSucceed()
#a.setUp()
#a.test_must_pass_0(vm_b)


# In[42]:


a = testCrucialBackendsSucceed()
a.setUp() 
attrs = a.test_must_pass_0(copy.copy(vm_b))


# In[43]:



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


# In[44]:



attrs


# In[46]:


C_m  =   1.0
"""membrane capacitance, in uF/cm^2"""

g_Na = 120.0
"""Sodium (Na) maximum conductances, in mS/cm^2"""

g_K  =  36.0
"""Postassium (K) maximum conductances, in mS/cm^2"""

g_L  =   0.3
"""Leak maximum conductances, in mS/cm^2"""

E_Na =  50.0
"""Sodium (Na) Nernst reversal potentials, in mV"""

E_K  = -77.0
"""Postassium (K) Nernst reversal potentials, in mV"""

E_L  = -54.387
"""Leak Nernst reversal potentials, in mV"""


# In[47]:


new_attrs = {}

new_attrs['E_L'] = -54.387


# In[48]:


attrs = a.test_must_pass_0(copy.copy(vm_b),attrs_=new_attrs)


# In[68]:


import scipy as sp
import pylab as plt
from scipy.integrate import odeint

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    C_m  =   1.0
    """membrane capacitance, in uF/cm^2"""

    g_Na = 120.0
    """Sodium (Na) maximum conductances, in mS/cm^2"""

    g_K  =  36.0
    """Postassium (K) maximum conductances, in mS/cm^2"""

    g_L  =   0.3
    """Leak maximum conductances, in mS/cm^2"""

    E_Na =  50.0
    """Sodium (Na) Nernst reversal potentials, in mV"""

    E_K  = -77.0
    """Postassium (K) Nernst reversal potentials, in mV"""

    E_L  = -54.387
    """Leak Nernst reversal potentials, in mV"""


    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*sp.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*sp.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*sp.exp(-(V+65) / 80.0)

    def I_Na(self, V, m, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)

        |  :param V:
        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_K  * n**4 * (V - self.E_K)
    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_L * (V - self.E_L)

    def I_inj(self, t,constant=None):
        """
        External Current

        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        
        return 0*(t<100) +2.240341901779175*(t>100) -2.240341901779175*(t>1100)
        #2.240341901779175 pA

    @staticmethod
    def dALLdt(X, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V, m, h, n = X

        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return dVdt, dmdt, dhdt, dndt

    def Main(self,t=None):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        if t is not None:
            self.t = t
        else:
            self.t = sp.arange(0.0, 1300.0, 0.01)
            """ The time to integrate over """

        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(self,))
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)

        plt.figure()

        #plt.subplot(4,1,1)
        len(V)
        scale = len(V)/1.3
        vm = AnalogSignal(V,units = pq.V,sampling_rate = scale * pq.Hz)
        plt.title('Hodgkin-Huxley Neuron')
        plt.plot(vm.times, vm, 'k')
        plt.ylabel('V (mV)')

        """

        plt.subplot(4,1,4)
        i_inj_values = [self.I_inj(t) for t in self.t]
        plt.plot(self.t, i_inj_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-1, 40)
        """

        plt.show()
        return vm

#if __name__ == '__main__':
runner = HodgkinHuxley()
t = sp.arange(0.0, 1300.0, 0.01)
""" The time to integrate over """

vm = runner.Main(t=t)


# In[69]:


new_attrs = {}
new_attrs['E_L'] = -54.387
b = str("HH")
attrs = {k:np.mean(v) for k,v in model_parameters.MODEL_PARAMS[b].items()}
if new_attrs is not None:
    for k,v in new_attrs.items():
        attrs[k] = new_attrs[k]

pre_model = DataTC()

pre_model.attrs = attrs
pre_model.backend = b
vm,plt = inject_and_plot_model(pre_model.attrs,b)
#dtc.rheobase


# In[70]:


dtc = dtc_to_rheo(pre_model)


# In[67]:


print(dtc.rheobase)


# In[ ]:


#2.240341901779175 pA

