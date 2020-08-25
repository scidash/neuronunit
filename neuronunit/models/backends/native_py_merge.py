

import scipy as sp
#import pylab as plt
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

