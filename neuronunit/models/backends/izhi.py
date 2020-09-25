import math

from quantities import mV, ms, s, V
import sciunit
from neo import AnalogSignal
import neuronunit.capabilities as cap
import numpy as np
from .base import *
import quantities as qt
#import matplotlib as mpl
try:
    import asciiplotlib as apl
except:
    pass
import numpy
voltage_units = mV

from elephant.spike_train_generation import threshold_detection
import time
def timer(func):
    def inner(*args, **kwargs):
        t1 = time.time()
        f = func(*args, **kwargs)
        t2 = time.time()
        print('time taken on block {0} '.format(t2-t1))
        return f
    return inner

from numba import jit, autojit
import cython
@jit(nopython=True)
def get_vm_new(C=89.7960714285714, a=0.01, b=15, c=-60, d=10, k=1.6, vPeak=(86.364525297619-65.2261863636364), vr=-65.2261863636364, vt=-50, Iext=[]):
    '''
    dt determined by
    Apply izhikevich equation as model
    This function can't get too pythonic (functional), it needs to be a simple loop for
    numba/jit to understand it.
    '''
    N = len(Iext)
    ## was
    # dt = 0.005
    ##
    dt = 0.125
    v = []#0 for i in range(0,N)]#np.zeros(N)
    u = []#0 for i in range(0,N)]#np.zeros(N)
    v.app(vr)
    u.app(0)
    for m in range(0,N-1):
        #v.app(0)
        #u.app(0)

        vT = v[-1]+ (dt/2) * (k*(v[-1] - vr)*(v[-1] - vt)-u[-1] + Iext[m])/C;
        v.app(vT + (dt/2)  * (k*(v[-1] - vr)*(v[-1] - vt)-u[-1] + Iext[m])/C)
        u.app(u[-1] + dt * a*(b*(v[-1]-vr)-u[-1]));
        u.app(u[-1] + dt * a*(b*(v[m+1]-vr)-u[m]));
        if v[-1]>= vPeak:# a spike is fired!
            v[-1] = vPeak;#  padding the spike amplitude
            v.app(c);#  membrane voltage reset
            u.app(u[-1] + d);# recovery variable update

    return v


#@cython.boundscheck(False)
#@cython.wraparound(False)
#@jit#(nopython=True)
@jit(nopython=True)
def get_vm_matlab(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6, 
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0):

    '''
    this was very slow
    '''
    tau = dt = 0.125
    u = []
    v = []
    #tau=0.125; #dt
    #I = 0
    u.append(0)
    v.append(vr)
    for i in range(N-1):
        u.append(0)
        v.append(vr)
        I = 0

        if start <= i <= stop:
            I = amp

        # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C

        ##  Cell-type specific dynamics
        if celltype < 5: ## default 
            u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable
        else:
            if celltype == 5:  # For FS neurons, include nonlinear U(v): U(v) = 0 when v<vb ; U(v) = 0.125(v-vb) when v>=vb (d=vb=-55)
                if v[i+1] < d:
                    u[i+1] = u[i] + tau*a*(0-u[i])
                else:
                    u[i+1] = u[i] + tau*a*((0.125*(v[i]-d)**3)-u[i])
                
            if celltype == 6: # For TC neurons, reset b
               if v[i+1] > -65: 
                   b=0;
               else:
                   b=15;
               
               u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]);
            if celltype==7: #For TRN neurons, reset b
                if v[i+1] > -65:
                    b=2;
                else:
                    b=10;
                
                u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]);
            
        

        #  Check if spike occurred and need to reset
        if celltype < 4 or celltype == 5 or celltype == 7: # default
            if v[i+1]>=vPeak:
                v[i]=vPeak;
                v[i+1]=c;
                if celltype != 5:
                    u[i+1]=u[i+1]+d;  # reset u, except for FS cells
            
        if celltype == 4: # LTS cell
            if v[i+1] > (vPeak - 0.1*u[i+1]):
                v[i]= vPeak - 0.1*u[i+1];
                v[i+1] = c+0.04*u[i+1]; # Reset voltage
                if (u[i]+d)<670:
                    u[i+1]=u[i+1]+d; # Reset recovery variable
                else:
                    u[i+1] = 670;
                
            
        if celltype == 6: # TC cell
            if v[i+1] > (vPeak + 0.1*u[i+1]):
                v[i]= vPeak + 0.1*u[i+1];
                v[i+1] = c-0.1*u[i+1]; # Reset voltage
                u[i+1]=u[i+1]+d;
    #print(np.std(v),'gets here')
    return v
@jit(nopython=True)
def get_vm_matlab_four(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6, 
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0):
    tau = dt = 0.125
    #I = 0
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0

        if start <= i <= stop:
            I = amp

        # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C

        if v[i+1] > (vPeak - 0.1*u[i+1]):
            v[i] = vPeak - 0.1*u[i+1]
            v[i+1] = c + 0.04*u[i+1]; # Reset voltage
            if (u[i]+d)<670:
                u[i+1] = u[i+1]+d; # Reset recovery variable
            else:
                u[i+1] = 670;                
            
    return v

@jit(nopython=True)
def get_vm_matlab_five(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6, 
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0):
    
    tau= dt = 0.125; #dt
    #I = 0
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0

        if start <= i <= stop:
            I = amp

        # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C

        #u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable
        if v[i+1] < d:
            u[i+1] = u[i] + tau*a*(0-u[i])
        else:
            u[i+1] = u[i] + tau*a*((0.125*(v[i]-d)**3)-u[i])
        if v[i+1]>=vPeak:
            v[i]=vPeak;
            v[i+1]=c;

    return v


@jit(nopython=True)
def get_vm_matlab_seven(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6, 
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0):
    tau= dt = 0.125; #dt

    #I = 0
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0

        if start <= i <= stop:
            I = amp

        # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C


        if v[i+1] > -65:
            b=2;
        else:
            b=10;
        
        u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]);
        if v[i+1]>=vPeak:
            v[i]=vPeak;
            v[i+1]=c;
            u[i+1]=u[i+1]+d;  # reset u, except for FS cells

            
    return v

@jit(nopython=True)
def get_vm_matlab_six(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6, 
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0):
    tau= dt = 0.125; #dt

    #I = 0
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0

        if start <= i <= stop:
            I = amp

        # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C


        u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]);
        if v[i+1] > -65: 
            b=0;
        else:
            b=15;
        if v[i+1] > (vPeak + 0.1*u[i+1]):
            v[i]= vPeak + 0.1*u[i+1];
            v[i+1] = c-0.1*u[i+1]; # Reset voltage
            u[i+1]=u[i+1]+d;
            
    return v


#@jit(fastmath=True)
@jit(nopython=True)
def get_vm_matlab_one_two_three(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6, 
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,
          N=0,start=0,stop=0,amp=0):
    tau= dt = 0.125; #dt

    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0

        if start <= i <= stop:
            I = amp
        # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C    
        u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable
        if v[i+1]>=vPeak:
            v[i]=vPeak
            v[i+1]=c
            u[i+1]=u[i+1]+d  # reset u, except for FS cells            
    return v
        
    

#@numba
@jit#(nopython=True)
def get_vm(C=89.7960714285714, a=0.01, b=15, c=-60, d=10, k=1.6, vPeak=(86.364525297619-65.2261863636364), vr=-65.2261863636364, vt=-50, Iext=[]):
    '''
    dt determined by
    Apply izhikevich equation as model
    This function can't get too pythonic (functional), it needs to be a simple loop for
    numba/jit to understand it.
    '''
    N = len(Iext)
    dt = 0.01

    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for m in range(0,N-1):

        #    V = V + tau*(0.04*V^2+4.1*V+108-u+I);
        # as opposed to
        #    V = V + tau*(0.04*V^2+5*V+140-u+I);

        vT = v[m]+ (dt/2) * (k*(v[m] - vr)*(v[m] - vt)-u[m] + Iext[m])/C;
        v[m+1] = vT + (dt/2)  * (k*(v[m] - vr)*(v[m] - vt)-u[m] + Iext[m])/C;
        u[m+1] = u[m] + dt * a*(b*(v[m]-vr)-u[m]);
        u[m+1] = u[m] + dt * a*(b*(v[m+1]-vr)-u[m]);
        if v[m+1]>= vPeak:# a spike is fired!
            v[m] = vPeak;#  padding the spike amplitude
            v[m+1] = c;#  membrane voltage reset
            u[m+1] = u[m+1] + d;# recovery variable update

    return v
'''
def get_vm(C=89.7960714285714, a=0.01, b=15, c=-60, d=10, k=1.6, vPeak=(86.364525297619-65.2261863636364), vr=-65.2261863636364, vt=-50, dt=0.010, Iext=[]):

    M = N = len(Iext)
    #N = size(current,1);
    #M = size(current,2);
    delta = 0.5; # resolution of simulation

    ### Neuron Parameters
    C = 281;# # capacitance in pF ... this is 281*10^(-12) F
    g_L = 30;# # leak conductance in nS
    E_L = -70.6;# # leak reversal potential in mV ... this is -0.0706 V
    delta_T = 2;# # slope factor in mV
    V_T = -50.4;# # spike threshold in mV
    tau_w = 144;# # adaptation time constant in ms
    V_peak = 20;# # when to call action potential in mV
    b = 0.0805;# # spike-triggered adaptation

    #loadPhysicalConstants;

    ### Init variables
    V = zeros(N,M);
    w = zeros(N,M);

    ### Boltzmann distributed initial conditions
    sigma_sq = 1/(thermoBeta*(C*10^(-12))); # make sure C is in F
    ## This variance is on the order of 1.5*10^(-11)
    V[1,:] = E_L/1000 + sqrt(sigma_sq)*randn(1,M); # make sure E_L is in V
    V[1,:] = V(1,:)*1000; # convert initial conditions into mV
    w[1,:] = 0; # initial condition is w = 0

    for n in range(0,N-1):

        V[n+1,:] = V[n,:] + (delta/C)*(spiking(V[n,:],g_L,E_L,delta_T,V_T)-w[n,:]+ current[n+1,:]);
        w[n+1,:] = w[n,:] + (delta/tau_w)*(a*(V[n,:]-E_L) - w[n,:]);
        
        ## spiking mechanism
        alreadySpiked = (V[n,:] == V_peak);
        V[n+1,alreadySpiked] = E_L;
        w[n+1,alreadySpiked] = w[n,alreadySpiked] + b;

        justSpiked = (V[n+1,:] > V_peak);
        V[n+1,justSpiked] = V_peak;
 
    return V
'''

class IZHIBackend(Backend):

    name = 'IZHI'

    def init_backend(self, attrs=None,DTC=None,
                     debug = False):
        super(IZHIBackend,self).init_backend()
        self.model._backend.use_memory_cache = False
        #self.vM = None
        self.attrs = attrs
        self.debug = debug
        self.temp_attrs = None
        self.default_attrs = {'C':89.7960714285714, 
            'a':0.01, 'b':15, 'c':-60, 'd':10, 'k':1.6, 
            'vPeak':(86.364525297619-65.2261863636364), 
            'vr':-65.2261863636364, 'vt':-50, 'celltype':1}

        if type(attrs) is not type(None):
            self.attrs = attrs
            # set default parameters anyway.
        if type(DTC) is not type(None):
            if type(DTC.attrs) is not type(None):
                self.set_attrs(**DTC.attrs)
        if self.attrs is None:
            self.attrs = self.default_attrs

            #if hasattr(DTC,'current_src_name'):
            #    self._current_src_name = DTC.current_src_name
            #if hasattr(DTC,'cell_name'):
            #    self.cell_name = DTC.cell_name

    def get_spike_count(self):
        #print(self.vM)
        #print(np.max(self.vM))
        thresh = threshold_detection(self.vM,0*qt.mV)
        #print(len(thresh),'spike count')
        return len(thresh)


    def set_stop_time(self, stop_time = 650*pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    #@jit
    #@timer

    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        And must destroy the hoc vectors that comprise it.
        """
        if type(self.vM) is not type(None):
            return self.vM


        if type(self.vM) is type(None):

            everything = copy.copy(self.attrs)
            if hasattr(self,'Iext'):
                everything.update({'Iext':self.Iext})
            
            if 'current_inj' in everything.keys():
                everything.pop('current_inj',None)
            everything = copy.copy(self.attrs)
            if 'current_inj' in everything.keys():
                everything.pop('current_inj',None)
            self.attrs['celltype'] = math.ceil(self.attrs['celltype'])
            if self.attrs['celltype'] <= 3:   
                everything.pop('celltype',None)         
                v = get_vm_matlab_one_two_three(**everything)
            else:
                if self.attrs['celltype'] == 4:
                    v = get_vm_matlab_four(**everything)
                if self.attrs['celltype'] == 5:
                    v = get_vm_matlab_five(**everything)
                if self.attrs['celltype'] == 6:
                    v = get_vm_matlab_six(**everything)
                if self.attrs['celltype'] == 7:
                    #print('gets into multiple regimes',self.attrs['celltype'])

                    v = get_vm_matlab_six(**everything)
            #import pdb
            #pdb.set_trace()
            self.vM = AnalogSignal(v,
                                units=pq.mV,
                                sampling_period=0.125*pq.ms)
            #if np.min(self.vM)<-70:
            #    print(np.min(self.vM))
            #self.get_spike_count()
            #import asciiplotlib as apl
            #fig = apl.figure()
            #fig.plot([float(f) for f in self.vM.times], [float(f) for f in self.vM], width=100, height=20)
            #try:
            #    fig.show()
            #except:
            #    pass    
            
        return self.vM

    def set_attrs(self, attrs):
        self.attrs = attrs
        self.model.attrs.update(attrs)

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    #@timer
    #@jit
    def inject_square_current(self, current):#, section = None, debug=False):
        """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """
        attrs = copy.copy(self.model.attrs)
        #print(attrs)
        self.attrs = attrs
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        amplitude = float(c['amplitude'])#*10.125.0.125#*1000.0 #this needs to be in every backends
        duration = float(c['duration'])#/dt#/dt.rescale('ms')
        delay = float(c['delay'])#/dt#.rescale('ms')
        #if 'sim_length' in c.keys():
        #    sim_length = c['sim_length']#.simplified
        tMax = delay + duration + 200.0#/dt#*pq.ms

        self.set_stop_time(tMax*pq.ms)
        tMax = self.tstop

        NORMAL = True
        if NORMAL == True:
            N = int(tMax/0.125)
        else:
            #print('adding in larger N seems to artificially dilate the width of neural events')
            N = int(tMax/0.125)*100

        Iext = np.zeros(N)
        delay_ind = int((delay/tMax)*N)
        duration_ind = int((duration/tMax)*N)

        Iext[0:delay_ind-1] = 0.0
        Iext[delay_ind:delay_ind+duration_ind-1] = amplitude
        Iext[delay_ind+duration_ind::] = 0.0
        self.Iext = None
        self.Iext = Iext

        #def 
        everything = copy.copy(self.attrs)
        everything.update({'N':len(Iext)})

        #everything.update({'Iext':Iext})
        everything.update({'start':delay_ind})
        everything.update({'stop':delay_ind+duration_ind})
        everything.update({'amp':amplitude})

        if 'current_inj' in everything.keys():
            everything.pop('current_inj',None)

        self.attrs['celltype'] = math.ceil(self.attrs['celltype'])
        if self.attrs['celltype'] <= 3:   
            everything.pop('celltype',None)         
            v = get_vm_matlab_one_two_three(**everything)
        else:
            #print('gets into multiple regimes',self.attrs['celltype'])

            #print('still slow',self.attrs['celltype'])
            if self.attrs['celltype'] == 4:
                v = get_vm_matlab_four(**everything)
            if self.attrs['celltype'] == 5:
                v = get_vm_matlab_five(**everything)
            if self.attrs['celltype'] == 6:
                v = get_vm_matlab_six(**everything)
            if self.attrs['celltype'] == 7:
                v = get_vm_matlab_seven(**everything)
        

        self.model.attrs.update(attrs)

        self.vM = AnalogSignal(v,
                            units=pq.mV,
                            sampling_period=0.125*pq.ms)
        #if np.min(self.vM)<-70:
        #    print(np.min(self.vM))

        #print(self.attrs['celltype'])
        #print(np.var(v))
  
        #import asciiplotlib as apl
        #fig = apl.figure()
        #fig.plot([float(f) for f in self.vM.times], [float(f) for f in self.vM], width=100, height=20)
        #fig.show()
        
        #(self.vM.times[-1],c['delay']+c['duration']+200*pq.ms)
        #assert self.vM.times[-1] == (c['delay']+c['duration']+200*pq.ms)

        return self.vM
    #@timer
    '''
    def _backend_run(self):
        results = {}
        #print(self.attrs,'is attributes the empty list?')
        if len(self.attrs) > 1:
            v = get_vm(**self.attrs)
        else:
            v = get_vm(self.attrs)
        #print('deep nothing')
        self.vM = AnalogSignal(v,
                               units = voltage_units,
                               sampling_period = 0.01*pq.ms)
        results['vm'] = self.vM.magnitude
        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results
    '''