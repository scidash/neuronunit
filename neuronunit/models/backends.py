from pyneuroml import pynml
import os
import sciunit
import time
import pdb
import neuronunit.capabilities as cap
import neuronunit.capabilities.spike_functions as sf
from quantities import ms, mV, nA
from neo.core import AnalogSignal

class Backend:
    """Based class for simulator backends that implement simulator-specific
    details of modifying, running, and reading results from the simulation
    """
    # Name of the backend
    backend = None

    #The function (e.g. from pynml) that handles running the simulation
    f = None

    def set_attrs(self, attrs):
        """Set model attributes, e.g. input resistance of a cell"""
        pass

    def update_run_params(self, attrs):
        """Set run-time parameters, e.g. the somatic current to inject"""
        pass

    def load_model(self):
        """Load the model into memory"""
        pass


class jNeuroMLBackend(Backend):
    """Used for simulation with jNeuroML, a reference simulator for NeuroML"""

    backend = 'jNeuroML'
    f = pynml.run_lems_with_jneuroml

    def set_attrs(self, attrs):
        self.set_lems_attrs(attrs)

    def update_run_params(self, attrs):
        self.update_lems_run_params(attrs)



class NEURONBackend(Backend):
    """Used for simulation with NEURON, a popular simulator
    http://www.neuron.yale.edu/neuron/
    Units used be NEURON are sometimes different to quantities/neo (note nA versus pA)
    http://neurosimlab.org/ramcd/pyhelp/modelspec/programmatic/mechanisms/mech.html#IClamp
    NEURONs units:
    del -- ms
    dur -- ms
    amp -- nA
    i -- nA

    """


    def __init__(self, name=None,attrs=None):
        self.neuron=None
        self.model_path=None
        self.LEMS_file_path=None#LEMS_file_path
        self.name=None
        self.attrs=attrs
        self.f=None
        self.h=None
        self.rheobase=None
        self.invokenrn()
        #self.h.cvode.active(1)
        #pdb.set_trace()
        #self.h.cvode.active


        return
    #make backend a global variable inside this class.
    backend = 'NEURON'

    def invokenrn(self):
        """Sets the NEURON h variable"""
        #Should check if MPI parallel neuron is supported and invoked.
        from neuron import h
        self.h=h
        self.h.load_file("stdlib.hoc")
        self.h.load_file("stdgui.hoc")

    def reset_h(self, hVariable):
        """Sets the NEURON h variable"""

        self.h = hVariable.h
        self.neuron = hVariable


    def setStopTime(self, stopTime = 1000*ms):
        """Sets the simulation duration"""
        """stopTimeMs: duration in milliseconds"""

        tstop = stopTime
        tstop.units = ms
        self.h.tstop = float(tstop)


    def setTimeStep(self, integrationTimeStep = 1/128.0 * ms):
        """Sets the simulation itegration fixed time step"""
        """integrationTimeStepMs: time step in milliseconds. Powers of two preferred. Defaults to 1/128.0"""

        dt = integrationTimeStep
        dt.units = ms

        self.h.dt = self.fixedTimeStep = float(dt)

    def setTolerance(self, tolerance = 0.001):
        """Sets the variable time step integration method absolute tolerance """
        """tolerance: absolute tolerance value"""

        self.h.cvode.atol(tolerance)

    def setIntegrationMethod(self, method = "fixed"):
        """Sets the simulation itegration method"""
        """method: either "fixed" or "variable". Defaults to fixed. cvode is used when "variable" """

        self.h.cvode.active(1 if method == "variable" else 0)


    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal."""
        print('got here')
        if self.h.cvode.active() == 0:
            fixedSignal = self.vVector.to_python()
            dt = self.h.dt

        else:
            fixedSignal = self.get_variable_step_analog_signal()
            dt = self.fixedTimeStep

        return AnalogSignal( \
                 fixedSignal, \
                 units = mV, \
                 sampling_period = dt * ms \
        )

    def get_variable_step_analog_signal(self):
        """ Converts variable dt array values to fixed dt array by using linear interpolation"""

        # Fixed dt potential
        fPots = []
        fDt = self.fixedTimeStep
        # Variable dt potential
        vPots = self.vVector.to_python()
        # Variable dt times
        vTimes = self.tVector.to_python()
        duration = vTimes[len(vTimes)-1]
        # Fixed and Variable dt times
        fTime = vTime = vTimes[0]
        # Index of variable dt time array
        vIndex = 0
        # Advance the fixed dt position
        while fTime <= duration:

            # If v and f times are exact, no interpolation needed
            if fTime == vTime:
                fPots.append(vPots[vIndex])

            # Interpolate between the two nearest vdt times
            else:

                # Increment vdt time until it surpases the fdt time
                while fTime > vTime and vIndex < len(vTimes):
                    vIndex += 1
                    vTime = vTimes[vIndex]

                # Once surpassed, use the new vdt time and t-1 for interpolation
                vIndexMinus1 = max(0, vIndex-1)
                vTimeMinus1 = vTimes[vIndexMinus1]

                fPot = self.linearInterpolate(vTimeMinus1, vTime, \
                                          vPots[vIndexMinus1], vPots[vIndex], \
                                          fTime)

                fPots.append(fPot)

            # Go to the next fdt time step
            fTime += fDt

        return fPots

    def linearInterpolate(self, tStart, tEnd, vStart, vEnd, tTarget):
        tRange = float(tEnd - tStart)
        tFractionAlong = (tTarget - tStart)/tRange

        vRange = vEnd - vStart

        vTarget = vRange*tFractionAlong + vStart

        return vTarget


    def load_model(self):
        '''
        Inputs: NEURONBackend instance object
        Outputs: nothing mutates input object.
        Take a declarative model description, and convert it into an implementation, stored in a pyhoc file.
        import the pyhoc file thus dragging the neuron variables into memory/python name space.
        Since this only happens once outside of the optimization loop its a tolerable performance hit.
        '''
        DEFAULTS={}
        DEFAULTS['v']=True
        #Create a pyhoc file using jneuroml to convert from NeuroML to pyhoc.
        #import the contents of the file into the current names space.
        def cond_load():
            from neuronunit.tests.NeuroML2 import LEMS_2007One_nrn
            self.reset_h(LEMS_2007One_nrn.neuron)
            #make sure mechanisms are loaded
            modeldirname=os.path.dirname(self.orig_lems_file_path)
            self.neuron.load_mechanisms(modeldirname)
            #import the default simulation protocol
            from neuronunit.tests.NeuroML2.LEMS_2007One_nrn import NeuronSimulation
            #this next step may be unnecessary: TODO delete it and check.
            self.ns = NeuronSimulation(tstop=1600, dt=0.0025)
            return self

        if os.path.exists(self.orig_lems_file_path):
            self=cond_load()
        else:
            pynml.run_lems_with_jneuroml_neuron(self.orig_lems_file_path,
                              skip_run=False,
                              nogui=False,
                              load_saved_data=False,
                              only_generate_scripts = True,
                              plot=False,
                              show_plot_already=False,
                              exec_in_dir = ".",
                              verbose=DEFAULTS['v'],
                              exit_on_fail = True)

            self=cond_load()
            more_attributes=pynml.read_lems_file(self.orig_lems_file_path)
            return self
            self.f=pynml.run_lems_with_jneuroml_neuron

        #Although the above approach successfuly instantiates a LEMS/neuroml model in pyhoc
        #the resulting hoc variables for current source and cell name are idiosyncratic (not generic).
        #The resulting idiosyncracies makes it hard not have a hard coded approach make non hard coded, and generalizable code.
        #work around involves predicting the hoc variable names from pyneuroml LEMS file that was used to generate them.
        more_attributes=pynml.read_lems_file(self.orig_lems_file_path)
        for i in more_attributes.components:
        #This code strips out simulation parameters from the xml tree also such as duration.
        #Strip out values from something a bit like an xml tree.
            if str('pulseGenerator') in i.type:
                self.current_src_name=i.id
            if str('Cell') in i.type:
                self.cell_name=i.id
        more_attributes=None#force garbage collection of more_attributes, its not needed anymore.
        return self

    def update_run_params(self,attrs):
        import re
        self.attrs=None
        self.attrs=attrs
        for key, value in self.attrs.items():
             h_variable=list(value.keys())
             h_variable=h_variable[0]
             h_assignment=list(value.values())
             h_assignment=h_assignment[0]
             self.h('m_RS_RS_pop[0].'+str(h_variable)+'='+str(h_assignment))
             self.h('m_'+str(self.cell_name)+'_'+str(self.cell_name)+'_pop[0].'+str(h_variable)+'='+str(h_assignment))
        self.h(' { v_time = new Vector() } ')
        self.h(' { v_time.record(&t) } ')
        self.h(' { v_v_of0 = new Vector() } ')
        self.h(' { v_v_of0.record(&RS_pop[0].v(0.5)) } ')
        self.h(' { v_u_of0 = new Vector() } ')
        self.h(' { v_u_of0.record(&m_RS_RS_pop[0].u) } ')

    def inject_square_current(self,current):
        '''
        Inputs: current : a dictionary
         like:
        {'amplitude':-10.0*pq.pA,
         'delay':100*pq.ms,
         'duration':500*pq.ms}}
        where 'pq' is the quantities package
        '''
        import quantities as pq
        import re
        import copy
        c=copy.copy(current)
        if 'injected_square_current' in c.keys():
            c=current['injected_square_current']

        c['delay'] = re.sub('\ ms$', '', str(c['delay']))
        c['duration'] = re.sub('\ ms$', '', str(c['duration']))
        c['amplitude'] = re.sub('\ pA$', '', str(c['amplitude']))
        #Todo want to convert from nano to pico amps using quantities.
        amps=float(c['amplitude'])/1000.0 #This is the right scale.
        self.h('explicitInput_'+str(self.current_src_name)+str(self.cell_name)+'_pop0.'+str('amplitude')+'='+str(amps))
        self.h('explicitInput_'+str(self.current_src_name)+str(self.cell_name)+'_pop0.'+str('duration')+'='+str(c['duration']))
        self.h('explicitInput_'+str(self.current_src_name)+str(self.cell_name)+'_pop0.'+str('delay')+'='+str(c['delay']))
        self.local_run()

    def local_run(self):
        initialized = True
        sim_start = time.time()
        #self.h.tstop=1600#))#TODO find a way to make duration changeable.
        #self.h.dt=0.0025

        print(self.h.cvode.active())
        #pdb.set_trace()
        print("Running a simulation of %sms (dt = %sms)" % (self.h.tstop, self.h.dt))
        self.h('run()')
        sim_end = time.time()
        sim_time = sim_end - sim_start
        print("Finished NEURON simulation in %f seconds (%f mins)..."%(sim_time, sim_time/60.0))
        self.results={}
        self.results['vm'] = [ float(x/1000.0) for x in self.neuron.h.v_v_of0.to_python() ]  # Convert to Python list for speed, variable has dim: voltage
        self.results['plausible']=True
        import math
        for i in self.results['vm']:
            if math.isnan(i):
                self.results['plausible']=False


        self.results['t'] = [ float(x) for x in self.neuron.h.v_time.to_python() ]  # Convert to Python list for speed, variable has dim: voltage
        self.results['sim_time']=sim_time
        if 'run_number' in self.results.keys():
            self.results['run_number']=self.results['run_number']+1
        else:
            self.results['run_number']=1
        return self.results
