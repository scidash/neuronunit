
from pyneuroml import pynml
import os
import platform
import sciunit
import time
import pdb
import neuronunit.capabilities as cap
import neuronunit.capabilities.spike_functions as sf
import re
import copy

from pyneuroml import pynml
import quantities as pq
from quantities import ms, mV, nA
from neo.core import AnalogSignal

import quantities as pq
import re
import copy
import neuronunit.capabilities as cap
import neuronunit.capabilities.spike_functions as sf

class Backend:
    """Base class for simulator backends that implement simulator-specific
    details of modifying, running, and reading results from the simulation
    """
    # Name of the backend
    backend = None

    #The function (e.g. from pynml) that handles running the simulation
    f = None

    def set_attrs(self, attrs):
        """Set model attributes, e.g. input resistance of a cell"""
        pass

    def update_run_params(self):
        """Set run-time parameters, e.g. the somatic current to inject"""
        pass

    def load_model(self):
        """Load the model into memory"""
        pass


class jNeuroMLBackend(Backend):
    """Used for simulation with jNeuroML, a reference simulator for NeuroML"""

    backend = 'jNeuroML'

    def set_attrs(self, attrs):
        self.set_lems_attrs(attrs)

    def update_run_params(self):
        self.update_lems_run_params()

    def inject_square_current(self, current):
        self.run_params['injected_square_current'] = current

    def local_run(self):
        f = pynml.run_lems_with_jneuroml
        result = f(self.lems_file_path, skip_run=self.skip_run,
                         nogui=self.run_params['nogui'],
                         load_saved_data=True, plot=False,
                         verbose=self.run_params['v']
                         )
        return result


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
        self.cell_name=name
        self.model_path=None
        self.LEMS_file_path=None#LEMS_file_path
        self.name=None
        self.f=None
        self.rheobase_memory=None
        self.invokenrn()



        return
    #make backend a global variable inside this class.
    backend = 'NEURON'

    def invokenrn(self):
        """Sets the NEURON h variable"""
        #Should check if MPI parallel neuron is supported and invoked.
        from neuron import h
        import neuron
        self.neuron = neuron
        self.h=h
        self.h.load_file("stdlib.hoc")
        self.h.load_file("stdgui.hoc")


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
            self.invokenrn()
            modeldirname=os.path.dirname(self.orig_lems_file_path)
            print(modeldirname, 'name ')
            previousdir=os.getcwd()
            os.chdir(modeldirname)
            exec_string=str('nrnivmodl ')+str(modeldirname)#+str('')
            os.system(exec_string)
            self.neuron.load_mechanisms(str(modeldirname)+str('/NeuroML2/x86_64'))
            os.chdir(previousdir)
            #self.neuron.load_mechanisms(modeldirname)
            from neuronunit.tests.NeuroML2.LEMS_2007One_nrn import NeuronSimulation
            self.ns = NeuronSimulation(tstop=1600, dt=0.0025)
            return self


        architecture = platform.machine()
        LEMS_dir  = os.path.dirname(self.orig_lems_file_path) # Gets full path to directory with file.
        NEURON_file = os.path.join(LEMS_dir,architecture)


        if os.path.exists(NEURON_file):
            self = cond_load()

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
                print(self.cell_name)

        more_attributes=None#force garbage collection of more_attributes, its not needed anymore.
        return self

    def update_run_params(self,attrs):
        import re
        self.attrs=None
        self.attrs=attrs
        paramdict={}
        for v in self.attrs.values():
             paramdict = v

        for key, value in paramdict.items():
             h_variable=key
             h_assignment=value
             self.h('m_RS_RS_pop[0].'+str(h_variable)+'='+str(h_assignment))
             self.h('m_'+str(self.cell_name)+'_'+str(self.cell_name)+'_pop[0].'+str(h_variable)+'='+str(h_assignment))

        self.h(' { v_time = new Vector() } ')
        self.h(' { v_time.record(&t) } ')
        self.h(' { v_v_of0 = new Vector() } ')
        self.h(' { v_v_of0.record(&RS_pop[0].v(0.5)) } ')
        self.h(' { v_u_of0 = new Vector() } ')
        self.h(' { v_u_of0.record(&m_RS_RS_pop[0].u) } ')



    def re_init(self,attrs):
        self.load_model()
        self.update_run_params(attrs)
        #print(attrs)
        #self.h.psection()


    def inject_square_current(self,current):
        '''
        Inputs: current : a dictionary
         like:
        {'amplitude':-10.0*pq.pA,
         'delay':100*pq.ms,
         'duration':500*pq.ms}}
        where 'pq' is the quantities package
        '''
        self.re_init(self.attrs)
        #print(self.attrs)
        #self.h.psection()
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
        #print(self.h.cvode.active())
        self.h('run()')
        sim_end = time.time()
        sim_time = sim_end - sim_start
        #print("Finished NEURON simulation in %f seconds (%f mins)..."%(sim_time, sim_time/60.0))
        self.results={}
        import copy
        voltage_vector=self.neuron.h.v_v_of0.to_python()
        self.results['vm'] = [ float(x/1000.0) for x in copy.copy(voltage_vector) ]  # Convert to Python list for speed, variable has dim: voltage
        voltage_vector=None
        #self.neuron.h.v_v_of0 = [] # Convert to Python list for speed, variable has dim: voltage
        time_vector=self.neuron.h.v_time.to_python()
        self.results['t'] = [ float(x) for x in copy.copy(time_vector) ]  # Convert to Python list for speed, variable has dim: voltage
        time_vector=None
        #self.neuron.h.v_time = []
        if 'run_number' in self.results.keys():
            self.results['run_number']=self.results['run_number']+1
        else:
            self.results['run_number']=1
        return self.results
