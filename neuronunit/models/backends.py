from pyneuroml import pynml
import os
import sciunit
import time
import pdb
import neuronunit.capabilities as cap
import neuronunit.capabilities.spike_functions as sf

class Backend(sciunit.Model):
    # The function (e.g. from pynml) that handles running the simulation
    f = None



class jNeuroMLBackend(Backend):
    f = pynml.run_lems_with_jneuroml

    def set_attrs(self, attrs):
        self.set_lems_attrs(attrs)

    def update_run_params(self, attrs):
        self.update_lems_run_params(attrs)



class NEURONBackend(Backend,
                    cap.ReceivesCurrent,
                    cap.ProducesMembranePotential,
                    cap.ProducesActionPotentials):
    def __init__(self,file_path,model_path,name=None,attrs=None):
        '''
        Inputs: NEURONBackend instance object,file_path,model_path,name=None,attrs=None
        
        Arguably nml_file_path can move out of the constructor signature, and into load_model signature.
        self.neuron is just a place holder for the neuron object attribute. 
        neuron is made an object attribute as common usage of neuron is to mutate its contents
        neuron acts a bit like a global variable object 
        in the scope of this class whose contents are constantly mutated.
        '''
        self.neuron=None
        self.model_path=model_path
        self.nml_file_path=file_path
        self.name=name
        self.attrs=attrs

    def load_model(self):        
        '''
        Inputs: NEURONBackend instance object
        Outputs: nothing mutates input.
        Take a declarative model description, and convert it into an implementation, stored in a pyhoc file.
        import the pyhoc file thus dragging the neuron variables into memory/python name space.
        Since this only happens once outside of the optimization loop its a tolerable performance hit.
        '''
        DEFAULTS={}
        DEFAULTS['v']=True
        #Create a pyhoc file using jneuroml to convert from NeuroML to pyhoc.
        #TODO
        #also check if cached file NeuroML2/LEMS_2007One_nrn.py exists before exec: 
        f=pynml.run_lems_with_jneuroml_neuron(self.nml_file_path, 
                                          skip_run=False,
                                          nogui=False, 
                                          load_saved_data=False, 
                                          only_generate_scripts = True,
                                          plot=False, 
                                          show_plot_already=False, 
                                          exec_in_dir = ".",
                                          verbose=DEFAULTS['v'],
                                          exit_on_fail = True)
        
        #import the contents of the file into the current names space.
        from neuronunit.tests.NeuroML2 import LEMS_2007One_nrn 
        self.neuron=LEMS_2007One_nrn.neuron
        #make sure mechanisms are loaded
        self.neuron.load_mechanisms(self.model_path)  
        pdb.set_trace()  
        #import the default simulation protocol
        from neuronunit.tests.NeuroML2.LEMS_2007One_nrn import NeuronSimulation
        #this next step may be unnecessary: TODO delete it and check.
        self.ns = NeuronSimulation(tstop=1600, dt=0.0025)
        #return NeuronSimulation
        return self

        
    def set_attrs(self,attrs):
        print(attrs)
       
        #TODO find out the python3 syntax for dereferencing key value pairs.
        #Below syntax is stupid, but how to just get key generically without for knowledge of its name and without iterating?
        items=[ (key, value) for key,value in param_dict.items() ]
        for key, value in items:
           print(key, value) 
           #TODO make this solution such that it would work for other single cell models specified by a converted  from neuroml to pyhoc file.
           evalstring='self.neuron.hoc.execute("m_RS_RS_pop[0].'+str(key)+'='+str(value)+'")'
           eval(evalstring)


    def update_run_params(self,attrs):
        print(attrs)
        pdb.set_trace()
       #TODO find out the python3 syntax for dereferencing key value pairs.
        #Below syntax is stupid, but how to just get key generically without for knowledge of its name and without iterating?
        items=[ (key, value) for key,value in stim_dict.items() ]
        for key, value in items:
           print(key, value)
           #TODO make this solution such that it would work for other single cell models specified by a converted from neuroml to pyhoc file.
           evalstring='self.neuron.hoc.execute("explicitInput_RS_IextRS_pop0.'+str(key)+'='+str(value)+'")'
           print(evalstring) 
           eval(evalstring)
        self.neuron.hoc.execute('forall{ psection() }')

        

        
    def get_membrane_potential(self):
        '''
        method required by neuronunit/sciunit
        '''
        '''
        Extracts HOC recording variables from the HOC object namespace,
        converts them to neo analog signal and returns a tuple. Each element in the tuple is of neo
        Analog signal type.
        WARNING probably bug(s). Time conversion not done. Raw time just handed straight to neo without thinking about it.
        implemented conversions from other sources without thinking about it!
        conversion that I have applied here probably doesn't apply
        '''
        from neo.core import AnalogSignal
        import quantities as pq
        import pdb
        t = self.neuron.h.v_time.to_python()   # Convert to Python list for speed...
        v = self.neuron.h.v_v_of0.to_python()   # Convert to Python list for speed, variable has dim: voltage
        current = [ self.neuron.h.v_u_of0.to_python() ]  # Con
        dt = (t[1]-t[0])*pq.s # Time per sample in milliseconds.  
        vm = AnalogSignal(v,units=pq.mV,sampling_rate=1.0/dt)
        #im = AnalogSignal(current,units=pq.nA,sampling_rate=1.0/dt)
        #time = AnalogSignal(t,units=pq.ms,sampling_rate=1.0/dt)
        return vm#(vm,im,time)

    
    def get_spike_train(self):
        import quantities as pq
        mV=pq.mV
        from neuronunit.capabilities import spike_functions
        st=spike_functions.get_spike_train(self.get_membrane_potential(),threshold=0.0*mV)
        pdb.set_trace()
        return spike_functions.get_spike_train(self.get_membrane_potential())
    
    def get_spike_count(self):
        spike_train = self.get_spike_train()
        print(self.get_membrane_potential())
        print(spike_train)
        print('this is the spike count:')
        print(len(spike_train))
        return len(spike_train)
   
    
        

    def inject_square_current(self,current):
        '''                                                                                            
        current : a dictionary like:                                                                                                                                          
        {'amplitude':-10.0*pq.pA,                                                                                                                             
         'delay':100*pq.ms,                                                                                                                                   
         'duration':500*pq.ms}}                                                                                                                               
        where 'pq' is the quantities package        
        '''
        import quantities as pq
        import re
        import pdb
        c=current['injected_square_current']
        print(current['injected_square_current'])
        c['amplitude'] = re.sub('\ pA$', '', str(c['amplitude']))
        c['delay'] = re.sub('\ ms', '', str(c['delay']))
        c['duration'] = re.sub('\ ms$', '', str(c['duration']))
        amps=float(c['amplitude'])/1000.0
        print(amps)
        pdb.set_trace()
        evalstring='self.neuron.hoc.execute("explicitInput_RS_IextRS_pop0.'+str('amplitude')+'='+str(amps)+'")'
        print(evalstring) 
        eval(evalstring)
        evalstring='self.neuron.hoc.execute("explicitInput_RS_IextRS_pop0.'+str('duration')+'='+str(c['delay'])+'")'
        print(evalstring) 
        eval(evalstring)
        evalstring='self.neuron.hoc.execute("explicitInput_RS_IextRS_pop0.'+str('delay')+'='+str(c['duration'])+'")'
        print(evalstring) 
        eval(evalstring)
        self.neuron.hoc.execute('forall{ psection() }')


        
    def local_run(self,duration=1600):
        '''
        Optional argument time duration. 
        Executes the neuron simulation specified in the context of the context of NEURONbackend class
        '''
        
        initialized = True
        sim_start = time.time()
        self.neuron.hoc.execute('tstop='+str(duration))
        self.neuron.hoc.execute('dt=0.0025')
        self.neuron.hoc.tstop=1600
        self.neuron.hoc.dt=0.0025   
        
        print("Running a simulation of %sms (dt = %sms)" % (self.neuron.hoc.tstop, self.neuron.hoc.dt))
        #pdb.set_trace()
        self.neuron.hoc.execute('run()')
        self.neuron.hoc.execute('forall{ psection() }')

        sim_end = time.time()
        sim_time = sim_end - sim_start
        print("Finished NEURON simulation in %f seconds (%f mins)..."%(sim_time, sim_time/60.0))
    
        
    def out_to_neo(self):
        '''
        Extracts HOC recording variables from the HOC object namespace,
        converts them to neo analog signal and returns a tuple. Each element in the tuple is of neo
        Analog signal type.
        WARNING probably bug(s). Time conversion not done. Raw time just handed straight to neo without thinking about it.
        implemented conversions from other sources without thinking about it!
        conversion that I have applied here probably doesn't apply
        '''
        from neo.core import AnalogSignal
        import quantities as pq
        import pdb
        #The pyhoc file generated by pyNeuroML does the conversions below:
        #voltage is scaled by 1000 (to make it millivolts).
        #current is scaled by 1,000,000,000 (to make it nanoAmps)
        #LEMS_2007One_nrn.py
        #t = [ t/1000 for t in self.neuron.h.v_time.to_python() ]  # Convert to Python list for speed...
        #v = [ float(x  / 1000.0) for x in self.neuron.h.v_v_of0.to_python() ]  # Convert to Python list for speed, variable has dim: voltage
        #current = [ float(x  / 1.0E9) for x in self.neuron.h.v_u_of0.to_python() ]  # Convert to Python list for speed, variable has dim: current
        #probably this can be done in the invocation of AnalogSignal below instead.
        t = self.neuron.h.v_time.to_python()   # Convert to Python list for speed...
        v = self.neuron.h.v_v_of0.to_python()   # Convert to Python list for speed, variable has dim: voltage
        current = [ self.neuron.h.v_u_of0.to_python() ]  # Con
        #pdb.set_trace()
        dt = (t[1]-t[0])*pq.s # Time per sample in milliseconds.  
        vm = AnalogSignal(v,units=pq.mV,sampling_rate=1.0/dt)
        im = AnalogSignal(current,units=pq.nA,sampling_rate=1.0/dt)
        time = AnalogSignal(t,units=pq.ms,sampling_rate=1.0/dt)
        return (vm,im,time)
    

    def advance(self):
        '''
        This method might not do anything.
        '''
        #over write/load regular advance

        if not initialized:
            neuron.hoc.finitialize()
            initialized = True

        neuron.hoc.fadvance()
        
