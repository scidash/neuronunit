from pyneuroml import pynml
import os
import sciunit
import time
import pdb
import neuronunit.capabilities as cap
import neuronunit.capabilities.spike_functions as sf

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
                    #cap.ReceivesCurrent,
                    #cap.ProducesMembranePotential,
                    #cap.ProducesActionPotentials):
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

    def __init__(self,name=None,attrs=None):
        # Init should have options to specify model-independent NEURON configuration, e.g. parallel environemnt, 
        # various environment variables, and other options.  
        '''
        Inputs: name=None,attrs=None
        
        Arguably nml_file_path can move out of the constructor signature, and into load_model signature.
        self.neuron is just a place holder for the neuron object attribute. 
        neuron is made an object attribute as common usage of neuron is to mutate its contents
        neuron acts a bit like a global variable object 
        in the scope of this class.
        '''
        self.neuron=None
        self.model_path=None
        self.LEMS_file_path=None#LEMS_file_path
        self.name=name
        self.attrs=attrs
        self.f=None
        
        #pdb.set_trace()

    backend = 'NEURON'

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
            self.neuron=LEMS_2007One_nrn.neuron
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
        #This makes the approach hard to make non hard coded, and generalizable code.
        #work around involves predicting the hoc variable names from LEMS file that was used to generate them.
        more_attributes=pynml.read_lems_file(self.orig_lems_file_path)
        for i in more_attributes.components:
            if str('pulseGenerator') in i.type: 
                self.current_src_name=i.id
            if str('Cell') in i.type: 
                self.cell_name=i.id
        more_attributes=None#garbage collect more_attributes, its not needed anymore.
        return self        
    

    '''
    def set_attrs(self,attrs):

        pdb.set_trace()
        #TODO find out the python3 syntax for dereferencing key value pairs.
        #Below syntax is stupid, but how to just get key generically without for knowledge of its name and without iterating?
        items=[ (key, value) for key,value in attrs.items() ]
        for key, value in items:
           #TODO make this solution such that it would work for other single cell models specified by a converted  from neuroml to pyhoc file.
           self.neuron.hoc.execute('m_'+str(self.cell_name)+'_'+str(self.cell_name)+'_pop[0].'+str(key)+'='+str(value))   
        #print('PSECTION shows model parameters changing:')
        #neuron.hoc.execute('forall{ psection() }')
        #Reset/reinit HOC recording variables 
        #potentially on every run.
        self.neuron.h(' { v_time = new Vector() } ')
        self.neuron.h(' { v_time.record(&t) } ')
        self.neuron.h(' { v_v_of0 = new Vector() } ')
        self.neuron.h(' { v_v_of0.record(&RS_pop[0].v(0.5)) } ')
        self.neuron.h(' { v_u_of0 = new Vector() } ')
        self.neuron.h(' { v_u_of0.record(&m_RS_RS_pop[0].u) } ')
    '''   
     

    def update_run_params(self,attrs):
       
        #TODO find out the python3 syntax for accessing key value pairs.
        #Below syntax is stupid, but how to just get key generically without for knowledge of its name and without iterating?
        #issue is discussed here: https://www.python.org/dev/peps/pep-3106/
        import re
        #items=[ (key, value) for key,value in self.attrs.items() ]
        for key, value in self.attr.items(): 
             h_variable=list(value.keys())
             h_variable=h_variable[0]

             h_assignment=list(value.values())
             h_assignment=h_assignment[0]
             h_assignment = re.sub('\mV$', '', str(h_assignment))


             self.neuron.hoc.execute('m_RS_RS_pop[0].'+str(h_variable)+'='+str(h_assignment))   
             self.neuron.hoc.execute('m_'+str(self.cell_name)+'_'+str(self.cell_name)+'_pop[0].'+str(h_variable)+'='+str(h_assignment))   

        print('PSECTION shows model parameters changing:')
        self.neuron.hoc.execute('forall{ psection() }')
        
     
        self.neuron.h(' { v_time = new Vector() } ')
        self.neuron.h(' { v_time.record(&t) } ')

        self.neuron.h(' { v_v_of0 = new Vector() } ')
        self.neuron.h(' { v_v_of0.record(&RS_pop[0].v(0.5)) } ')

        self.neuron.h(' { v_u_of0 = new Vector() } ')
        self.neuron.h(' { v_u_of0.record(&m_RS_RS_pop[0].u) } ')


        

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


        #Conditionally remake format the dictionary. 
        #For a reason I don't understand the current dictionary is nested side another dictionary whose only key is:
        #'injected_square_current'
        #I think this might have to do with an incomplete transition python2.7->3
        #switching to python3 has meant that updating and accessing dictionary elements is significantly different now.
        #TODO make it so this is no longer necessary.        
   
        import copy   
        c=copy.copy(current)
        if 'injected_square_current' in c.keys():
            c=current['injected_square_current']
            c['amplitude'] = re.sub('\ pA$', '', str(c['amplitude']))
            c['delay'] = re.sub('\ ms$', '', str(c['delay']))
            c['duration'] = re.sub('\ ms$', '', str(c['duration']))


        c['amplitude'] = re.sub('\ pA$', '', str(c['amplitude']))
        c['delay'] = re.sub('\ ms$', '', str(c['delay']))
        c['duration'] = re.sub('\ ms$', '', str(c['duration']))
        amps=float(c['amplitude'])/1000.0 #This is the right scale.
        #'explicitInput_%s%s_pop0.amplitude%s'+=%f' % (self.current_src_name,self.cell_name,amps)
        self.neuron.hoc.execute('explicitInput_'+str(self.current_src_name)+str(self.cell_name)+'_pop0.'+str('amplitude')+'='+str(amps))
        self.neuron.hoc.execute('explicitInput_'+str(self.current_src_name)+str(self.cell_name)+'_pop0.'+str('duration')+'='+str(c['duration']))
        self.neuron.hoc.execute('explicitInput_'+str(self.current_src_name)+str(self.cell_name)+'_pop0.'+str('delay')+'='+str(c['delay']))
 
        self.neuron.hoc.execute('forall{ psection() }')
        self.local_run()


        
    def local_run(self):
        # Maybe this can be called _run.  
        # In some cases 
        '''
        TODO make this comment true again.
        Optional argument time duration. 
        Executes the neuron simulation specified in the context of the context of NEURONbackend class
        '''
        
        initialized = True
        sim_start = time.time()
        self.neuron.hoc.execute('tstop='+str(1600))#TODO find a way to make duration changeable.
        self.neuron.hoc.execute('dt=0.0025')
        self.neuron.hoc.tstop=1600
        self.neuron.hoc.dt=0.0025   
        
        print("Running a simulation of %sms (dt = %sms)" % (self.neuron.hoc.tstop, self.neuron.hoc.dt))
        self.neuron.hoc.execute('run()')
        self.neuron.hoc.execute('forall{ psection() }')

        sim_end = time.time()
        sim_time = sim_end - sim_start
        print("Finished NEURON simulation in %f seconds (%f mins)..."%(sim_time, sim_time/60.0))
        self.results={}
        self.results['vm']=self.neuron.h.v_v_of0.to_python()
        self.results['t']=self.neuron.h.v_time.to_python()
        return self.results
 
