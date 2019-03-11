import io
import math
import pdb


from sciunit.utils import redirect_stdout
from .base import os, copy, subprocess
from .base import pq, AnalogSignal, NEURON_SUPPORT, neuron, h, pynml
from .base import Backend, BackendException, import_module_from_path


class NEURONBackend(Backend):
    """Use for simulation with NEURON, a popular simulator.

    http://www.neuron.yale.edu/neuron/
    Units used by NEURON are sometimes different to quantities/neo
    (note nA versus pA)
    http://neurosimlab.org/ramcd/pyhelp/modelspec/programmatic/mechanisms/mech.html#IClamp
    NEURON's units:
    del -- ms
    dur -- ms
    amp -- nA
    i -- nA
    """

    def init_backend(self, attrs=None, cell_name=None, current_src_name=None,
                     DTC=None):
        """Initialize the NEURON backend for neuronunit.

        Arguments should be consistent with an underlying model files.

        Args:
            attrs (dict): a dictionary of items used to update NEURON
                          model attributes.
            cell_name (string): A string that represents the cell models name
                                in the NEURON HOC space.
            current_src_name (string): A string that represents the current
                                       source models name in the NEURON HOC
                                       space.
            DTC (DataTransportContainer): The data transport container contain
                                          a dictionary of model attributes
                                          When the DTC object is provided,
                                          it's attribute dictionary can be used
                                          to update the NEURONBackends model
                                          attribute dictionary.
        """
        if not NEURON_SUPPORT:
            msg = "The neuron module was not successfully imported"
            raise BackendException(msg)

        self.stdout = io.StringIO()
        self.neuron = None
        self.model_path = None
        self.h = h

        backend = 'NEURON'

        super(NEURONBackend, self).init_backend()
        self.model._backend.use_memory_cache = False
        self.model.unpicklable += ['h', 'ns', '_backend']

        if type(DTC) is not type(None):
            if type(DTC.attrs) is not type(None):

                self.set_attrs(**DTC.attrs)
                if len(DTC.attrs):
                    assert len(self.model.attrs) > 0

            if hasattr(DTC, 'current_src_name'):
                self._current_src_name = DTC.current_src_name

            if hasattr(DTC, 'cell_name'):
                self._cell_name = DTC.cell_name

    def reset_neuron(self, neuronVar):
        """Reset the neuron simulation.

        Refreshes the the HOC module, purging it's variable namespace.
        Sets the NEURON h variable, and resets the NEURON h variable.
        The NEURON h variable, may benefit from being reset between simulation
        runs as a way of insuring that each simulation is freshly initialized.
        the reset_neuron method is used to prevent a situation where a new
        model's initial conditions are erroneously updated from a stale model's
        final state.

        Args:
            neuronVar (module): a reference to the neuron module
        """
        self.h = neuronVar.h
        self.neuron = neuronVar

    def set_run_params(self, **run_params):
        pass

    def set_stop_time(self, stop_time=650*pq.ms):
        """Set the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.h.tstop = float(stop_time.rescale(pq.ms))

    def set_time_step(self, integrationTimeStep=(pq.ms/128.0)):
        """Set the simulation itegration fixed time step
        integrationTimeStepMs: time step in milliseconds.
        Powers of two preferred. Defaults to 1/128.0

        Args:
            integrationTimeStep (float): time step in milliseconds.
                Powers of two preferred. Defaults to 1/128.0
        """
        dt = integrationTimeStep
        self.h.dt = float(dt)

    def set_tolerance(self, tolerance=0.001):
        """Set the variable time step integration method absolute tolerance.

        Args:
            tolerance (float): absolute tolerance value
        """
        self.h.cvode.atol(tolerance)

    def set_integration_method(self, method="fixed"):
        """Set the simulation itegration method.

        cvode is used when method is "variable"

        Args:
            method (string): either "fixed" or "variable". Defaults to fixed.
        """
        # This line is compatible with the above cvodes statements.
        self.h.cvode.active(1 if method == "variable" else 0)

        try:
            assert self.cvode.active()
        except AssertionError:
            self.cvode = self.h.CVode()
            self.cvode.active(1 if method == "variable" else 0)

    def get_membrane_potential(self):
        """Get a membrane potential traces from the simulation.

        Must destroy the hoc vectors that comprise it.

        Returns:
            neo.core.AnalogSignal: the membrane potential trace
        """
        if self.h.cvode.active() == 0:
            dt = float(copy.copy(self.h.dt))
            fixed_signal = copy.copy(self.vVector.to_python())
        else:
            dt = float(copy.copy(self.fixedTimeStep))
            fixed_signal = copy.copy(self.get_variable_step_analog_signal())

        self.h.dt = dt
        self.fixedTimeStep = float(1.0/dt)
        return AnalogSignal(fixed_signal,
                            units=pq.mV,
                            sampling_period=dt*pq.ms)

    def get_variable_step_analog_signal(self):
        """Convert variable dt array values to fixed dt array.

        Uses linear interpolation.
        """
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

                # Once surpassed, use the new vdt time and t-1
                # for interpolation
                vIndexMinus1 = max(0, vIndex-1)
                vTimeMinus1 = vTimes[vIndexMinus1]

                fPot = self.linearInterpolate(vTimeMinus1, vTime,
                                              vPots[vIndexMinus1],
                                              vPots[vIndex], fTime)

                fPots.append(fPot)

            # Go to the next fdt time step
            fTime += fDt

        return fPots

    def linearInterpolate(self, tStart, tEnd, vStart, vEnd, tTarget):
        """Perform linear interpolation."""
        tRange = float(tEnd - tStart)
        tFractionAlong = (tTarget - tStart)/tRange
        vRange = vEnd - vStart
        vTarget = vRange*tFractionAlong + vStart

        return vTarget

    def load(self, tstop=650*pq.ms):
        nrn_path = (os.path.splitext(self.model.orig_lems_file_path)[0] +
                    '_nrn.py')
        nrn = import_module_from_path(nrn_path)
        self.reset_neuron(nrn.neuron)
        self.h.tstop = tstop
        self.set_stop_time(tstop)  # previously 500ms add on 150ms of recovery
        with redirect_stdout(self.stdout):
            self.ns = nrn.NeuronSimulation(self.h.tstop, dt=0.0025)

    def load_mechanisms(self):
        with redirect_stdout(self.stdout):
            neuron.load_mechanisms(self.neuron_model_dir)

    def load_model(self, verbose=True):
        """Load a NEURON model.

        Side Effects: Substantially mutates neuronal model stored in self.
        Description: Take a declarative model description, and call JneuroML
        to convert it into an python/neuron implementation stored in a pyho
        file. Then import the pyhoc file thus dragging the neuron variables
        into memory/python name space. Since this only happens once outside
        of the optimization loop its a tolerable performance hit.

        Create a pyhoc file using jneuroml to convert from NeuroML to pyhoc.
        import the contents of the file into the current names space.
        """
        assert os.path.isfile(self.model.orig_lems_file_path)
        base_name = os.path.splitext(self.model.orig_lems_file_path)[0]
        NEURON_file_path = '{0}_nrn.py'.format(base_name)
        self.neuron_model_dir = os.path.dirname(self.model.orig_lems_file_path)
        assert os.path.isdir(self.neuron_model_dir)
        if not os.path.exists(NEURON_file_path):
            pynml.run_lems_with_jneuroml_neuron(
                self.model.orig_lems_file_path,
                skip_run=False,
                nogui=True,
                load_saved_data=False,
                only_generate_scripts=True,
                plot=False,
                show_plot_already=False,
                exec_in_dir=self.neuron_model_dir,
                verbose=verbose,
                exit_on_fail=True)
            # use a different process to call NEURONS compiler nrnivmodl in the
            # background if the NEURON_file_path does not yet exist.
            subprocess.run(["cd %s; nrnivmodl" % self.neuron_model_dir],
                           shell=True)
            self.load_mechanisms()
        elif os.path.realpath(os.getcwd()) != \
                os.path.realpath(self.neuron_model_dir):
            # Load mechanisms unless they've already been loaded
            #       subprocess.run(["cd %s; nrnivmodl" % self.neuron_model_dir],shell=True)
            self.load_mechanisms()
            self.load()

        # Although the above approach successfuly instantiates a LEMS/neuroml model in pyhoc
        # the resulting hoc variables for current source and cell name are idiosyncratic (not generic).
        # the non generic approach described above makes it hard to create a generalizable code.
        # work around involves predicting the hoc variable names from pyneuroml LEMS file that was used to generate them.
        if not hasattr(self,'_current_src_name') or not hasattr(self,'_cell_name'):
            more_attributes = pynml.read_lems_file(self.model.orig_lems_file_path, include_includes=True,
                                                   debug=False)
            for i in more_attributes.components:
                # This code strips out simulation parameters from the xml tree also such as current source name.
                # and cell_name
                if str('pulseGenerator') in i.type:
                    self._current_src_name = i.id
                if str('Cell') in i.type:
                    self._cell_name = i.id
            more_attributes = None  # explicitly perform garbage collection on
                                    # more_attributes since its not needed
                                    # anymore.
        return self

    @property
    def cell_name(self):
        """Get the name of the cell."""
        return getattr(self, '_cell_name', 'RS')

    @property
    def current_src_name(self):
        """Get the name of the current source."""
        return getattr(self, '_current_src_name', 'RS')

    def reset_vm(self):
        ass_vr = self.h.m_RS_RS_pop[0].vr
        self.h.m_RS_RS_pop[0].v0 = ass_vr
        self.h.m_RS_RS_pop[0].u = ass_vr


        self.h('m_{0}_{1}_pop[0].{2} = {3}'\
                .format(self.cell_name,self.cell_name,'v0',ass_vr))


    def set_attrs(self, **attrs):
        self.model.attrs = {}
        self.model.attrs.update(attrs)
        for h_key, h_value in attrs.items():
            self.h('m_{0}_{1}_pop[0].{2} = {3}'
                   .format(self.cell_name, self.cell_name, h_key, h_value))


        for h_key,h_value in attrs.items():
            h_value = float(h_value)
            if h_key is str('C'):
                sec = self.h.Section(self.h.m_RS_RS_pop[0])
                #sec.L, sec.diam = 6.3, 5 # empirically tuned
                sec.cm = h_value
            else:
                self.h('m_{0}_{1}_pop[0].{2} = {3}'.format(self.cell_name,self.cell_name,h_key,h_value))

        ass_vr = self.h.m_RS_RS_pop[0].vr
        self.h.m_RS_RS_pop[0].v0 = ass_vr

        #print(self.model.attrs)
        # Below create a list of NEURON experimental recording rig parameters.
        # This is where parameters of the artificial neuron experiment are
        # initiated.
        # Code is sent to the python interface to neuron by executing strings:
        neuron_sim_rig = []
        neuron_sim_rig.append(' { v_time = new Vector() } ')
        neuron_sim_rig.append(' { v_time.record(&t) } ')
        neuron_sim_rig.append(' { v_v_of0 = new Vector() } ')
        neuron_sim_rig.append(' { v_v_of0.record(&RS_pop[0].v(0.5)) } ')
        neuron_sim_rig.append(' { v_u_of0 = new Vector() } ')
        neuron_sim_rig.append(' { v_u_of0.record(&m_RS_RS_pop[0].u) } ')

        for string in neuron_sim_rig:
            # execute hoc code strings in the python interface to neuron.
            self.h(string)

        # These two variables have been aliased in the code below:
        self.tVector = self.h.v_time
        self.vVector = self.h.v_v_of0
        return self

    def inject_square_current(self, current, section=None, debug=False):
        """Apply current injection into the soma or a specific compartment.

        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms,
                            'duration':float*pq.ms}}
        where 'pq' is a physical unit representation, implemented by casting
        float values to the quanitities 'type'.
        Currently only single section neuronal models are supported, the
        neurite section is understood to be simply the soma.

        Args:
            current (dict): a dictionary with exactly three items,
              whose keys are: 'amplitude', 'delay', 'duration'

        Implementation:
        1. purge the HOC space, by calling reset_neuron()
        2. Redefine the neuronal model in the HOC namespace, which was recently
           cleared out.
        3. Strip away quantities representation of physical units.
        4. Translate the dictionary of current injection parameters into
           executable HOC code.
        """
        self.h = None
        self.neuron = None

        nrn_path = os.path.splitext(self.model.orig_lems_file_path)[0]+'_nrn.py'
        nrn = import_module_from_path(nrn_path)

        ##
        # init_backend is the most reliable way to purge existing NEURON simulations.
        # however we don't want to purge the model attributes, we only want to purge
        # the NEURON model code.
        # store the model attributes, in a temp buffer such that they persist throughout the model reinitialization.
        ##
        # These lines are crucial.
        temp_attrs = copy.copy(self.model.attrs)
        self.init_backend()
        self.set_attrs(**temp_attrs)

        current = copy.copy(current)
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        ##
        # critical code:
        ##
        self.set_stop_time(c['delay']+c['duration']+100.0*pq.ms)
        # translate pico amps to nano amps
        # NEURONs default unit multiplier for current injection values is nano amps.
        # to make sure that pico amps are not erroneously interpreted as a larger nano amp.
        # current injection value, the value is divided by 1000.
        #amps = float(c['amplitude'].rescale('nA')) #float(c['amplitude'])#/1000.0# #This is the right scale.
        amps = float(c['amplitude']/1000.0)
        prefix = 'explicitInput_%s%s_pop0.' % (self.current_src_name,self.cell_name)
        define_current = []
        define_current.append('{0}amplitude={1}'.format(prefix,amps))
        duration = float(c['duration'])#.rescale('ms'))
        delay = float(c['delay'])#.rescale('ms'))
        define_current.append('{0}duration={1}'.format(prefix,duration))
        define_current.append('{0}delay={1}'.format(prefix,delay))


        for string in define_current:
            # execute hoc code strings in the python interface to neuron.
            self.h(string)
        self.neuron.h.psection()
        if debug == True:
            self.neuron.h.psection()
        self._backend_run()

    def _backend_run(self):
        self.h('run()')
        results = {}
        # Prepare NEURON vectors for quantities/sciunit
        # By rescaling voltage to milli volts, and time to milli seconds.
        results['vm'] = [float(x/1000.0) for x in
                         copy.copy(self.neuron.h.v_v_of0.to_python())]
        results['t'] = [float(x/1000.0) for x in
                        copy.copy(self.neuron.h.v_time.to_python())]
        results['run_number'] = results.get('run_number', 0) + 1

        return results
