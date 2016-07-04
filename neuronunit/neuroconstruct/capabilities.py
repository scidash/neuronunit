"""NeuronUnit capability classes implemented using a combination of
NeuroConstruct and neo."""

from __init__ import *
from datetime import datetime
from sciunit import Capability
from neuronunit import *
from neuronunit.capabilities import *
if CPYTHON:
    from pythonnC.utils import putils as utils # From the neuroConstruct pythonnC package.
    import numpy as np
if JYTHON:
    from pythonnC.utils import jutils as utils # From the neuroConstruct pythonnC package.
from pythonnC.utils import neurotools as nc_neurotools
JUTILS_PATH = 'pythonnC.utils.jutils'

class Runnable_NC(Capability):
    """Implementation of Runnable for neuroConstruct."""

    def __init__(self):
        self.ran = False
        self.rerun = False
        self.always_rerun = False
        self.runtime_methods = {}
        self.sim_path = None

    def prepare(self):
        if not hasattr(self,'gateway'):
            if CPYTHON:
                self.gateway = utils.open_gateway(useSocket=True,
                                        automatic_socket=utils.AUTOMATIC_SOCKET)
                cmd = 'import %s as j;' % JUTILS_PATH
                cmd += 'import sys;'
                cmd += 'j.sim = j.Sim(project_path="%s");' % self.project_path
                cmd += 'channel.send(0)'
                channel = self.gateway.remote_exec(cmd)
                channel.receive()
                #self.gateway.terminate()

    def run(self, only_generate=False):
        """Runs the model using jython via execnet and returns a
        directory of simulation results"""
        self.prepare()
        if self.ran is False or self.rerun is True or self.always_rerun is True:
            if only_generate:
                print("Generating simulation files...")
            else:
                print("Running simulation...")
            self.sim_path = utils.run_sim(project_path=self.project_path,
                                          only_generate=only_generate,
                                          useSocket=True,
                                          useNC=True,
                                          useNeuroTools=True,
                                          runtime_methods=self.runtime_methods,
                                          gw=self.gateway)
            self.run_t = datetime.now()
            self.ran = not only_generate
            self.rerun = False
            del self.gateway
        else:
            print("Already ran simulation...")


class ProducesMembranePotential_NC(ProducesMembranePotential,Runnable_NC):
    """An array of somatic membrane potential samples"""

    def get_membrane_potential(self,**kwargs):
        """Returns a neo.core.AnalogSignal object"""

        if self.sim_path is None or self.ran is False \
            or self.rerun or self.always_rerun:
            self.run(**kwargs)
        if self.sim_path == '':
            vm = None
        else:
            #print("Getting membrane potential...")#" from %s/%s" \
            #	  % (self.sim_path,self.population_name))
            vm = nc_neurotools.get_analog_signal(self.sim_path,
                                                 self.population_name)
            # An AnalogSignal instance.
        return vm

    def get_median_vm(self,**kwargs):
        """Returns a quantity corresponding the median membrane potential.
        This will in some cases be the resting potential."""

        vm = self.get_membrane_potential(**kwargs)
        # A neo.core.AnalogSignal object
        median_vm = np.median(vm)
        return median_vm

    def get_initial_vm(self):
        """Returns a quantity corresponding to the starting membrane potential.
        This will in some cases be the resting potential."""
        vm = self.get_membrane_potential()
        # A neo.core.AnalogSignal object
        return vm[0]


class ProducesSpikes_NC(ProducesSpikes,ProducesMembranePotential):
    """Requires ProducesMembranePotential.
    Produces MembranePotentialNC is a logical choice."""

    def get_spike_train(self,**kwargs):
        """Returns a neo.core.SpikeTrain object"""
        vm = self.get_membrane_potential(**kwargs)
        # A neo.core.AnalogSignal object
        return spike_functions.get_spike_train(vm)

    def get_spike_waveforms(self,**kwargs):
        """Returns an neo.core.AnalogSignalArray of spike snippets"""
        vm = self.get_membrane_potential(**kwargs)
        # A neo.core.AnalogSignal object
        return spike_functions.get_spike_waveforms(vm)

    def get_spike_widths(self,**kwargs):
        """Returns an array of spike widths"""
        spikes = self.get_spike_waveforms(**kwargs)
        return spike_functions.spikes2widths(spikes)


class ReceivesCurrent_NC(ReceivesCurrent,Runnable_NC):
    """An array of somatic injected current samples"""

    def __init__(self):
        self.current = self.Current()

    class Current(object):
        ampl = 0
        duration = 0
        offset = 0

    def inject_square_current(self,injected_current):
        self.prepare()
        cmd = 'import %s as j;' % JUTILS_PATH
        cmd += 'import sys;'
        cmd += 'err = j.sim.set_current_ampl(%f);' % \
                    injected_current['amplitude']
        cmd += 'channel.send(err);'
        channel = self.gateway.remote_exec(cmd)
        #print(cmd)
        err = channel.receive() # This will be an error code.
        if len(err):
            raise NotImplementedError(err)
        #self.current.ampl = current_ampl
        #self.runtime_methods['set_current_ampl']=[current_ampl]

