import os
import numpy
import sciunit
from neuronunit.capabilities import ReceivesCurrent,ProducesMembranePotential
from quantities import ms,mV,Hz
from neuron import h

class KaliFreund(sciunit.Model,
                 ReceivesCurrent,
                 ProducesMembranePotential):

    def __init__(self, name="Kali"):
        """ Constructor. """

        modelpath = "./hippounit/models/hoc_models/Kali_Freund_modell/scppinput/"
        libpath = "x86_64/.libs/libnrnmech.so.0"

        if os.path.isfile(modelpath + libpath) is False:
            os.system("cd " + modelpath + "; nrnivmodl")

        h.nrn_load_dll(modelpath + libpath)

        self.name = name
        self.threshold = -20
        self.stim = None
        self.soma = "soma"
        sciunit.Model.__init__(self, name=name)

        self.c_step_start = 0.00004
        self.c_step_stop = 0.000004
        self.c_minmax = numpy.array([0.00004, 0.004])
        self.dend_loc = [[80,0.27],[80,0.83],[54,0.16],[54,0.95],[52,0.38],[52,0.83],[53,0.17],[53,0.7],[28,0.35],[28,0.78]]

        self.AMPA_tau1 = 0.1
        self.AMPA_tau2 = 2
        self.start=150

        self.ns = None
        self.ampa = None
        self.nmda = None
        self.ampa_nc = None
        self.nmda_nc = None

        self.ndend = None
        self.xloc = None

    def translate(self, sectiontype, distance=0):

        if "soma" in sectiontype:
            return self.soma
        else:
            return False

    def initialise(self):
        # load cell
        h.load_file("./hippounit/models/hoc_models/Kali_Freund_modell/scppinput/ca1_syn.hoc")

    def set_cclamp(self, amp):

        self.stim = h.IClamp(h.soma(0.5))
        self.stim.amp = amp
        self.stim.delay = 500
        self.stim.dur = 1000

    def run_cclamp(self):

        print "- running model", self.name

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(h.soma(0.5)._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 1600
        h.run()

        t = numpy.array(rec_t)
        v = numpy.array(rec_v)

        return t, v

    def set_ampa_nmda(self, dend_loc0=[80,0.27]):

        ndend, xloc = dend_loc0
        self.ampa = h.Exp2Syn(xloc, sec=h.dendrite[ndend])
        self.ampa.tau1 = self.AMPA_tau1
        self.ampa.tau2 = self.AMPA_tau2

        self.nmda = h.NMDA_JS(xloc, sec=h.dendrite[ndend])

        self.ndend = ndend
        self.xloc = xloc


    def set_netstim_netcon(self, interval):

        self.ns = h.NetStim()
        self.ns.interval = interval
        self.ns.number = 0
        self.ns.start = self.start

        self.ampa_nc = h.NetCon(self.ns, self.ampa, 0, 0, 0)
        self.nmda_nc = h.NetCon(self.ns, self.nmda, 0, 0, 0)


    def set_num_weight(self, number=1, AMPA_weight=0.0004):

        self.ns.number = number
        self.ampa_nc.weight[0] = AMPA_weight
        self.nmda_nc.weight[0] =AMPA_weight/0.2

    def run_syn(self):

        # initiate recording
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(h.soma(0.5)._ref_v)

        rec_v_dend = h.Vector()
        rec_v_dend.record(h.dendrite[self.ndend](self.xloc)._ref_v)

        print "- running model", self.name
        # initialze and run
        #h.load_file("stdrun.hoc")
        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 300
        h.run()

        # get recordings
        t = numpy.array(rec_t)
        v = numpy.array(rec_v)
        v_dend = numpy.array(rec_v_dend)

        return t, v, v_dend

    def set_cclamp_somatic_feature(self, amp, delay, dur, section_stim, loc_stim):

        exec("self.sect_loc=h." + str(section_stim)+"("+str(loc_stim)+")")


        self.stim = h.IClamp(self.sect_loc)
        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur

    def run_cclamp_somatic_feature(self, section_rec, loc_rec):

        exec("self.sect_loc=h." + str(section_rec)+"("+str(loc_rec)+")")

        print "- running model", self.name

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.sect_loc._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 1600
        h.run()

        t = numpy.array(rec_t)
        v = numpy.array(rec_v)

        return t, v

class Migliore(sciunit.Model,
                 ReceivesCurrent,
                 ProducesMembranePotential):

    def __init__(self, name="Migliore"):
        """ Constructor. """

        modelpath = "./hippounit/models/hoc_models/Migliore_Schizophr/"
        libpath = "x86_64/.libs/libnrnmech.so.0"

        if os.path.isfile(modelpath + libpath) is False:
            os.system("cd " + modelpath + "; nrnivmodl")

        h.nrn_load_dll(modelpath + libpath)

        self.name = "Migliore"
        self.threshold = -20
        self.stim = None
        self.soma = "soma[0]"
        sciunit.Model.__init__(self, name=name)

        self.c_step_start = 0.00004
        self.c_step_stop = 0.000004
        self.c_minmax = numpy.array([0.00004, 0.004])
        self.dend_loc = [[17,0.3],[17,0.9],[24,0.3],[24,0.7],[22,0.3],[22,0.7],[25,0.2],[25,0.5],[30,0.1],[30,0.5]]
        self.trunk_dend_loc_distr=[[10,0.167], [10,0.5], [10,0.83], [11,0.5], [9,0.5], [8,0.5], [7,0.5]]
        self.trunk_dend_loc_clust=[10,0.167]

        self.AMPA_tau1 = 0.1
        self.AMPA_tau2 = 2
        self.start=150

        self.ns = None
        self.ampa = None
        self.nmda = None
        self.ampa_nc = None
        self.nmda_nc = None
        self.ndend = None
        self.xloc = None

    def translate(self, sectiontype, distance=0):

        if "soma" in sectiontype:
            return self.soma
        else:
            return False

    def initialise(self):
        # load cell
        h.load_file("./hippounit/models/hoc_models/Migliore_Schizophr/basic_sim_9068802-test.hoc")


    def set_cclamp(self, amp):

        self.stim = h.IClamp(h.soma[0](0.5))
        self.stim.amp = amp
        self.stim.delay = 500
        self.stim.dur = 1000

    def run_cclamp(self):

        print "- running model", self.name

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(h.soma[0](0.5)._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 1600
        h.run()

        t = numpy.array(rec_t)
        v = numpy.array(rec_v)

        return t, v

    def set_ampa_nmda(self, dend_loc0=[17,0.3]):

        ndend, xloc = dend_loc0
        self.ampa = h.Exp2Syn(xloc, sec=h.apical_dendrite[ndend])
        self.ampa.tau1 = self.AMPA_tau1
        self.ampa.tau2 = self.AMPA_tau2

        self.nmda = h.NMDA_JS(xloc, sec=h.apical_dendrite[ndend])

        self.ndend = ndend
        self.xloc = xloc


    def set_netstim_netcon(self, interval):

        self.ns = h.NetStim()
        self.ns.interval = interval
        self.ns.number = 0
        self.ns.start = self.start

        self.ampa_nc = h.NetCon(self.ns, self.ampa, 0, 0, 0)
        self.nmda_nc = h.NetCon(self.ns, self.nmda, 0, 0, 0)


    def set_num_weight(self, number=1, AMPA_weight=0.0004):

        self.ns.number = number
        self.ampa_nc.weight[0] = AMPA_weight
        self.nmda_nc.weight[0] = AMPA_weight/0.2

    def run_syn(self):

        # initiate recording
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(h.soma[0](0.5)._ref_v)

        rec_v_dend = h.Vector()
        rec_v_dend.record(h.apical_dendrite[self.ndend](self.xloc)._ref_v)

        print "- running model", self.name
        # initialze and run
        #h.load_file("stdrun.hoc")
        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 300
        h.run()

        # get recordings
        t = numpy.array(rec_t)
        v = numpy.array(rec_v)
        v_dend = numpy.array(rec_v_dend)

        return t, v, v_dend

    def set_cclamp_somatic_feature(self, amp, delay, dur, section_stim, loc_stim):

        exec("self.sect_loc=h." + str(section_stim)+"("+str(loc_stim)+")")


        self.stim = h.IClamp(self.sect_loc)
        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur

    def run_cclamp_somatic_feature(self, section_rec, loc_rec):

        exec("self.sect_loc=h." + str(section_rec)+"("+str(loc_rec)+")")

        print "- running model", self.name

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.sect_loc._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 1600
        h.run()

        t = numpy.array(rec_t)
        v = numpy.array(rec_v)

        return t, v

class Bianchi(sciunit.Model,
                 ReceivesCurrent,
                 ProducesMembranePotential):

    def __init__(self, name="Bianchi"):
        """ Constructor. """

        modelpath = "./hippounit/models/hoc_models/Ca1_Bianchi/experiment/"
        libpath = "x86_64/.libs/libnrnmech.so.0"

        if os.path.isfile(modelpath + libpath) is False:
            os.system("cd " + modelpath + "; nrnivmodl")

        h.nrn_load_dll(modelpath + libpath)

        self.name = "Bianchi"
        self.threshold = -20
        self.stim = None
        self.soma = "soma[0]"

        sciunit.Model.__init__(self, name=name)

        self.c_step_start = 0.00004
        self.c_step_stop = 0.000004
        self.c_minmax = numpy.array([0.00004, 0.004])
        self.dend_loc = [[112,0.375],[112,0.875],[118,0.167],[118,0.99],[30,0.167],[30,0.83],[107,0.25],[107,0.75],[114,0.01],[114,0.75]]
        self.trunk_dend_loc_distr=[[65,0.5], [69,0.5], [71,0.5], [64,0.5], [62,0.5], [60,0.5], [81,0.5]]
        self.trunk_dend_loc_clust=[65,0.5]

        self.AMPA_tau1 = 0.1
        self.AMPA_tau2 = 2
        self.start=150

        self.ns = None
        self.ampa = None
        self.nmda = None
        self.ampa_nc = None
        self.nmda_nc = None
        self.ndend = None
        self.xloc = None


    def translate(self, sectiontype, distance=0):

        if "soma" in sectiontype:
            return self.soma
        else:
            return False


    def initialise(self):
        # load cell
        h.load_file("./hippounit/models/hoc_models/Ca1_Bianchi/experiment/basic.hoc")


    def set_cclamp(self, amp):

        self.stim = h.IClamp(h.soma[0](0.5))
        self.stim.amp = amp
        self.stim.delay = 500
        self.stim.dur = 1000

    def run_cclamp(self):

        print "- running model", self.name

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(h.soma[0](0.5)._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 1600
        h.run()

        t = numpy.array(rec_t)
        v = numpy.array(rec_v)

        return t, v

    def set_ampa_nmda(self, dend_loc0=[112,0.375]):

        ndend, xloc = dend_loc0
        self.ampa = h.Exp2Syn(xloc, sec=h.apical_dendrite[ndend])
        self.ampa.tau1 = self.AMPA_tau1
        self.ampa.tau2 = self.AMPA_tau2

        self.nmda = h.NMDA_JS(xloc, sec=h.apical_dendrite[ndend])

        self.ndend = ndend
        self.xloc = xloc


    def set_netstim_netcon(self, interval):

        self.ns = h.NetStim()
        self.ns.interval = interval
        self.ns.number = 0
        self.ns.start = self.start

        self.ampa_nc = h.NetCon(self.ns, self.ampa, 0, 0, 0)
        self.nmda_nc = h.NetCon(self.ns, self.nmda, 0, 0, 0)


    def set_num_weight(self, number=1, AMPA_weight=0.0004):

        self.ns.number = number
        self.ampa_nc.weight[0] = AMPA_weight
        self.nmda_nc.weight[0] = AMPA_weight/0.2

    def run_syn(self):

        # initiate recording
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(h.soma[0](0.5)._ref_v)

        rec_v_dend = h.Vector()
        rec_v_dend.record(h.apical_dendrite[self.ndend](self.xloc)._ref_v)

        print "- running model", self.name
        # initialze and run
        #h.load_file("stdrun.hoc")
        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 300
        h.run()

        # get recordings
        t = numpy.array(rec_t)
        v = numpy.array(rec_v)
        v_dend = numpy.array(rec_v_dend)

        return t, v, v_dend

    def set_cclamp_somatic_feature(self, amp, delay, dur, section_stim, loc_stim):

        exec("self.sect_loc=h." + str(section_stim)+"("+str(loc_stim)+")")


        self.stim = h.IClamp(self.sect_loc)
        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur

    def run_cclamp_somatic_feature(self, section_rec, loc_rec):

        exec("self.sect_loc=h." + str(section_rec)+"("+str(loc_rec)+")")

        print "- running model", self.name

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.sect_loc._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 1600
        h.run()

        t = numpy.array(rec_t)
        v = numpy.array(rec_v)

        return t, v

class Golding(sciunit.Model,
                 ReceivesCurrent,
                 ProducesMembranePotential):

    def __init__(self, name="Golding"):
        """ Constructor. """

        modelpath = "./hippounit/models/hoc_models/Golding_dichotomy/fig08/"
        libpath = "x86_64/.libs/libnrnmech.so.0"

        if os.path.isfile(modelpath + libpath) is False:
            os.system("cd " + modelpath + "; nrnivmodl")

        h.nrn_load_dll(modelpath + libpath)

        #self.dendA5_01111111100 = h.Section(name='foo', cell=self)

        self.dendrite=None
        self.name = "Golding"
        self.threshold = -40
        self.stim = None
        self.soma = "somaA"

        sciunit.Model.__init__(self, name=name)

        self.c_step_start = 0.00004
        self.c_step_stop = 0.000004
        self.c_minmax = numpy.array([0.00004, 0.004])
        self.dend_loc = [["dendA5_00",0.275],["dendA5_00",0.925],["dendA5_01100",0.15],["dendA5_01100",0.76],["dendA5_0111100",0.115],["dendA5_0111100",0.96],["dendA5_01111100",0.38],["dendA5_01111100",0.98],["dendA5_0111101",0.06],["dendA5_0111101",0.937]]
        self.trunk_dend_loc_distr=[["dendA5_01111111111111",0.68], ["dendA5_01111111111111",0.136], ["dendA5_01111111111111",0.864], ["dendA5_011111111111111",0.5], ["dendA5_0111111111111111",0.5], ["dendA5_0111111111111",0.786], ["dendA5_0111111111111",0.5]]
        self.trunk_dend_loc_clust=["dendA5_01111111111111",0.68]

        self.AMPA_tau1 = 0.1
        self.AMPA_tau2 = 2
        self.start=150

        self.ns = None
        self.ampa = None
        self.nmda = None
        self.ampa_nc = None
        self.nmda_nc = None
        self.ndend = None
        self.xloc = None

    def translate(self, sectiontype, distance=0):

        if "soma" in sectiontype:
            return self.soma
        else:
            return False

    def initialise(self):
        # load cell
        h.load_file("./hippounit/models/hoc_models/Golding_dichotomy/fig08/run_basic.hoc")

    def set_cclamp(self, amp):

        self.stim = h.IClamp(h.somaA(0.5))
        self.stim.amp = amp
        self.stim.delay = 500
        self.stim.dur = 1000

    def run_cclamp(self):

        print "- running model", self.name

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(h.somaA(0.5)._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 1600
        h.run()

        t = numpy.array(rec_t)
        v = numpy.array(rec_v)

        return t, v

    def set_ampa_nmda(self, dend_loc0=["dendA5_01111111100",0.375]):
        #self.dendrite=h.dendA5_01111111100

        ndend, xloc = dend_loc0

        exec("self.dendrite=h." + ndend)

        self.ampa = h.Exp2Syn(xloc, sec=self.dendrite)
        self.ampa.tau1 = self.AMPA_tau1
        self.ampa.tau2 = self.AMPA_tau2

        self.nmda = h.NMDA_JS(xloc, sec=self.dendrite)

        self.ndend = ndend
        self.xloc = xloc


    def set_netstim_netcon(self, interval):

        self.ns = h.NetStim()
        self.ns.interval = interval
        self.ns.number = 0
        self.ns.start = self.start

        self.ampa_nc = h.NetCon(self.ns, self.ampa, 0, 0, 0)
        self.nmda_nc = h.NetCon(self.ns, self.nmda, 0, 0, 0)


    def set_num_weight(self, number=1, AMPA_weight=0.0004):

        self.ns.number = number
        self.ampa_nc.weight[0] = AMPA_weight
        self.nmda_nc.weight[0] = AMPA_weight/0.2

    def run_syn(self):

        # initiate recording
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(h.somaA(0.5)._ref_v)

        rec_v_dend = h.Vector()
        rec_v_dend.record(self.dendrite(self.xloc)._ref_v)

        print "- running model", self.name
        # initialze and run
        #h.load_file("stdrun.hoc")
        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 300
        h.run()

        # get recordings
        t = numpy.array(rec_t)
        v = numpy.array(rec_v)
        v_dend = numpy.array(rec_v_dend)

        return t, v, v_dend

    def set_cclamp_somatic_feature(self, amp, delay, dur, section_stim, loc_stim):

        exec("self.sect_loc=h." + str(section_stim)+"("+str(loc_stim)+")")


        self.stim = h.IClamp(self.sect_loc)
        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur

    def run_cclamp_somatic_feature(self, section_rec, loc_rec):

        exec("self.sect_loc=h." + str(section_rec)+"("+str(loc_rec)+")")

        print "- running model", self.name

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.sect_loc._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = -65

        h.celsius = 34
        h.init()
        h.tstop = 1600
        h.run()

        t = numpy.array(rec_t)
        v = numpy.array(rec_v)

        return t, v
