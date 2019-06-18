import sciunit
import quantities as pq
from neo import AnalogSignal
import neuronunit
import numpy as np

def get_zero_crossings_neg2pos(voltage, after_delay=None):
    '''
    Returns the index locations where voltage value crossed 0 from neg->pos direction

    :param voltage: AnalogSignal or numpy array of voltage values
    :return: numpy array of 0-crossing indices
    '''
    if after_delay is not None:
        voltage = voltage.magnitude[np.where(voltage.times >= after_delay.rescale(pq.ms))]

    neg = voltage < 0
    return (neg[:-1] & ~neg[1:]).nonzero()[0]

class NeuronCellModel(sciunit.Model,
                      sciunit.capabilities.Runnable,
                      neuronunit.capabilities.ReceivesSquareCurrent,
                      neuronunit.capabilities.ProducesMembranePotential,
                      neuronunit.capabilities.SupportsVoltageClamp,
                      neuronunit.capabilities.ProducesSpikes,
                      neuronunit.capabilities.SupportsSettingStopTime,
                      neuronunit.capabilities.SupportsSettingTemperature):
    '''
    Defines a NeuronUnit model for running NeuronUnit tests against a
    cell model (1+ sections) implemented in NEURON simulator.

    The class implements methods to inject current and record membrane potential at specified cell segments.

    The class assumes the NEURON model has been loaded, synaptically isolated, and ready for current
    injection experiments. As input, it takes references to the NEURON segments where current is to be injected and
    membrane voltage measured.

    IMPORTANT: When modifying this class, ensure all unit tests pass before checking in your changes to prevent
    breaking of dependent NeuronUnit tests.

    Usage:

    # Load and setup your model in NEURON first
    from neuron import h
    h.load_file('cell.hoc')
    soma = h.Cell[0].soma
    dendrite = h.Cell[0].dend

    # Pass the segment where the current will be injected, and where the membrane potential will be measured
    model = NeuronCellModel(in_seg=soma(0.5), out_seg=dendrite(1.0), name="Smith et. al. (1996) Random Cell")

    # Judge the model
    test.judge(model)
    '''

    default_sampling_period = 1 # ms

    def __init__(self, in_seg, out_seg=None, name=None):

        super(NeuronCellModel, self).__init__()

        self.name = name
        self.in_seg = in_seg
        self.out_seg = out_seg if out_seg else in_seg

        from neuron import h
        self.h = h

        # Set up current and voltage clamps
        self.injector = self.h.IClamp(self.in_seg)
        self.vclamp = self.h.SEClamp(self.in_seg)

        # Setup recorders and clear clamps
        self.reset_instruments()


    def reset_instruments(self):
        '''
        Resets the current and voltage clamps and Vector recorders
        '''

        # Reset current clamp
        self.injector.delay = 0
        self.injector.dur = 0
        self.injector.amp = 0

        # Reset voltage clamp
        self.vclamp.amp1, self.vclamp.amp2, self.vclamp.amp3 = [0,0,0]
        self.vclamp.dur1, self.vclamp.dur2, self.vclamp.dur3 = [0,0,0]
        self.vclamp.rs = 0.001  # MOhm

        # Reset any changes to sampling period
        self.setup_recorders(self.default_sampling_period)

    def setup_recorders(self, sampling_period):
        self.sampling_period = sampling_period

        self.h.steps_per_ms = round(1.0 / self.sampling_period)

        vec_buff_size = 10000*self.h.steps_per_ms

        # Set up recorders for simulation time and membrane voltage
        self.tVector = self.h.Vector()
        self.vVector = self.h.Vector()
        self.vciVector = self.h.Vector()


        # Allocate 10s worth of recording space - to avoid resizing
        self.vVector.buffer_size(vec_buff_size)
        self.tVector.buffer_size(vec_buff_size)
        self.vciVector.buffer_size(vec_buff_size)

        self.vVector.record(self.out_seg._ref_v, self.sampling_period)
        self.tVector.record(self.h._ref_t, self.sampling_period)
        self.vciVector.record(self.vclamp._ref_i, self.sampling_period)



    def get_backend(self):
        return self

    def set_stop_time(self, tstop):
        self.h.tstop = tstop.rescale(pq.ms).magnitude

    def set_temperature(self, celsius):
        self.h.celsius = celsius

    def inject_square_current(self, current = {"delay":0*pq.ms, "duration": 0*pq.ms, "amplitude": 0*pq.nA}, stop_on_spike=False):
        self.reset_instruments()

        # Set the units that NEURON uses
        current["delay"].units = pq.ms
        current["duration"].units = pq.ms
        current["amplitude"].units = pq.nA

        self.injector.delay = float(current["delay"])
        self.injector.dur = float(current["duration"])
        self.injector.amp = float(current["amplitude"])

        if "sampling_period" in current:
            self.setup_recorders(current["sampling_period"])

        if not stop_on_spike:
            self.h.run()
        else:

            self.h.stdinit()
            final_stop = self.h.tstop
            step = 10
            while self.h.t < final_stop - 0.001:
                if self.h.t >= self.injector.delay - 0.001:
                    next_stop = min(self.h.t + step, final_stop)
                else:
                    next_stop = self.injector.delay

                self.h.runStopAt = next_stop
                self.h.continuerun(next_stop)
                self.h.stoprun = 1

                voltage = self.get_membrane_potential()
                aps = get_zero_crossings_neg2pos(voltage, current["delay"])

                if len(aps) > 0:
                    return voltage

        voltage = self.get_membrane_potential()

        return voltage

    def clamp_voltage(self, voltages=[0*pq.mV, 0*pq.mV, 0*pq.mV], durations=[0]*3*pq.ms):

        self.reset_instruments()

        self.vclamp.amp1, self.vclamp.amp2, self.vclamp.amp3 = [float(v) for v in voltages]
        self.vclamp.dur1, self.vclamp.dur2, self.vclamp.dur3 = [float(d) for d in durations]

        self.h.run()

        current = self.nrn_vector_to_AnalogSignal(self.vciVector, pq.nA)


        return current

    def get_membrane_potential(self):
        return self.nrn_vector_to_AnalogSignal(self.vVector, pq.mV)


    def nrn_vector_to_AnalogSignal(self, vector, units):
        '''
        Resample the signal stored by the NEURON vector at the specified steps_per_ms frequency

        :param vector: reference to a NEURON h.Vector()
        :param units: the units to use with the result
        :param steps_per_ms: the number of points to use to represent each ms of the recorded signal
        :return:
        '''
        #t = self.tVector.as_numpy()
        signal = vector.as_numpy()

        # new_t = np.linspace(t[0],t[-1],int(round(steps_per_ms*(t[-1]-t[0]))))
        # new_sig = np.interp(new_t, t, signal)

        return AnalogSignal(signal, sampling_period=self.sampling_period * pq.ms, units=units)



    def __hash__(self):
        hash_tuple = (
            self.name,
            self.h.dt if self.h.cvode_active() == 0 else 0,
            self.default_sampling_period,
            str(self.in_seg),
            str(self.out_seg)
        )

        result = hash(hash_tuple)

        return result
