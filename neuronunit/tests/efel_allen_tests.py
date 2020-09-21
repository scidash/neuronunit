
from neuronunit.tests.base import np, pq, ncap, VmTest, scores, AMPL#, DELAY, DURATION
import quantities as qt
from neuronunit.tests.target_spike_current import SpikeCountSearch, SpikeCountRangeSearch

import efel

from sciunit import scores

class AdaptionTest(VmTest):
    """Tests the input resistance of a cell.
    """
    score_type = scores.RatioScore
    def __init__(self,observation=None,observed_spk_cnt=None):
        self = self
        self.obervation = observation
        self.observed_spk_cnt = observed_spk_cnt
        scs = SpikeCountSearch(self.observed_spk_cnt)
        self.model = dtc.dtc_to_model()
        self.ampl = scs.generate_prediction(model)

    def generate_prediction(model):
        params = {'injected_square_current':
            {'amplitude':self.ampl, 'delay':100*pq.ms, 'duration':1000*pq.ms}}
        vm_used = self.model.inject_square_current(params)
        efel.setThreshold(0)

        trace3 = {'T': [float(t)*1000.0 for t in vm_used.times],
                  'V': [float(v) for v in vm_used.magnitude],
                  'stimulus_current': [current]}
        ALLEN_DURATION = 2000*qt.ms
        ALLEN_DELAY = 1000*qt.ms
        trace3['stim_end'] = [ float(ALLEN_DELAY)+float(ALLEN_DURATION) ]
        trace3['stim_start'] = [ float(ALLEN_DELAY)]
        results = efel.getMeanFeatureValues([trace3],['adaptation_index','adaptation_index2'])
        self.prediction = results
        return results
    def compute_score(self):
        if self.prediction is None:
            return None
        else:
            score = super(VmTest, self).compute_score(self.observation,
                                                                    self.prediction)
        score.related_data['vm'] = self.vm
        return score


class BaseVMTest(VmTest):
    score_type = scores.RatioScore

    def __init__(self,observation=None,observed_spk_cnt=None):
        self = self
        self.obervation = observation
        self.observed_spk_cnt = observed_spk_cnt
        scs = SpikeCountSearch(self.observed_spk_cnt)
        self.model = dtc.dtc_to_model()
        self.ampl = scs.generate_prediction(model)

    def generate_prediction(model):
        params = {'injected_square_current':
            {'amplitude':self.ampl, 'delay':100*pq.ms, 'duration':1000*pq.ms}}
        vm_used = self.model.inject_square_current(params)
        efel.setThreshold(0)

        trace3 = {'T': [float(t)*1000.0 for t in vm_used.times],
                  'V': [float(v) for v in vm_used.magnitude],
                  'stimulus_current': [current]}
        ALLEN_DURATION = 2000*qt.ms
        ALLEN_DELAY = 1000*qt.ms

        trace3['stim_end'] = [ float(ALLEN_DELAY)+float(ALLEN_DURATION) ]
        trace3['stim_start'] = [ float(ALLEN_DELAY)]
        results = efel.getMeanFeatureValues([trace3],['voltage_base'])
        self.prediction = results
        return results
    def compute_score(self):
        if self.prediction is None:
            return None
        else:
            score = super(VmTest, self).compute_score(self.observation,
                                                                    self.prediction)
        score.related_data['vm'] = self.vm
        return score

class SpikeHeightVMTest(VmTest):
    score_type = scores.RatioScore

    def __init__(self,observation=None,observed_spk_cnt=None):
        self = self
        self.obervation = observation
        self.observed_spk_cnt = observed_spk_cnt
        scs = SpikeCountSearch(self.observed_spk_cnt)
        self.model = dtc.dtc_to_model()
        self.ampl = scs.generate_prediction(model)

    def generate_prediction(model):
        params = {'injected_square_current':
            {'amplitude':self.ampl, 'delay':100*pq.ms, 'duration':1000*pq.ms}}
        vm_used = self.model.inject_square_current(params)
        efel.setThreshold(0)

        trace3 = {'T': [float(t)*1000.0 for t in vm_used.times],
                  'V': [float(v) for v in vm_used.magnitude],
                  'stimulus_current': [current]}
        ALLEN_DURATION = 2000*qt.ms
        ALLEN_DELAY = 1000*qt.ms

        trace3['stim_end'] = [ float(ALLEN_DELAY)+float(ALLEN_DURATION) ]
        trace3['stim_start'] = [ float(ALLEN_DELAY)]
        results = efel.getMeanFeatureValues([trace3],['mean_AP_amplitude'])
        self.prediction = results
        return results
    def compute_score(self):
        if self.prediction is None:
            return None
        else:
            score = super(VmTest, self).compute_score(self.observation,
                                                                    self.prediction)
        score.related_data['vm'] = self.vm
        return score
#simple_yes_list = ['mean_AP_amplitude','mean_frequency','min_AHP_values','min_voltage_between_spikes','minimum_voltage','all_ISI_values','ISI_log_slope','mean_frequency','adaptation_index2','first_isi','ISI_CV','median_isi','AHP_depth_abs','sag_ratio2','sag_ratio2','peak_voltage','voltage_base','Spikecount','ohmic_input_resistance_vb_ssse','ohmic_input_resistance']

class PeakSpikesHeightVMTest(VmTest):
    score_type = scores.RatioScore

    def __init__(self,observation=None,observed_spk_cnt=None):
        self = self
        self.obervation = observation
        self.observed_spk_cnt = observed_spk_cnt
        scs = SpikeCountSearch(self.observed_spk_cnt)
        self.model = dtc.dtc_to_model()
        self.ampl = scs.generate_prediction(model)

    def generate_prediction(model):
        params = {'injected_square_current':
            {'amplitude':self.ampl, 'delay':100*pq.ms, 'duration':1000*pq.ms}}
        vm_used = self.model.inject_square_current(params)
        efel.setThreshold(0)

        trace3 = {'T': [float(t)*1000.0 for t in vm_used.times],
                  'V': [float(v) for v in vm_used.magnitude],
                  'stimulus_current': [current]}
        ALLEN_DURATION = 2000*qt.ms
        ALLEN_DELAY = 1000*qt.ms

        trace3['stim_end'] = [ float(ALLEN_DELAY)+float(ALLEN_DURATION) ]
        trace3['stim_start'] = [ float(ALLEN_DELAY)]
        results = efel.getMeanFeatureValues([trace3],['peak_voltage'])
        self.prediction = results
        return results
    def compute_score(self):
        if self.prediction is None:
            return None
        else:
            score = super(VmTest, self).compute_score(self.observation,
                                                                    self.prediction)
        score.related_data['vm'] = self.vm
        return score

class sag_ratio2VMTest(VmTest):
    score_type = scores.RatioScore

    def __init__(self,observation=None,observed_spk_cnt=None):
        self = self
        self.obervation = observation
        self.observed_spk_cnt = observed_spk_cnt
        scs = SpikeCountSearch(self.observed_spk_cnt)
        self.model = dtc.dtc_to_model()
        self.ampl = scs.generate_prediction(model)

    def generate_prediction(model):
        params = {'injected_square_current':
            {'amplitude':self.ampl, 'delay':100*pq.ms, 'duration':1000*pq.ms}}
        vm_used = self.model.inject_square_current(params)
        efel.setThreshold(0)

        trace3 = {'T': [float(t)*1000.0 for t in vm_used.times],
                  'V': [float(v) for v in vm_used.magnitude],
                  'stimulus_current': [current]}
        ALLEN_DURATION = 2000*qt.ms
        ALLEN_DELAY = 1000*qt.ms

        trace3['stim_end'] = [ float(ALLEN_DELAY)+float(ALLEN_DURATION) ]
        trace3['stim_start'] = [ float(ALLEN_DELAY)]
        results = efel.getMeanFeatureValues([trace3],['sag_ratio2'])
        self.prediction = results
        return results
    def compute_score(self):
        if self.prediction is None:
            return None
        else:
            score = super(VmTest, self).compute_score(self.observation,
                                                                    self.prediction)
        score.related_data['vm'] = self.vm
        return score


class AHP_depth_abs(VmTest):
    score_type = scores.RatioScore

    def __init__(self,observation=None,observed_spk_cnt=None):
        self = self
        self.obervation = observation
        self.observed_spk_cnt = observed_spk_cnt
        scs = SpikeCountSearch(self.observed_spk_cnt)
        self.model = dtc.dtc_to_model()
        self.ampl = scs.generate_prediction(model)

    def generate_prediction(model):
        params = {'injected_square_current':
            {'amplitude':self.ampl, 'delay':100*pq.ms, 'duration':1000*pq.ms}}
        vm_used = self.model.inject_square_current(params)
        efel.setThreshold(0)

        trace3 = {'T': [float(t)*1000.0 for t in vm_used.times],
                  'V': [float(v) for v in vm_used.magnitude],
                  'stimulus_current': [current]}
        ALLEN_DURATION = 2000*qt.ms
        ALLEN_DELAY = 1000*qt.ms

        trace3['stim_end'] = [ float(ALLEN_DELAY)+float(ALLEN_DURATION) ]
        trace3['stim_start'] = [ float(ALLEN_DELAY)]
        results = efel.getMeanFeatureValues([trace3],['sag_ratio2'])
        self.prediction = results
        return results
    def compute_score(self):
        if self.prediction is None:
            return None
        else:
            score = super(VmTest, self).compute_score(self.observation,
                                                                    self.prediction)
        score.related_data['vm'] = self.vm
        return score
