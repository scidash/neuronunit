"""
Tests of features described in Druckmann et. al. 2013 (https://academic.oup.com/cercor/article/23/12/2994/470476)

AP analysis details (from suplementary info): https://github.com/scidash/neuronunit/files/2295064/bhs290supp.pdf

Spikes were detected by a crossing of a voltage threshold (-20 mV).
The beginning of a spike was then determined by a crossing of a threshold on the derivative of the voltage (12mV/msec).
The end of the spike was determined by the minimum value of the afterhyperpolarization (AHP) following the spike.
The peak point of the spike is the maximum in between these two points.
The amplitude of a spike is given by the difference between the voltage at the beginning and peak of the spike.
"""

from .base import np, pq, ncap, VmTest, scores, AMPL, DELAY, DURATION

class AP12AmplitudeDrop(VmTest):
    """
    1. Drop in AP amplitude (amp.) from first to second spike (mV)

    Difference in the voltage value between the amplitude of the first and second AP.
    """

class AP1SSAmplitudeChange(VmTest):
    """
    2. AP amplitude change from first spike to steady-state (mV)

    Steady state AP amplitude is calculated as the mean amplitude of the set of APs
    that occurred during the latter third of the current step.
    """

class AP1Amplitude(VmTest):
    """
    3. AP 1 amplitude (mV)

    Amplitude of the first AP.
    """

class AP1WidthHalfHeight(VmTest):
    """
    4. AP 1 width at half height (ms)

    Amount of time in between the first crossing (in the upwards direction) of the
    half-height voltage value and the second crossing (in the downwards direction) of
    this value, for the first AP. Half-height voltage is the voltage at the beginning of
    the AP plus half the AP amplitude.
    """

class AP1WidthPeakToTrough(VmTest):
    """
    5. AP 1 peak to trough time (ms)

    Amount of time between the peak of the first AP and the trough, i.e., the
    minimum of the AHP.
    """

class AP1RateOfChangePeakToTrough(VmTest):
    """
    6. AP 1 peak to trough rate of change (mV/ms)

    Difference in voltage value between peak and trough over the amount of time in
    between the peak and trough.
    """

class AP1AHPdepth(VmTest):
    """
    7. AP 1 Fast AHP depth (mV)

    Difference between the minimum of voltage at the trough and the voltage value at
    the beginning of the AP.
    """

class AP2Amplitude(VmTest):
    """
    8. AP 1 amplitude (mV)

    Same as :any:`AP1Amplitude` but for second AP
    """

class AP2WidthHalfHeight(VmTest):
    """
    9. AP 1 width at half height (ms)

    Same as :any:`AP1WidthHalfHeight` but for second AP
    """

class AP2WidthPeakToTrough(VmTest):
    """
    10. AP 1 peak to trough time (ms)

    Same as :any:`AP1WidthPeakToTrough` but for second AP
    """

class AP2RateOfChangePeakToTrough(VmTest):
    """
    11. AP 1 peak to trough rate of change (mV/ms)

    Same as :any:`AP1WidthPeakToTrough` but for second AP
    """

class AP2AHPdepth(VmTest):
    """
    12. AP 1 Fast AHP depth (mV)

    Same as :any:`AP1AHPdepth` but for second AP
    """




"""


9 	AP 2 width at half height (ms)
10 	AP 2 peak to trough time (ms)
11 	AP 2 peak to trough rate of change (mV/ms)
12 	AP 2 Fast AHP depth (mV)
13 	Percent change in AP amplitude, first to second spike (%)
14 	Percent change in AP width at half height, first to second spike (%)
15 	Percent change in AP peak to trough rate of change, first to second spike (%)
16 	Percent change in AP fast AHP depth, first to second spike (%)
17 	Input resistance for steady-state current (Ohm)
18 	Average delay to AP 1 (ms)
19 	SD of delay to AP 1 (ms)
20 	Average delay to AP 2 (ms)
21 	SD of delay to AP 2 (ms)
22 	Average initial burst interval (ms)
23 	SD of average initial burst interval (ms)
24 	Average initial accommodation (%)
25 	Average steady-state accommodation (%)
26 	Rate of accommodation to steady-state (1/ms)
27 	Average accommodation at steady-state (%)
28 	Average rate of accommodation during steady-state
29 	Average inter-spike interval (ISI) coefficient of variation (CV) (unit less)
30 	Median of the distribution of ISIs (ms)
31 	Average change in ISIs during a burst (%)
32 	Average rate, strong stimulus (Hz)
33 	Average delay to AP 1, strong stimulus (ms)
34 	SD of delay to AP 1, strong stimulus (ms)
35 	Average delay to AP 2, strong stimulus (ms)
36 	SD of delay to AP 2, strong stimulus (ms)
37 	Average initial burst ISI, strong stimulus (ms)
38 	SD of average initial burst ISI, strong stimulus (ms)
"""

class APWidthTest(VmTest):
    """Test the full widths of action potentials at their half-maximum."""

    required_capabilities = (ncap.ProducesActionPotentials,)

    name = "AP width test"

    description = ("A test of the widths of action potentials "
                   "at half of their maximum height.")

    score_type = scores.ZScore

    units = pq.ms

    ephysprop_name = 'Spike Half-Width'

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        # if get_spike_count is zero, then widths will be None
        # len of None returns an exception that is not handled
        model.rerun = True

        widths = model.get_AP_widths()
        # Put prediction in a form that compute_score() can use.
        prediction = {'mean': np.mean(widths) if len(widths) else None,
                      'std': np.std(widths) if len(widths) else None,
                      'n': len(widths)}

        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if isinstance(prediction, type(None)):
            score = scores.InsufficientDataScore(None)
        elif prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APWidthTest, self).compute_score(observation,
                                                           prediction)
        return score


class InjectedCurrentAPWidthTest(APWidthTest):
    """
    Tests the full widths of APs at their half-maximum
    under current injection.
    """
    required_capabilities = (ncap.ReceivesSquareCurrent,)

    params = {'injected_square_current':
              {'amplitude': 100.0*pq.pA, 'delay': DELAY, 'duration': DURATION}}

    name = "Injected current AP width test"

    description = ("A test of the widths of action potentials "
                   "at half of their maximum height when current "
                   "is injected into cell.")

    def generate_prediction(self, model):
        model.inject_square_current(self.params['injected_square_current'])
        prediction = super(InjectedCurrentAPWidthTest, self).\
         generate_prediction(model)

        return prediction


class APAmplitudeTest(VmTest):
    """Test the heights (peak amplitude) of action potentials."""

    required_capabilities = (ncap.ProducesActionPotentials,)

    name = "AP amplitude test"

    description = ("A test of the amplitude (peak minus threshold) of "
                   "action potentials.")

    score_type = scores.ZScore

    units = pq.mV

    ephysprop_name = 'Spike Amplitude'

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        model.rerun = True
        heights = model.get_AP_amplitudes() - model.get_AP_thresholds()
        # Put prediction in a form that compute_score() can use.
        prediction = {'mean': np.mean(heights) if len(heights) else None,
                      'std': np.std(heights) if len(heights) else None,
                      'n': len(heights)}
        return prediction

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APAmplitudeTest, self).compute_score(observation,
                                                               prediction)
        return score


class InjectedCurrentAPAmplitudeTest(APAmplitudeTest):
    """
    Tests the heights (peak amplitude) of action potentials
    under current injection.
    """

    required_capabilities = (ncap.ReceivesSquareCurrent,)

    params = {'injected_square_current':
              {'amplitude': 100.0*pq.pA, 'delay': DELAY, 'duration': DURATION}}

    name = "Injected current AP amplitude test"

    description = ("A test of the heights (peak amplitudes) of "
                   "action potentials when current "
                   "is injected into cell.")

    def generate_prediction(self, model):
        model.inject_square_current(self.params['injected_square_current'])
        prediction = super(InjectedCurrentAPAmplitudeTest, self).\
            generate_prediction(model)
        return prediction


class APThresholdTest(VmTest):
    """Tests the full widths of action potentials at their half-maximum."""

    required_capabilities = (ncap.ProducesActionPotentials,)

    name = "AP threshold test"

    description = ("A test of the membrane potential threshold at which "
                   "action potentials are produced.")

    score_type = scores.ZScore

    units = pq.mV

    ephysprop_name = 'Spike Threshold'

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        model.rerun = True
        threshes = model.get_AP_thresholds()
        # Put prediction in a form that compute_score() can use.
        prediction = {'mean': np.mean(threshes) if len(threshes) else None,
                      'std': np.std(threshes) if len(threshes) else None,
                      'n': len(threshes)}
        return prediction

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APThresholdTest, self).compute_score(observation,
                                                               prediction)
        return score


class InjectedCurrentAPThresholdTest(APThresholdTest):
    """Test the thresholds of action potentials under current injection."""

    required_capabilities = (ncap.ReceivesSquareCurrent,)

    params = {'injected_square_current':
              {'amplitude': 100.0*pq.pA, 'delay': DELAY, 'duration': DURATION}}

    name = "Injected current AP threshold test"

    description = ("A test of the membrane potential threshold at which "
                   "action potentials are produced under current injection.")

    def generate_prediction(self, model):
        model.inject_square_current(self.params['injected_square_current'])
        return super(InjectedCurrentAPThresholdTest, self).\
            generate_prediction(model)
