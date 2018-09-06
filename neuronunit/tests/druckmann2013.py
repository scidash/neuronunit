"""
Tests of features described in Druckmann et. al. 2013 (https://academic.oup.com/cercor/article/23/12/2994/470476)

AP analysis details (from suplementary info): https://github.com/scidash/neuronunit/files/2295064/bhs290supp.pdf

Spikes were detected by a crossing of a voltage threshold (-20 mV).
The beginning of a spike was then determined by a crossing of a threshold on the derivative of the voltage (12mV/msec).
The end of the spike was determined by the minimum value of the afterhyperpolarization (AHP) following the spike.
The peak point of the spike is the maximum in between these two points.
The amplitude of a spike is given by the difference between the voltage at the beginning and peak of the spike.

Numbers in class names refer to the numbers in the publication table
"""

from .base import np, pq, ncap, VmTest, scores, AMPL, DELAY, DURATION

class AP12AmplitudeDropTest(VmTest):
    """
    1. Drop in AP amplitude (amp.) from first to second spike (mV)

    Difference in the voltage value between the amplitude of the first and second AP.
    """

class AP1SSAmplitudeChangeTest(VmTest):
    """
    2. AP amplitude change from first spike to steady-state (mV)

    Steady state AP amplitude is calculated as the mean amplitude of the set of APs
    that occurred during the latter third of the current step.
    """

class AP1AmplitudeTest(VmTest):
    """
    3. AP 1 amplitude (mV)

    Amplitude of the first AP.
    """

class AP1WidthHalfHeightTest(VmTest):
    """
    4. AP 1 width at half height (ms)

    Amount of time in between the first crossing (in the upwards direction) of the
    half-height voltage value and the second crossing (in the downwards direction) of
    this value, for the first AP. Half-height voltage is the voltage at the beginning of
    the AP plus half the AP amplitude.
    """

class AP1WidthPeakToTroughTest(VmTest):
    """
    5. AP 1 peak to trough time (ms)

    Amount of time between the peak of the first AP and the trough, i.e., the
    minimum of the AHP.
    """

class AP1RateOfChangePeakToTroughTest(VmTest):
    """
    6. AP 1 peak to trough rate of change (mV/ms)

    Difference in voltage value between peak and trough over the amount of time in
    between the peak and trough.
    """

class AP1AHPDepthTest(VmTest):
    """
    7. AP 1 Fast AHP depth (mV)

    Difference between the minimum of voltage at the trough and the voltage value at
    the beginning of the AP.
    """

class AP2AmplitudeTest(VmTest):
    """
    8. AP 1 amplitude (mV)

    Same as :any:`AP1AmplitudeTest` but for second AP
    """

class AP2WidthHalfHeightTest(VmTest):
    """
    9. AP 1 width at half height (ms)

    Same as :any:`AP1WidthHalfHeightTest` but for second AP
    """

class AP2WidthPeakToTroughTest(VmTest):
    """
    10. AP 1 peak to trough time (ms)

    Same as :any:`AP1WidthPeakToTroughTest` but for second AP
    """

class AP2RateOfChangePeakToTroughTest(VmTest):
    """
    11. AP 1 peak to trough rate of change (mV/ms)

    Same as :any:`AP1RateOfChangePeakToTroughTest` but for second AP
    """

class AP2AHPDepthTest(VmTest):
    """
    12. AP 1 Fast AHP depth (mV)

    Same as :any:`AP1AHPDepthTest` but for second AP
    """

class AP12AmplitudeChangePercentTest(VmTest):
    """
    13.	Percent change in AP amplitude, first to second spike (%)

    Difference in AP amplitude between first and second AP divided by the first AP
    amplitude.
    """

class AP12HalfWidthChangePercentTest(VmTest):
    """
    14. Percent change in AP width at half height, first to second spike (%)

    Difference in AP width at half-height between first and second AP divided by the
    first AP width at half-height.
    """

class AP12PercentChangeInRateOfChangePeakToTroughTest(VmTest):
    """
    15. Percent change in AP peak to trough rate of change, first to second spike (%)

    Difference in peak to trough rate of change between first and second AP divided
    by the first AP peak to trough rate of change.
    """

class AP12PercentChangeInAHPDepthTest(VmTest):
    """
    16 	Percent change in AP fast AHP depth, first to second spike (%)

    Difference in depth of fast AHP between first and second AP divided by the first
    AP depth of fast AHP.
    """

class InputResistanceTest(VmTest):
    """
    17 	Input resistance for steady-state current (Ohm)

    Input resistance calculated by injecting weak subthreshold hyperpolarizing and
    depolarizing step currents. Input resistance was taken as linear fit of current to
    voltage difference.
    """

class AP1DelayMeanTest(VmTest):
    """
    18 	Average delay to AP 1 (ms)

    Mean of the delay to beginning of first AP over experimental repetitions of step
    currents.
    """

class AP1DelaySDTest(VmTest):
    """
    19 	SD of delay to AP 1 (ms)

    Standard deviation of the delay to beginning of first AP over experimental
    repetitions of step currents.
    """

class AP2DelayMeanTest(VmTest):
    """
    20 	Average delay to AP 2 (ms)

    Same as :any:`AP1DelayMeanTest` but for 2nd AP
    """

class AP2DelaySDTest(VmTest):
    """
    21 	SD of delay to AP 2 (ms)

    Same as :any:`AP1DelaySDTest` but for 2nd AP

    Only stochastic models will have a non-zero value for this test
    """

class Burst1ISIMeanTest(VmTest):
    """
    22 	Average initial burst interval (ms)

    Initial burst interval is defined as the average of the first two ISIs, i.e., the average
    of the time differences between the first and second AP and the second and third
    AP. This feature is the average the initial burst interval across experimental
    repetitions.
    """

class Burst1ISISDTest(VmTest):
    """
    23 	SD of average initial burst interval (ms)

    The standard deviation of the initial burst interval across experimental repetitions.
    """

class AccommodationMeanInitialTest(VmTest):
    """
    24 	Average initial accommodation (%)

    Initial accommodation is defined as the percent difference between the spiking rate of the
    first fifth of the step current and the *third* fifth of the step current.
    """


class AccommodationMeanSSTest(VmTest):
    """
    25 	Average steady-state accommodation (%)

    Steady-state accommodation is defined as the percent difference between the spiking rate
    of the first fifth of the step current and the last *fifth* of the step current.
    """


class AccommodationRateToSSTest(VmTest):
    """
    26 	Rate of accommodation to steady-state (1/ms)

    The percent difference between the spiking rate of the first fifth of the step current and
    final fifth of the step current divided by the time taken to first reach the rate of
    steady state accommodation.
    """

class AccommodationMeanAtSSTest(VmTest):
    """
    27 	Average accommodation at steady-state (%)

    Accommodation analysis based on a fit of the ISIs to an exponential function:
    ISI = A+B*exp(-t/tau). This feature gives the relative size of the constant term (A) to
    the term before the exponent (B).
    """

class AccommodationRateMeanAtSSTest(VmTest):
    """
    28 	Average rate of accommodation during steady-state

    Accommodation analysis based on a fit of the ISIs to an exponential function.
    This feature is the time constant of the exponent.
    """

class ISICVTest(VmTest):
    """
    29 	Average inter-spike interval (ISI) coefficient of variation (CV) (unit less)

    Coefficient of variation (mean divided by standard deviation) of the distribution
    of ISIs.
    """

class ISIMedianTest(VmTest):
    """
    30 	Median of the distribution of ISIs (ms)

    Median of the distribution of ISIs.
    """

class ISIBurstMeanChangeTest(VmTest):
    """
    31 	Average change in ISIs during a burst (%)

    Difference between the first and second ISI divided by the value of the first ISI.
    """

class SpikeRateStrongStimTest(VmTest):
    """
    32 	Average rate, strong stimulus (Hz)

    Firing rate of strong stimulus.
    """

class AP1DelayMeanStrongStimTest(VmTest):
    """
    33 	Average delay to AP 1, strong stimulus (ms)

    Same as :any:`AP1DelayMeanTest` but for strong stimulus
    """

class AP1DelaySDStrongStimTest(VmTest):
    """
    34 	SD of delay to AP 1, strong stimulus (ms)

    Same as :any:`AP1DelaySDTest` but for strong stimulus
    """

class AP2DelayMeanStrongStimTest(VmTest):
    """
    35 	Average delay to AP 2, strong stimulus (ms)


    Same as :any:`AP2DelayMeanTest` but for strong stimulus
    """

class AP2DelaySDStrongStimTest(VmTest):
    """
    36 	SD of delay to AP 2, strong stimulus (ms)

    Same as :any:`AP2DelaySDTest` but for strong stimulus
    """

class Burst1ISIMeanStrongStimTest(VmTest):
    """
    37 	Average initial burst ISI, strong stimulus (ms)

    Same as :any:`Burst1ISIMeanTest` but for strong stimulus
    """

class Burst1ISISDStrongStimTest(VmTest):
    """
    38 	SD of average initial burst ISI, strong stimulus (ms)

    Same as :any:`Burst1ISISDTest` but for strong stimulus
    """