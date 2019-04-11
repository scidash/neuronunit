import sys, json, quantities
from scipy.interpolate import interp1d
import numpy as np
from neo import AnalogSignal
from neuronunit.models.static import StaticModel
import quantities as pq

if sys.version_info[0] >= 3:
    import urllib.request as urllib
else:
    import urllib

class NeuroMLDBModel:
    def __init__(self, model_id = "NMLCL000086"):
        self.model_id = model_id
        self.api_url = "https://neuroml-db.org/api/" # See docs at: https://neuroml-db.org/api

        self.waveforms = None

        self.waveform_signals = {}
        self.url_responses = {}

    def read_api_url(self, url):
        if url not in self.url_responses:
            response = urllib.urlopen(url).read()

            if sys.version_info[0] >= 3:
                response = response.decode("utf-8")

            self.url_responses[url] = json.loads(response)

        return self.url_responses[url]

    def fetch_waveform_list(self):

        # Fetch the list of waveforms from the API and cache the result
        if not self.waveforms:
            data = self.read_api_url(self.api_url + "model?id=" + str(self.model_id))
            self.waveforms = data["waveform_list"]

        return self.waveforms

    def fetch_waveform_as_AnalogSignal(self, waveform_id, resolution_ms = 0.01, units = "mV"):

        # If signal not in cache
        if waveform_id not in self.waveform_signals:
            # Load api URL into Python
            data = self.read_api_url(self.api_url + "waveform?id=" + str(waveform_id))

            # Get time and signal values (from CSV format)
            t = np.array(data["Times"].split(','),float)
            signal = np.array(data["Variable_Values"].split(','),float)

            # Interpolate to regularly sampled series (API returns irregularly sampled)
            sig = interp1d(t,signal,fill_value="extrapolate")
            signal = sig(np.arange(min(t),max(t),resolution_ms))

            # Convert to neo.AnalogSignal
            signal = AnalogSignal(signal,units=units, sampling_period=resolution_ms*quantities.ms)

            starts_from_ss = next(w for w in self.waveforms if w["ID"] == waveform_id)["Starts_From_Steady_State"] == 1

            if starts_from_ss:
                rest_wave = self.get_steady_state_waveform()

                t = np.concatenate((rest_wave.times, signal.times + rest_wave.t_stop)) * quantities.s
                v = np.concatenate((np.array(rest_wave), np.array(signal))) * quantities.mV

                signal = AnalogSignal(v, units=units, sampling_period=resolution_ms * quantities.ms)

            self.waveform_signals[waveform_id] = signal

        return self.waveform_signals[waveform_id]

    def get_steady_state_waveform(self):
        if not hasattr(self, "steady_state_waveform") or self.steady_state_waveform is None:
            for w in self.waveforms:
                if w["Protocol_ID"] == "STEADY_STATE" and w["Variable_Name"] == "Voltage":
                    self.steady_state_waveform = self.fetch_waveform_as_AnalogSignal(w["ID"])
                    return self.steady_state_waveform

            raise Exception("Did not find the resting waveform." +
                            " See " + self.api_url + "model?id=" + self.model_id +
                            " for the list of available model waveforms.")

        return self.steady_state_waveform

    def get_waveform_by_current(self, amplitude_nA):
        for w in self.waveforms:
            if w["Variable_Name"] == "Voltage":
                wave_amp = self.get_waveform_current_amplitude(w)
                if ((amplitude_nA < 0 * pq.nA and w["Protocol_ID"] == "SQUARE") or
                    (amplitude_nA >= 0 * pq.nA and w["Protocol_ID"] == "LONG_SQUARE")) \
                        and amplitude_nA == wave_amp:
                    return self.fetch_waveform_as_AnalogSignal(w["ID"])

        raise Exception("Did not find a Voltage waveform with injected " + str(amplitude_nA) +
                        ". See " + self.api_url + "model?id=" + self.model_id +
                        " for the list of available model waveforms.")

    def get_druckmann2013_standard_current(self):
        currents = []
        for w in self.waveforms:
            if w["Protocol_ID"] == "LONG_SQUARE" and w["Variable_Name"] == "Voltage":
                currents.append(self.get_waveform_current_amplitude(w))

        if len(currents) != 4:
            raise Exception("The LONG_SQUARE protocol for the model should have 4 waveforms")

        return [currents[-2]] # 2nd to last one is RBx1.5 waveform

    def get_druckmann2013_strong_current(self):
        currents = []

        for w in self.waveforms:
            if w["Protocol_ID"] == "LONG_SQUARE" and w["Variable_Name"] == "Voltage":
                currents.append(self.get_waveform_current_amplitude(w))

        if len(currents) != 4:
            raise Exception("The LONG_SQUARE protocol for the model should have 4 waveforms")

        return [currents[-1]] # The last one is RBx3 waveform

    def get_druckmann2013_input_resistance_currents(self):
        currents = []

        # Find and return negative square current injections
        for w in self.waveforms:
            if w["Protocol_ID"] == "SQUARE" and w["Variable_Name"] == "Voltage":
                amp = self.get_waveform_current_amplitude(w)
                if amp < 0 * pq.nA:
                    currents.append(amp)

        return currents

    def get_waveform_current_amplitude(self, waveform):
        return float(waveform["Waveform_Label"].replace(" nA", "")) * pq.nA


class NeuroMLDBStaticModel(StaticModel):
    def __init__(self, model_id, **params):
        self.nmldb_model = NeuroMLDBModel(model_id)
        self.nmldb_model.fetch_waveform_list()

    def inject_square_current(self, current):
        self.vm = self.nmldb_model.get_waveform_by_current(current["amplitude"])
