import urllib, json, quantities
from scipy.interpolate import interp1d
import numpy as np
from neo import AnalogSignal
from neuronunit.models.static import StaticModel
import quantities as q

class NeuroMLDBModel:
    def __init__(self, model_id = "NMLCL000086"):
        self.model_id = model_id
        self.api_url = "https://neuroml-db.org/api/" # See docs at: https://neuroml-db.org/api

        self.waveforms = None

        self.waveform_signals = {}

    def read_api_url(self, url):
        response = urllib.urlopen(url)
        data = json.loads(response.read())
        return data

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
            t = np.array(data["Times"].decode('UTF-8').split(','),float)
            signal = np.array(data["Variable_Values"].decode('UTF-8').split(','),float)

            # Interpolate to regularly sampled series (API returns irregularly sampled)
            sig = interp1d(t,signal,fill_value="extrapolate")
            signal = sig(np.arange(min(t),max(t),resolution_ms))

            # Convert to neo.AnalogSignal
            signal = AnalogSignal(signal,units=units, sampling_period=resolution_ms*quantities.ms)

            self.waveform_signals[waveform_id] = signal

        return self.waveform_signals[waveform_id]



    def get_waveform_by_current(self, amplitude_nA):
        for w in self.waveforms:
            if w["Variable_Name"] == "Voltage":
                wave_amp = self.get_waveform_current_amplitude(w)
                if amplitude_nA == wave_amp:
                    return self.fetch_waveform_as_AnalogSignal(w["ID"])

        raise Exception("Did not find a Voltage waveform with injected " + str(amplitude_nA) +
                        ". See " + self.api_url + "model?id=" + self.model_id +
                        " for the list of available model waveforms.")

    def get_druckmann2013_standard_current(self):
        currents = []
        for w in self.waveforms:
            if w["Protocol_ID"] == "LONG_SQUARE" and w["Variable_Name"] == "Voltage":
                currents.append(self.get_waveform_current_amplitude(w))

        return [currents[-2]] # 2nd to last one is RBx1.25 waveform

    def get_druckmann2013_strong_current(self):
        currents = []
        for w in self.waveforms:
            if w["Protocol_ID"] == "LONG_SQUARE" and w["Variable_Name"] == "Voltage":
                currents.append(self.get_waveform_current_amplitude(w))

        return [currents[-1]] # The last one is RBx1.50 waveform

    def get_druckmann2013_input_resistance_currents(self):
        currents = []

        # Find and return negative square current injections
        for w in self.waveforms:
            if w["Protocol_ID"] == "SQUARE" and w["Variable_Name"] == "Voltage":
                amp = self.get_waveform_current_amplitude(w)
                if amp < 0:
                    currents.append(amp)

        return currents

    def get_waveform_current_amplitude(self, waveform):
        return float(waveform["Waveform_Label"].replace(" nA", "")) * q.nA


class NeuroMLDBStaticModel(StaticModel):
    def __init__(self, model_id, protocol_to_fetch="LONG_SQUARE", **params):
        self.nmldb_model = NeuroMLDBModel(model_id)
        self.nmldb_model.fetch_waveform_list()
        self.protocol = protocol_to_fetch

    def inject_square_current(self, current):
        self.vm = self.nmldb_model.get_waveform_by_current(current["amplitude"])







