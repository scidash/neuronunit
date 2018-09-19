import urllib, json, quantities
from scipy.interpolate import interp1d
import numpy as np
from neo import AnalogSignal

class NeuroMLDBModel:
    def __init__(self, model_id = "NMLCL000086"):
        self.model_id = model_id
        self.api_url = "https://neuroml-db.org/api/"

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

    def fetch_waveform_as_AnalogSignal(self, waveform_id, resolution_ms = 0.05, units = "mV"):

        # If signal not in cache
        if waveform_id not in self.waveform_signals:
            # Load api URL into Python
            data = self.read_api_url(self.api_url + "waveform?id=" + str(waveform_id))

            # Get time and signal values (from CSV format)
            t = np.array(data["Times"].decode('UTF-8').split(','),float)
            signal = np.array(data["Variable_Values"].decode('UTF-8').split(','),float)

            # Interpolate to regularly sampled series (API returns irregular)
            sig = interp1d(t,signal,fill_value="extrapolate")
            signal = sig(np.arange(min(t),max(t),resolution_ms))

            # Convert to neo.AnalogSignal
            signal = AnalogSignal(signal,units=units, sampling_period=resolution_ms*quantities.ms)

            self.waveform_signals[waveform_id] = signal

        return self.waveform_signals[waveform_id]
