"""NeuronUnit Test classes for ion channel models"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import quantities as pq

from sciunit.tests import ProtocolToFeaturesTest
from sciunit.scores import BooleanScore, FloatScore
from sciunit.converters import AtMostToBoolean
from neuronunit.capabilities.channel import NML2ChannelAnalysis
import pyneuroml.analysis.NML2ChannelAnalysis as ca


class _IVCurveTest(ProtocolToFeaturesTest):
    """Test the agreement between channel IV curves produced by models and
    those observed in experiments"""

    def __init__(self, observation, name="IV Curve Test", scale=False, **params):
        self.validate_observation(observation)
        for key, value in observation.items():
            setattr(self, key, value)
        # Whether to scale the predicted IV curve to
        # minimize distance from the observed IV curve
        self.scale = scale
        self.converter = AtMostToBoolean(pq.Quantity(1.0, "pA**2"))
        super(_IVCurveTest, self).__init__(observation, name=name, **params)

    required_capabilities = (NML2ChannelAnalysis,)

    units = {"v": pq.V, "i": pq.pA}

    score_type = BooleanScore

    observation_schema = [
        (
            "Current Array, Voltage Array",
            {
                "i": {"units": True, "iterable": True, "required": True},
                "v": {"units": True, "iterable": True, "required": True},
            },
        ),
    ]

    default_params = {
        "v_min": -80.0 * pq.mV,
        "v_step": 20.0 * pq.mV,
        "v_max": 60.0 * pq.mV,
        "dt": 0.025 * pq.ms,
        "tmax": 100 * pq.ms,
    }

    params_schema = {
        "v_min": {"type": "voltage", "required": True},
        "v_step": {"type": "voltage", "min": 1e-3, "required": True},
        "v_max": {"type": "voltage", "required": True},
        "dt": {"type": "time", "min": 0, "required": False},
        "tmax": {"type": "time", "min": 0, "required": False},
    }

    def validate_observation(self, observation):
        super(_IVCurveTest, self).validate_observation(observation)
        if hasattr(observation["v"], "units"):
            observation["v"] = observation["v"].rescale(pq.V)  # Rescale to V
        if hasattr(observation["i"], "units"):
            observation["i"] = observation["i"].rescale(pq.pA)  # Rescale to pA

    def validate_params(self, params):
        params = super(_IVCurveTest, self).validate_params(params)
        assert params["v_max"] > params["v_min"], "v_max must be greater than v_min"
        return params

    def condition_model(self, model):
        # Start with the model defaults
        params = model.default_params.copy()
        # Required parameter by ChannelAnalysis to commpute an IV curve
        params.update(**{"ivCurve": True})
        # Use test parameters as well to build the new LEMS file
        params.update(self.params)
        # Make a new LEMS file with these parameters
        model.ca_make_lems_file(**params)

    def setup_protocol(self, model):
        """Implement sciunit.tests.ProtocolToFeatureTest.setup_protocol."""
        self.condition_model(model)

    def get_result(self, model):
        results = model.ca_run_lems_file(verbose=True)
        return results

    def extract_features(self, model, result):
        """Implemented in the subclasses"""
        return NotImplementedError()

    def interp_IV_curves(self, v_obs, i_obs, v_pred, i_pred):
        """
        Interpolate IV curve along a larger and evenly spaced range of holding
        potentials. Ensures that test is restricted to ranges present in both
        predicted and observed data.
        v_pred: The array of holding potentials in model.
        i_pred: The array or dict of resulting currents (single values)
                from model .
        v_obs: The array of holding potentials from the experiments.
        i_obs: The array or dict of resulting currents (single values)
               from the experiment .
        """

        if isinstance(i_obs, dict):
            # Convert dict to numpy array.
            units = list(i_obs.values())[0].units
            i_obs = np.array([i_obs[float(key)] for key in v_obs]) * units
        if isinstance(i_pred, dict):
            # Convert dict to numpy array.
            units = list(i_pred.values())[0].units
            i_pred = np.array([i_pred[float(key)] for key in v_pred]) * units

        # Removing units temporarily due to
        # https://github.com/python-quantities/python-quantities/issues/105
        v_obs = v_obs.rescale(pq.mV).magnitude
        i_obs = i_obs.rescale(pq.pA).magnitude
        v_pred = v_pred.rescale(pq.mV).magnitude
        i_pred = i_pred.rescale(pq.pA).magnitude

        f_obs = interp1d(v_obs, i_obs, kind="cubic")
        f_pred = interp1d(v_pred, i_pred, kind="cubic")
        start = max(v_obs[0], v_pred[0])  # Min voltage in mV
        stop = min(v_obs[-1], v_pred[-1])  # Max voltage in mV

        v_new = np.linspace(start, stop, 100)  # 1 mV interpolation
        i_obs_interp = f_obs(v_new) * pq.pA  # Interpolated current from data
        i_pred_interp = f_pred(v_new) * pq.pA  # Interpolated current from model

        return {"v": v_new * pq.mV, "i_pred": i_pred_interp, "i_obs": i_obs_interp}

    def compute_score(self, observation, prediction):
        # Sum of the difference between the curves.
        o = observation
        p = prediction
        interped = self.interp_IV_curves(o["v"], o["i"], p["v"], p["i"])

        if self.scale:

            def f(sf):
                score = FloatScore.compute_ssd(
                    interped["i_obs"], (10 ** sf) * interped["i_pred"]
                )
                return score.score.magnitude

            result = minimize(f, 0.0)
            scale_factor = 10 ** result.x
            interped["i_pred"] *= scale_factor
        else:
            scale_factor = 1

        score = FloatScore.compute_ssd(interped["i_obs"], interped["i_pred"])
        score.related_data["scale_factor"] = scale_factor
        self.interped = interped
        return score

    def bind_score(self, score, model, observation, prediction):
        score.description = (
            "The sum-squared difference in the observed and "
            "predicted current values over the range of the "
            "tested holding potentials."
        )
        # Observation and prediction are automatically stored in score,
        # but since that is before interpolation, we store the interpolated
        # versions as well.
        score.related_data.update(self.interped)

        def plot(score):
            import matplotlib.pyplot as plt

            rd = score.related_data
            rd["v"] = rd["v"].rescale("V")
            rd["i_obs"] = rd["i_obs"].rescale("A")
            rd["i_pred"] = rd["i_pred"].rescale("A")
            score.test.last_model.plot_iv_curve(
                rd["v"], rd["i_obs"], color="k", label="Observed (data)"
            )
            score.test.last_model.plot_iv_curve(
                rd["v"],
                rd["i_pred"],
                color="r",
                same_fig=True,
                label="Predicted (model)",
            )
            plt.title("%s on %s: %s" % (score.model, score.test, score))

        score.plot = plot.__get__(score)  # Binds this method to 'score'.
        return score


class IVCurveSSTest(_IVCurveTest):
    """Test IV curves using steady-state curent"""

    def extract_features(self, model, results):
        iv_data = model.ca_compute_iv_curve(results)
        return {"v": iv_data["hold_v"], "i": iv_data["i_steady"]}


class IVCurvePeakTest(_IVCurveTest):
    """Test IV curves using steady-state curent"""

    def extract_features(self, model, results):
        iv_data = model.ca_compute_iv_curve(results)
        return {"v": iv_data["hold_v"], "i": iv_data["i_peak"]}
