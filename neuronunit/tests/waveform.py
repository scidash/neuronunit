"""Waveform neuronunit tests, e.g. testing AP waveform properties"""

from .base import np, pq, ncap, VmTest, scores


class InjectedCurrent:
    """Metaclass to mixin with InjectedCurrent tests."""

    required_capabilities = (ncap.ReceivesSquareCurrent,)

    default_params = dict(VmTest.default_params)
    default_params.update({"amplitude": 100 * pq.pA})

    def compute_params(self):
        self.params["injected_square_current"] = self.get_injected_square_current()
        self.params["injected_square_current"]["amplitude"] = self.params["amplitude"]


class APWidthTest(VmTest):
    """Test the full widths of action potentials at their half-maximum."""

    required_capabilities = (ncap.ProducesActionPotentials,)

    name = "AP width test"

    description = (
        "A test of the widths of action potentials " "at half of their maximum height."
    )

    score_type = scores.RatioScore

    units = pq.ms

    ephysprop_name = "Spike Half-Width"

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        # if get_spike_count is zero, then widths will be None
        # len of None returns an exception that is not handled
        model.inject_square_current(**self.params["injected_square_current"])

        widths = model.get_AP_widths()
        # Put prediction in a form that compute_score() can use.
        prediction = {
            "mean": np.mean(widths) if len(widths) else None,
            "std": np.std(widths) if len(widths) else None,
            "n": len(widths),
        }

        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if isinstance(prediction, type(None)):
            score = scores.InsufficientDataScore(None)
        elif prediction["n"] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APWidthTest, self).compute_score(observation, prediction)
        return score


class InjectedCurrentAPWidthTest(InjectedCurrent, APWidthTest):
    """Tests the full widths of APs at their half-maximum
    under current injection.
    """

    score_type = scores.ZScore

    units = pq.ms

    name = "Injected current AP width test"

    description = (
        "A test of the widths of action potentials "
        "at half of their maximum height when current "
        "is injected into cell."
    )

    def generate_prediction(self, model):
        print(self.params["injected_square_current"])
        model.inject_square_current(**self.params["injected_square_current"])
        prediction = super(InjectedCurrentAPWidthTest, self).generate_prediction(model)

        return prediction


class APAmplitudeTest(VmTest):
    """Test the heights (peak amplitude) of action potentials."""

    required_capabilities = (ncap.ProducesActionPotentials,)

    name = "AP amplitude test"

    description = (
        "A test of the amplitude (peak minus threshold) of " "action potentials."
    )

    score_type = scores.ZScore

    units = pq.mV

    ephysprop_name = "Spike Amplitude"

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        model.inject_square_current(**self.params["injected_square_current"])

        heights = model.get_AP_amplitudes() - model.get_AP_thresholds()
        # Put prediction in a form that compute_score() can use.
        prediction = {
            "mean": np.mean(heights) if len(heights) else None,
            "std": np.std(heights) if len(heights) else None,
            "n": len(heights),
        }
        return prediction

    def compute_score(self, observation, prediction):
        """Implementat sciunit.Test.score_prediction."""
        if prediction["n"] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APAmplitudeTest, self).compute_score(observation, prediction)
        return score


class InjectedCurrentAPAmplitudeTest(InjectedCurrent, APAmplitudeTest):
    """Test the heights (peak amplitude) of action potentials.

    Uses current injection.
    """

    name = "Injected current AP amplitude test"

    description = (
        "A test of the heights (peak amplitudes) of "
        "action potentials when current "
        "is injected into cell."
    )

    def generate_prediction(self, model):
        model.inject_square_current(**self.params["injected_square_current"])
        prediction = super(InjectedCurrentAPAmplitudeTest, self).generate_prediction(
            model
        )
        return prediction


class APThresholdTest(VmTest):
    """Test the full widths of action potentials at their half-maximum."""

    required_capabilities = (ncap.ProducesActionPotentials,)

    name = "AP threshold test"

    description = (
        "A test of the membrane potential threshold at which "
        "action potentials are produced."
    )

    score_type = scores.ZScore

    units = pq.mV

    ephysprop_name = "Spike Threshold"

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        model.inject_square_current(**self.params["injected_square_current"])

        threshes = model.get_AP_thresholds()
        # Put prediction in a form that compute_score() can use.
        prediction = {
            "mean": np.mean(threshes) if len(threshes) else None,
            "std": np.std(threshes) if len(threshes) else None,
            "n": len(threshes),
        }
        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction["n"] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APThresholdTest, self).compute_score(observation, prediction)
        return score


class InjectedCurrentAPThresholdTest(InjectedCurrent, APThresholdTest):
    """Test the thresholds of action potentials under current injection."""

    name = "Injected current AP threshold test"

    description = (
        "A test of the membrane potential threshold at which "
        "action potentials are produced under current injection."
    )

    def generate_prediction(self, model):
        model.inject_square_current(self.params["injected_square_current"])
        return super(InjectedCurrentAPThresholdTest, self).generate_prediction(model)
