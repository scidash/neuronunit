from neuronunit.tests.base import VmTest
import pickle
import numpy as np
from allensdk.core.cell_types_cache import CellTypesCache
from neuronunit.models.optimization_model_layer import OptimizationModel

from neuronunit.optimization.optimization_management import (
    multi_spiking_feature_extraction,
)
from sciunit.scores import RelativeDifferenceScore


class AllenTest(VmTest):
    def __init__(
        self,
        observation={"mean": None, "std": None},
        name="generic_allen",
        prediction={"mean": None, "std": None},
    ):
        super(AllenTest, self).__init__(observation, name)
        self.name = name
        self.score_type = RelativeDifferenceScore
        self.observation = observation
        self.prediction = prediction

    aliases = ""

    def generate_prediction(self, model=None):
        if self.prediction is None:
            dtc = OptimizationModel()
            dtc.backed = model.backend
            dtc.attrs = model.attrs
            dtc.rheobase = model.rheobase
            dtc.tests = [self]
            dtc = multi_spiking_feature_extraction(dtc)
            dtc, ephys0 = allen_wave_predictions(dtc, thirty=True)
            dtc, ephys1 = allen_wave_predictions(dtc, thirty=False)
            if self.name in ephys0.keys():
                feature = ephys0[self.name]
                self.prediction = {}
                self.prediction["value"] = feature
                # self.prediction['std'] = feature
            if self.name in ephys1.keys():
                feature = ephys1[self.name]
                self.prediction = {}
                self.prediction["value"] = feature
                # self.prediction['std'] = feature
            return self.prediction
        # ephys1.update()
        # if not len(self.prediction.keys()):

    def compute_params(self):
        self.params["t_max"] = (
            self.params["delay"] + self.params["duration"] + self.params["padding"]
        )

    # @property
    # def prediction(self):
    #    return self._prediction

    # @property
    # def observation(self):
    #    return self._observation

    # @observation.setter
    def set_observation(self, value):
        self.observation = {}
        self.observation["mean"] = value
        self.observation["std"] = value

    # @prediction.setter
    def set_prediction(self, value):
        self.prediction = {}
        self.prediction["mean"] = value
        self.prediction["std"] = value
