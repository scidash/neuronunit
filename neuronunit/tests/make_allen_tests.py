from neuronunit.tests.base import VmTest
import pickle
import numpy as np
from allensdk.core.cell_types_cache import CellTypesCache


class AllenTest(VmTest):
    def __init__(
        self,
        observation={"mean": None, "std": None},
        name="generic_allen",
        prediction={"mean": None, "std": None},
        **params
    ):
        super(AllenTest, self).__init__(observation, name, **params)

    name = ""
    aliases = ""

    def compute_params(self):
        self.params["t_max"] = (
            self.params["delay"] + self.params["duration"] + self.params["padding"]
        )

    def set_observation(self, observation):
        self.observation = {}
        self.observation["mean"] = observation
        self.observation["std"] = observation

    def set_prediction(self, prediction):
        self.prediction = {}
        self.prediction["mean"] = prediction
        self.prediction["std"] = prediction


ctc = CellTypesCache(manifest_file="manifest.json")
features = ctc.get_ephys_features()
