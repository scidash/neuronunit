from pyneuroml import pynml


class Backend:
    # The function (e.g. from pynml) that handles running the simulation
    f = None

    def load_model(self):
        # Loads the model into memory
        pass

    def set_attrs(self, attrs):
        # Set attributes (e.g. resting potential) of model in memory or on disk
        pass

    def update_run_params(self):
        # Set attributes (e.g. somatic current injection) of simulation 
        # in memory or on disk
        pass


class jNeuroMLBackend(Backend):
    f = pynml.run_lems_with_jneuroml

    def set_attrs(self, attrs):
        self.set_lems_attrs(attrs)

    def update_run_params(self, attrs):
        self.update_lems_run_params(attrs)

        

class NEURONBackend(Backend):
    f = pynml.run_lems_with_jneuroml_neuron

    def load_model(self):
        pass

    def set_attrs(self, attrs):
        pass

    def update_run_params(self):
        pass