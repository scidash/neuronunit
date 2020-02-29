# coding: utf-8
import unittest

import matplotlib as mpl
mpl.use('Agg')




from neuronunit.optimisation.optimization_management import inject_and_plot_model, dtc_to_rheo
import numpy as np
from neuronunit.optimisation.data_transport_container import DataTC
from neuronunit.optimisation import model_parameters
from elephant.spike_train_generation import threshold_detection
import quantities as pq


class TestCrucialBackendsSucceed(unittest.TestCase):
    def setUp(self):
        model_parameters.MODEL_PARAMS.keys()
        self.backends =  ["RAW", "HH", "ADEXP", "BHH"]
        self.backends_complex =  ["GLIF", "NEURON"]


        raw_attrs = {k:np.mean(v) for k,v in model_parameters.MODEL_PARAMS[backend].items()}
        self.backends = backends
        self.model_parameters = model_parameters
    def must_pass_0(self):
	fig, axs = plt.subplots(len(self.backends)*2+1,figsize=(40, 40))
	cnt=0

        for b in self.backends:
            attrs = {k:np.mean(v) for k,v in self.model_parameters.MODEL_PARAMS[b].items()}
            pre_model = DataTC()
	    if str("V_REST") in attrs.keys():
		attrs["V_REST"] = -75.0

            pre_model.attrs = attrs
            pre_model.backend = b

            vm,_ = inject_and_plot_model(pre_model.attrs,b)
            axs[cnt].plot(vm.times,vm.magnitude)
	    axs[cnt].set_title(b)
            cnt+=1
            thresh = threshold_detection(vm,0.0*pq.mV)

            if len(thresh)>0 and vm is not None:
                boolean = True
            else:
                boolean = False
            self.assertTrue(boolean)

            vm,_ = inject_and_plot_passive_model(pre_model.attrs,b)
	    axs[cnt].plot(vm.times,vm.magnitude)
	    axs[cnt].set_title(b)

            if len(vm)>0 and vm is not None:
                boolean = True
            else:
                boolean = False
            self.assertTrue(boolean)

        return true


    """
    def not_required_to_pass_1(self):
        for b in self.backends:
            attrs = {k:np.mean(v) for k,v in model_parameters.MODEL_PARAMS[b].items()}
            pre_model = DataTC()
            pre_model.attrs = attrs
            pre_model.backend = b
            #inject_and_plot_model(attrs,b)
            inject_and_plot_model(raw_attrs,b)
            vm,_ = inject_and_plot_passive_model(raw_attrs,b)
    """

if __name__ == '__main__':
    unittest.main()
    a = TestCrucialBackendsSucceed()
    a.setUp()
    _ = a.must_pass_0()

#pre_model.rheobase
#exit()