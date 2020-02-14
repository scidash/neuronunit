# coding: utf-8
import unittest

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt


from neuronunit.optimisation.optimization_management import inject_and_plot_model, dtc_to_rheo
from neuronunit.optimisation.optimization_management import inject_and_plot_passive_model

import numpy as np
from neuronunit.optimisation.data_transport_container import DataTC
from neuronunit.optimisation import model_parameters
from elephant.spike_train_generation import threshold_detection
import quantities as pq


class testCrucialBackendsSucceed(unittest.TestCase):
    def setUp(self):
        model_parameters.MODEL_PARAMS.keys()
        self.backends =  ["RAW", "HH"]
        self.other_backends =["NEURON","NEURON"]#["BHH","ADEXP","NEURON"]
        self.backends_complex =  ["GLIF"]#,"NEURON"]
        #self.julia_backend ="JHH"

        #raw_attrs = {k:np.mean(v) for k,v in model_parameters.MODEL_PARAMS[backend].items()}
        #self.backends = backends
        self.model_parameters = model_parameters

    def test_must_pass_0(self):
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

        return True
    def test_prefer_pass_1(self):
        fig, axs = plt.subplots(len(self.backends)*2+1,figsize=(40, 40))
        cnt=0

        for b in self.other_backends:
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

        return True

    def test_prefer_pass_2(self):
        fig, axs = plt.subplots(len(self.backends)*2+1,figsize=(40, 40))
        cnt=0
        for b in self.backends_complex:
            if b in str("GLIF"):
                print(self.model_parameters.MODEL_PARAMS[b])
                attrs_ = {k:v for k,v in model_parameters.MODEL_PARAMS["GLIF"].items() if type(v) is not type(dict())}

                attrs_ = {k:v for k,v in attrs_.items() if type(v) is not type(dict()) }
                attrs_ = {k:v for k,v in attrs_.items() if type(v) is not type(None) }
                attrs_ = {k:np.mean(v) for k,v in attrs_.items() if type(v[0]) is not type(str())}
                attrs = attrs_
            else:
                print('actually NEURON support only in docker container')
                #attrs = {k:np.mean(v) for k,v in self.model_parameters.MODEL_PARAMS[b].items()}
                return

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

        return True

if __name__ == '__main__':
    unittest.main()
    #a = testCrucialBackendsSucceed()
    #a.setUp()
    #a.test_prefer_pass_1()
    #import pdb
    #pdb.set_trace()
    #boolean = a.must_pass_0()
    #print(dir(a))
    #pre_model.rheobase
    #exit()
    '''
    def luxury_pass_0j(self):
    fig, axs = plt.subplots(len(self.backends)*2+1,figsize=(40, 40))
    cnt=0
    b =  self.julia_backend
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
    cnt+=1

    if len(vm)>0 and vm is not None:
        boolean = True
    else:
        boolean = False
    self.assertTrue(boolean)

    return True
    '''

    """
    def test_prefer_pass_3(self):
        fig, axs = plt.subplots(len(self.backends)*2+1,figsize=(40, 40))
        cnt=0
        #for b in self.complex_backends:
        b = self.julia_backend
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

        return True
    """

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
