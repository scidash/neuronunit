import numpy as np
import quantities as qt
import copy
from collections import OrderedDict
from sciunit import scores
from sciunit.scores.collections import ScoreArray
import sciunit
import pandas as pd
from jithub.models import model_classes


class DataTC(object):
    def model_default(self):

        if self.backend is not None:
            if self.attrs is None:
                from neuronunit.optimization.model_parameters import (
                    MODEL_PARAMS,
                    BPO_PARAMS,
                )

                if str("MAT") in self.backend:
                    self.backend_ref = str("MAT")
                if str("IZHI") in self.backend:
                    self.backend_ref = str("IZHI")
                if str("ADEXP") in self.backend:
                    self.backend_ref = str("ADEXP")

                self.attrs = {
                    k: np.mean(v) for k, v in MODEL_PARAMS[self.backend_ref].items()
                }
                self = DataTC(backend=self.backend, attrs=self.attrs)
                model = self.dtc_to_model()
                self.attrs = model._backend.default_attrs
                del model
            else:
                model = self.dtc_to_model()
                self.attrs = model._backend.default_attrs
                del model

    """
    Data Transport Container

    This Object class serves as a data type for storing rheobase search
    attributes and apriori model parameters,
    with the distinction that unlike the LEMS model this class
    can be cheaply transported across HOSTS/CPUs
    """

    def __init__(self, attrs=None, backend=None, _backend=None):
        self.lookup = {}
        self.rheobase = None
        self.previous = 0
        self.run_number = 0
        self.attrs = attrs
        self.boolean = False
        self.initiated = False
        self.backend = backend
        self._backend = _backend
        if self.backend is not None:
            if (
                str("MAT") in self.backend
                or str("IZHI") in self.backend
                or str("ADEXP") in self.backend
            ):
                self.jithub = True
            else:
                self.jithub = False
        if attrs is None:
            self.attrs = None
            self.model_default()
        else:
            self.attrs = attrs

    def to_bpo_param(self, attrs):
        from bluepyopt.parameters import Parameter

        lop = {}
        for k, v in attrs.items():
            p = Parameter(name=k, bounds=v, frozen=False)
            lop[k] = p
        self.param = lop
        return lop

    def self_evaluate(self, tests=None):
        from neuronunit.optimization import optimization_management as om_

        if tests is not None:
            self.tests = tests

        OM = self.dtc_to_opt_man()
        self = om_.dtc_to_rheo(self)
        d = OM.format_test(self)
        model = self.dtc_to_model()
        scores_ = []
        for t in self.tests:
            if hasattr(t, "allen"):
                continue
            if "RheobaseTest" in t.name:
                t.score_type = sciunit.scores.ZScore
            try:
                score_gene = t.judge(model)
            except:
                score_gene = None
                lns = 100

            if score_gene is not None:
                if not isinstance(
                    type(score_gene), sciunit.scores.InsufficientDataScore
                ):
                    if not isinstance(type(score_gene.log_norm_score), type(None)):
                        try:

                            lns = np.abs(score_gene.log_norm_score)
                        except:
                            if score_gene.raw is not None:
                                lns = np.abs(float(score_gene.raw))
                            else:
                                lns = 100.0
                    else:
                        try:
                            lns = np.abs(float(score_gene.raw))
                        except:
                            lns = 100
                else:
                    try:
                        lns = np.abs(float(score_gene.raw))
                    except:
                        lns = 100
            if lns == np.inf or lns == -np.inf:
                lns = np.abs(float(score_gene.raw))
            scores_.append(lns)
        for i, s in enumerate(scores_):
            if s == np.inf or s == -np.inf:
                scores_[i] = 100  # np.abs(float(score_gene.raw))
        self.scores_ = scores_

        temp = [t for t in self.tests if not hasattr(t, "allen")]
        self.SA = ScoreArray(temp, self.scores_)

        return self

    def make_pretty(self, tests):
        import pandas as pd

        self.self_evaluate(tests=tests)
        self.obs_preds = pd.DataFrame(columns=["observations", "predictions"])
        holding_obs = {
            t.name: t.observation["mean"] for t in self.tests if not hasattr(t, "allen")
        }
        grab_keys = []
        for t in self.tests:
            if "value" in t.prediction.keys():
                grab_keys.append("value")
            else:
                grab_keys.append("mean")
        holding_preds = {
            t.name: t.prediction[k]
            for t, k in zip(self.tests, grab_keys)
            if not hasattr(t, "allen")
        }
        ##
        # This step only partially undoes quantities sins.
        ##
        qt.quantity.PREFERRED = [qt.mV, qt.pA, qt.MOhm, qt.ms, qt.pF, qt.Hz / qt.pA]

        for k, v in holding_preds.items():
            if k in holding_obs.keys() and k in holding_preds:
                v.rescale_preferred()
                v = v.simplified
                if np.round(v, 2) != 0:
                    v = np.round(v, 2)

        for k, v in holding_obs.items():
            if k in holding_obs.keys() and k in holding_preds:
                v.rescale_preferred()
                v = v.simplified
                if np.round(v, 2) != 0:
                    v = np.round(v, 2)

        for k, v in holding_preds.items():
            if k in holding_obs.keys() and k in holding_preds:
                units1 = holding_preds[k].rescale_preferred().units  # v.units)

                holding_preds[k] = holding_preds[k].simplified
                holding_preds[k] = holding_preds[k].rescale(units1)
                if np.round(holding_preds[k], 2) != 0:
                    holding_preds[k] = np.round(holding_preds[k], 2)

                holding_obs[k] = holding_obs[k].simplified
                holding_obs[k] = holding_obs[k].rescale(units1)
                if np.round(holding_obs[k], 2) != 0:
                    holding_obs[k] = np.round(holding_obs[k], 2)

        temp_obs = pd.DataFrame([holding_obs], index=["observations"])
        temp_preds = pd.DataFrame([holding_preds], index=["predictions"])
        # like a score array but nicer reporting of test name instead of test data type.
        not_SA = {
            t.name: np.round(score, 2) for t, score in zip(self.tests, self.scores_)
        }
        temp_scores = pd.DataFrame([not_SA], index=["Z-Scores"])
        self.obs_preds = pd.concat([temp_obs, temp_preds, temp_scores])
        self.obs_preds = self.obs_preds.T
        return self.obs_preds

    def dtc_to_opt_man(self):
        from neuronunit.optimization.optimization_management import OptMan
        from collections import OrderedDict
        from neuronunit.optimization.model_parameters import MODEL_PARAMS

        OM = OptMan(self.tests, self.backend)
        OM.boundary_dict = MODEL_PARAMS[self.backend]
        OM.backend = self.backend
        OM.td = list(OrderedDict(OM.boundary_dict).keys())
        return OM

    def get_agreement(self):
        self = self.self_evaluate()
        OM = self.dtc_to_opt_man()
        self = OM.get_agreement(self)
        return self

    def add_constant(self):
        if self.constants is not None:
            self.attrs.update(self.constants)
        return  # self.attrs

    def format_test(self):
        from neuronunit.optimisation.optimization_management import (
            switch_logic,
            active_values,
            passive_values,
        )

        # pre format the current injection dictionary based on pre computed
        # rheobase values of current injection.
        # This is much like the hooked method from the old get neab file.
        self.protocols = {}
        if not hasattr(self, "tests"):
            self.tests = copy.copy(self.tests)
        if hasattr(self.tests, "keys"):  # is type(dict):
            tests = [key for key in self.tests.values()]
            self.tests = switch_logic(tests)  # ,self.tests.use_rheobase_score)
        else:
            self.tests = switch_logic(self.tests)

        for v in self.tests:
            k = v.name
            self.protocols[k] = {}
            if hasattr(v, "passive"):
                if v.passive == False and v.active == True:
                    keyed = self.protocols[k]  # .params
                    self.protocols[k] = active_values(keyed, self.rheobase)

                    if str("APThresholdTest") in v.name and not self.threshold:
                        model = self.dtc_to_model()
                        model.attrs = model._backend.default_attrs
                        treshold = v.generate_prediction(model)
                        self.threshold = treshold

                    if str("APThresholdTest") in v.name:
                        v.threshold = None
                        v.threshold = self.threshold
                elif v.passive == True and v.active == False:
                    keyed = self.protocols[k]  # .params
                    self.protocols[k] = passive_values(keyed)

            if v.name in str("RestingPotentialTest"):
                self.protocols[k]["injected_square_current"] = {}
                self.protocols[k]["injected_square_current"]["amplitude"] = 0.0 * qt.pA
            keyed = v.params["injected_square_current"]
            v.params["t_max"] = keyed["delay"] + keyed["duration"] + 200.0 * pq.ms
        return self.tests

    def dtc_to_model(self):
        if (
            str("MAT") in self.backend
            or str("IZHI") in self.backend
            or str("ADEXP") in self.backend
        ):
            self.jithub = True

        else:
            self.jithub = False
        if self.attrs is None:
            self.model_default()
        if self.jithub:
            if str("MAT") in self.backend:
                model = model_classes.MATModel()
                model._backend.attrs = self.attrs
                model.attrs = self.attrs
                model.params = self.to_bpo_param(self.attrs)
                assert len(self.attrs)
                assert len(model.attrs)
                return model
            if str("IZHI") in self.backend:
                model = model_classes.IzhiModel()
                model._backend.attrs = self.attrs
                model.attrs = self.attrs
                model.params = self.to_bpo_param(self.attrs)
                assert len(self.attrs)
                assert len(model.attrs)
                return model
            if str("ADEXP") in self.backend:
                model = model_classes.ADEXPModel()
                model._backend.attrs = self.attrs
                model.attrs = self.attrs
                model.params = self.to_bpo_param(self.attrs)
                assert len(self.attrs)
                assert len(model.attrs)
                return model

        else:
            from neuronunit.models.very_reduced_sans_lems import VeryReducedModel

            model = VeryReducedModel(backend=self.backend, attrs=self.attrs)
            if "GLIF" in self.backend:
                model._backend.set_attrs(self.attrs)
                model.attrs = self.attrs
            else:
                model.set_attrs(self.attrs)
            if model.attrs is None:
                model.attrs = self.attrs
            return model

    def dtc_to_sciunit_model(self):
        model = self.dtc_to_model()
        sciunit_model = model._backend.as_sciunit_model()
        return sciunit_model

    def dtc_to_gene(self, subset_params=None):
        """
        These imports probably need to be contained to stop recursive imports
        """
        from deap import base
        import array
        from deap import creator

        creator.create(
            "FitnessMin", base.Fitness, weights=tuple(-1.0 for i in range(0, 10))
        )
        creator.create(
            "Individual", array.array, typecode="d", fitness=creator.FitnessMin
        )

        # from neuronunit.optimisation.optimization_management import WSListIndividual
        # print('warning translation dictionary should be used, to garuntee correct attribute order from random access dictionaries')
        if "IZHI" in self.backend:
            self.attrs.pop("dt", None)
            self.attrs.pop("Iext", None)
        if subset_params:
            pre_gene = OrderedDict()
            for k in subset_params:
                pre_gene[k] = self.attrs[k]
        else:
            pre_gene = OrderedDict(self.attrs)
        pre_gene = list(pre_gene.values())
        gene = creator.Individual(pre_gene)
        return gene

    def judge_test(self, index=0):
        model = self.dtc_to_model()
        if not hasattr(self, "tests"):
            print("warning dtc object does not contain NU-tests yet")
            return dtc

        ts = self.tests
        # this_test = ts[index]
        if not hasattr(self, "preds"):
            self.preds = {}
        for this_test in self.tests:
            this_test.setup_protocol(model)
            pred = this_test.extract_features(model, this_test.get_result(model))
            pred1 = this_test.generate_prediction(model)
            self.preds[this_test.name] = pred
        return self.preds

    """
    def jt_ratio(self,index=0):
        from sciunit import scores
        model = self.dtc_to_model()
        if not hasattr(self,'tests'):
            print('warning dtc object does not contain NU-tests yet')
            return dtc

        ts = self.tests
        #this_test = ts[index]
        if not hasattr(self,'preds'):
            self.preds = {}
        tests = self.format_test()
        for this_test in self.tests:
            if this_test.passive:
                this_test.params['injected_square_current'] = {}
                this_test.params['injected_square_current']['amplitude'] = -10*qt.pA

                this_test.params['injected_square_current']['duration'] = 1000*qt.ms
                this_test.params['injected_square_current']['delay'] = 200*qt.ms

                this_test.setup_protocol(model)
                pred = this_test.extract_features(model,this_test.get_result(model))
            else:
                this_test.params['injected_square_current']['amplitude'] = self.rheobase
                this_test.params['injected_square_current']['duration'] = 1000*qt.ms
                this_test.params['injected_square_current']['delay'] = 200*qt.ms

                pred = this_test.generate_prediction(model)
            #self.preds[this_test.name] = pred
            ratio_type = scores.RatioScore
            temp = copy.copy(this_test.score_type)
            this_test.score_type = ratio_type
            try:
                #print(this_test.name)
                self.rscores[rscores.name] = this_test.compute_score(this_test.observation,pred)

            except:
                this_test.score_type = temp
                #self.rscores = this_test.compute_score(model)
                self.rscores[rscores.name] = this_test.compute_score(this_test.observation,pred)

                self.failed = {}
                self.failed['pred'] = pred
                self.failed['observation'] = this_test.observation

        return self.preds
    """

    def check_params(self):
        self.judge_test()

        return self.preds

    def plot_obs(self, ow):
        """
        assuming a waveform exists (observed waved-form) plot to terminal with ascii
        This is useful for debugging new backends, in bash big/fast command line orientated optimization routines.
        """

        t = [float(f) for f in ow.times]
        v = [float(f) for f in ow.magnitude]
        fig = apl.figure()
        fig.plot(
            t,
            v,
            label=str("observation waveform from inside dtc: "),
            width=100,
            height=20,
        )
        fig.show()

    def iap(self, tests=None):
        """
        Inject and plot to terminal with ascii
        This is useful for debugging new backends, in bash big/fast command line orientated optimization routines.
        """
        from neuronunit.optimisation import optimization_management as om_

        if tests is not None:
            self.tests = tests

        OM = self.dtc_to_opt_man()
        self = om_.dtc_to_rheo(self)

        model = self.dtc_to_model()
        pms = uset_t.params
        pms["injected_square_current"]["amplitude"] = self.rheobase
        model.inject_square_current(pms["injected_square_current"])
        nspike = model.get_spike_train()
        self.nspike = nspike
        vm = model.get_membrane_potential()
        return vm
