"""F/I neuronunit tests.

For example, investigating firing rates and patterns as a
function of input current.
"""

import os
import multiprocessing
import copy

import dask.bag as db

import neuronunit
from neuronunit.optimization.data_transport_container import DataTC
from neuronunit.models.reduced import ReducedModel
from .base import np, pq, ncap, VmTest, scores

N_CPUS = multiprocessing.cpu_count()


class RheobaseTest(VmTest):
    """Serial implementation of a binary search to test the rheobase.

    Strengths: this algorithm is faster than the parallel class, present in
    this file under important and limited circumstances: this serial algorithm
    is faster than parallel for model backends that are able to implement
    numba jit optimization.

    Weaknesses this serial class is significantly slower, for many backend
    implementations including raw NEURON, NEURON via PyNN, and possibly GLIF.
    """

    def _extra(self):
        self.prediction = {}
        self.high = 300*pq.pA
        self.small = 0*pq.pA
        self.rheobase_vm = None

    required_capabilities = (ncap.ReceivesSquareCurrent,
                             ncap.ProducesSpikes)

    name = "Rheobase test"
    description = ("A test of the rheobase, i.e. the minimum injected current "
                   "needed to evoke at least one spike.")
    units = pq.pA
    ephysprop_name = 'Rheobase'
    score_type = scores.RatioScore

    default_params = dict(VmTest.default_params)
    default_params.update({'amplitude': 100*pq.pA,
                           'duration': 1000*pq.ms,
                           'tolerance': 1.0*pq.pA})

    params_schema = dict(VmTest.params_schema)
    params_schema.update({'tolerance': {'type': 'current', 'min': 1, 'required': False}})

    def condition_model(self, model):
        model.set_run_params(t_stop=self.params['tmax'])

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        self.condition_model(model)
        prediction = {'value': None}
        try:
            units = self.observation['value'].units
        except KeyError:
            units = self.observation['mean'].units
        # begin_rh = time.time()
        lookup = self.threshold_FI(model, units)
        sub = np.array([x for x in lookup if lookup[x] == 0])*units
        supra = np.array([x for x in lookup if lookup[x] > 0])*units
        if self.verbose:
            if len(sub):
                print("Highest subthreshold current is %s"
                      % (float(sub.max())*units))
            else:
                print("No subthreshold current was tested.")
            if len(supra):
                print("Lowest suprathreshold current is %s"
                      % supra.min())
            else:
                print("No suprathreshold current was tested.")
        if len(sub) and len(supra):
            rheobase = supra.min()
        else:
            rheobase = None
        prediction['value'] = rheobase
        return prediction

    def threshold_FI(self, model, units, guess=None):
        """Use binary search to generate an FI curve including rheobase."""
        lookup = {}  # A lookup table global to the function below.

        def f(ampl):
            if float(ampl) not in lookup:
                current = self.get_injected_square_current()
                current['amplitude'] = ampl
                model.inject_square_current(current)
                n_spikes = model.get_spike_count()

                if self.verbose >= 2:
                    print("Injected %s current and got %d spikes" %
                          (ampl, n_spikes))
                lookup[float(ampl)] = n_spikes
                spike_counts = \
                    np.array([n for x, n in lookup.items() if n > 0])
                if n_spikes and n_spikes <= spike_counts.min():
                    self.rheobase_vm = model.get_membrane_potential()

        max_iters = 25

        # evaluate once with a current injection at 0pA
        high = self.high
        small = self.small

        f(high)
        i = 0

        while True:
            # sub means below threshold, or no spikes
            sub = np.array([x for x in lookup if lookup[x] == 0])*units
            # supra means above threshold,
            # but possibly too high above threshold.

            supra = np.array([x for x in lookup if lookup[x] > 0])*units
            # The actual part of the Rheobase test that is
            # computation intensive and therefore
            # a target for parellelization.

            if len(supra) and len(sub):
                delta = float(supra.min()) - float(sub.max())
                tolerance = float(self.params['tolerance'].rescale(pq.pA))
                if delta < tolerance or (str(supra.min()) == str(sub.max())):
                    break

            if i >= max_iters:
                break
            # Its this part that should be like an evaluate function
            # that is passed to futures map.
            if len(sub) and len(supra):
                f((supra.min() + sub.max())/2)

            elif len(sub):
                f(max(small, sub.max()*2))

            elif len(supra):
                f(min(-small, supra.min()*2))
            i += 1

        return lookup

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction is None or \
           (isinstance(prediction, dict) and prediction['value'] is None):
            score = scores.InsufficientDataScore(None)
        else:
            score = super(RheobaseTest, self).\
                          compute_score(observation, prediction)
            # self.bind_score(score,None,observation,prediction)
        return score

    def bind_score(self, score, model, observation, prediction):
        """Bind additional attributes to the test score."""
        super(RheobaseTest, self).bind_score(score, model,
                                             observation, prediction)
        if self.rheobase_vm is not None:
            score.related_data['vm'] = self.rheobase_vm


class RheobaseTestP(RheobaseTest):
    """Parallel implementation of a binary search to test the rheobase.

    Strengths: this algorithm is faster than the serial class, present in this
    file for model backends that are not able to implement numba jit
    optimization, which actually happens to be typical of a signifcant number
    of backends.
    """

    name = "Rheobase test"
    description = ("A test of the rheobase, i.e. the minimum injected current "
                   "needed to evoke at least one spike.")
    units = pq.pA
    ephysprop_name = 'Rheobase'
    score_type = scores.RatioScore
    get_rheobase_vm = True

    def condition_model(self, model):
        model.set_run_params(t_stop=self.params['tmax'])

    def generate_prediction(self, model):
        """Generate the test prediction."""
        self.condition_model(model)
        dtc = DataTC()
        dtc.attrs = {}
        for k, v in model.attrs.items():
            dtc.attrs[k] = v

        # this is not a perservering assignment, of value,
        # but rather a multi statement assertion that will be checked.

        dtc = init_dtc(dtc)

        if model.orig_lems_file_path:
            dtc.model_path = model.orig_lems_file_path
            dtc.backend = model.backend
            assert os.path.isfile(dtc.model_path),\
                "%s is not a file" % dtc.model_path

        prediction = {}

        rheobase = find_rheobase(self, dtc).rheobase
        if rheobase is not None:
            # Something like the below commented line must happen to set the
            # vm trace associated with the rheobase current.  One additional
            # simulation may need to be run, unless we want one of the compute
            # nodes to set it (when found) in either the dtc or in the calling
            # instance of the test.
            # self.rheobase_vm = model.get_membrane_potential()
            prediction['value'] = float(rheobase) * pq.pA
            if self.get_rheobase_vm:
                print("Getting rheobase vm")
                c = self.get_injected_square_current()
                c['amplitude'] = prediction['value']
                model.inject_square_current(c)
                self.rheobase_vm = model.get_membrane_potential()
        else:
            prediction = None
            self.rheobase_vm = None
        return prediction


"""
Functions to support the parallel rheobase search.
"""


def check_fix_range(dtc):
    """Check for the rheobase value.

    Inputs: lookup, A dictionary of previous current injection values
    used to search rheobase
    Outputs: A boolean to indicate if the correct rheobase current was
    found and a dictionary containing the range of values used.
    If rheobase was actually found then rather returning a boolean and
    a dictionary, instead logical True, and the rheobase current is
    returned.
    given a dictionary of rheobase search values, use that
    dictionary as input for a subsequent search.
    """
    steps = []
    dtc.rheobase = None
    sub, supra = get_sub_supra(dtc.lookup)

    if 0. in supra and len(sub) == 0:
        dtc.boolean = True
        dtc.rheobase = -1
        return dtc
    elif (len(sub) + len(supra)) == 0:
        # This assertion would only be occur if there was a bug
        assert sub.max() <= supra.min()
    elif len(sub) and len(supra):
        # Termination criterion
        steps = np.linspace(sub.max(), supra.min(), N_CPUS+1)*pq.pA
        steps = steps[1:-1]*pq.pA
    elif len(sub):
        steps = np.linspace(sub.max(), 2*sub.max(), N_CPUS+1)*pq.pA
        steps = steps[1:-1]*pq.pA
    elif len(supra):
        steps = np.linspace(supra.min()-100, supra.min(),
                            N_CPUS+1)*pq.pA
        steps = steps[1:-1]*pq.pA

    dtc.current_steps = steps
    return dtc


def get_sub_supra(lookup):
    """Get subthreshold and suprathreshold current values."""
    sub, supra = [], []
    for current, n_spikes in lookup.items():
        if n_spikes == 0:  # No spikes
            sub.append(current)
        elif n_spikes > 0:  # Some spikes
            supra.append(current)

    sub = np.array(sorted(list(set(sub))))
    supra = np.array(sorted(list(set(supra))))
    return sub, supra


def check_current(dtc):
    """Check the response to the proposed current and count spikes.

    Inputs are an amplitude to test and a virtual model
    output is an virtual model with an updated dictionary.
    """
    dtc.boolean = False
    LEMS_MODEL_PATH = str(neuronunit.__path__[0]) + \
        str('/models/NeuroML2/LEMS_2007One.xml')
    dtc.model_path = LEMS_MODEL_PATH
    model = ReducedModel(dtc.model_path, name='vanilla',
                         backend=(dtc.backend, {'DTC': dtc}))

    if dtc.backend is str('NEURON') or dtc.backend is str('jNEUROML'):
        dtc.current_src_name = model._backend.current_src_name
        assert dtc.current_src_name is not None
        dtc.cell_name = model._backend.cell_name

    if hasattr(model._backend, 'current_src_name'):
        dtc.current_src_name = model._backend.current_src_name
        assert dtc.current_src_name is not None
        dtc.cell_name = model._backend.cell_name

    ampl = float(dtc.ampl)
    if ampl not in dtc.lookup or len(dtc.lookup) == 0:
        current = RheobaseTest.get_default_injected_square_current()
        uc = {'amplitude': ampl*pq.pA}
        current.update(uc)
        dtc.run_number += 1
        model.inject_square_current(current)
        dtc.previous = ampl
        n_spikes = model.get_spike_count()
        dtc.lookup[float(ampl)] = n_spikes
    return dtc


def init_dtc(dtc):
    """Exploit memory of last model in genes."""
    # check for memory and exploit it.
    if dtc.initiated is True:
        dtc = check_current(dtc)
        if dtc.boolean:
            return dtc

        else:
            # Exploit memory of the genes to inform searchable range.

            # if this model has lineage, assume it didn't mutate that
            # far away from it's ancestor.
            # using that assumption, on first pass, consult a very
            # narrow range, of test current injection samples:
            # only slightly displaced away from the ancestors rheobase
            # value.

            if isinstance(dtc.current_steps, float):
                dtc.current_steps = [0.75 * dtc.current_steps,
                                     1.25 * dtc.current_steps]
            elif isinstance(dtc.current_steps, list):
                dtc.current_steps = [s * 1.25 for s
                                     in dtc.current_steps]
            # logically unnecessary but included for readibility
            dtc.initiated = True

    if dtc.initiated is False:
        dtc.boolean = False
        steps = np.linspace(1, 250, 7)
        steps_current = [i*pq.pA for i in steps]
        dtc.current_steps = steps_current
        dtc.initiated = True
    return dtc


def find_rheobase(self, dtc):
    assert os.path.isfile(dtc.model_path),\
        "%s is not a file" % dtc.model_path
    # If this it not the first pass/ first generation
    # then assume the rheobase value found before mutation still holds
    # until proven otherwise.
    # dtc = check_current(model.rheobase,dtc)
    # If its not true enter a search, with ranges informed by memory
    cnt = 0
    sub = np.array([0, 0])
    while dtc.boolean is False and cnt < 40:
        if len(sub):
            if sub.max() > 1500.0:
                dtc.rheobase = None
                dtc.boolean = False
                return dtc
        dtc_clones = [copy.copy(dtc) for i
                      in range(0, len(dtc.current_steps))]
        for i, s in enumerate(dtc.current_steps):
            dtc_clones[i].ampl = dtc.current_steps[i]
        dtc_clones = [d for d in dtc_clones if not np.isnan(d.ampl)]

        b0 = db.from_sequence(dtc_clones, npartitions=N_CPUS)
        dtc_clone = list(b0.map(check_current).compute())
        for dtc in dtc_clone:
            if dtc.boolean is True:
                return dtc

        for d in dtc_clone:
            dtc.lookup.update(d.lookup)
        dtc = check_fix_range(dtc)

        cnt += 1
        sub, supra = get_sub_supra(dtc.lookup)
        if len(supra) and len(sub):
            delta = float(supra.min()) - float(sub.max())
            tolerance = self.params['tolerance'].rescale(pq.pA)
            if delta < tolerance or (str(supra.min()) ==
                                     str(sub.max())):
                dtc.rheobase = supra.min()*pq.pA
                dtc.boolean = True
                return dtc

        if self.verbose >= 2:
            print("Try %d: SubMax = %s; SupraMin = %s" %
                  (cnt, sub.max() if len(sub) else None,
                   supra.min() if len(supra) else None))
    return dtc
