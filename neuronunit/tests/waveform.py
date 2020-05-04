"""Waveform neuronunit tests, e.g. testing AP waveform properties"""
from .base import np, pq, ncap, VmTest, scores, AMPL, DELAY, DURATION
'''
try:
    import asciiplotlib as apl
    fig = apl.figure()
    #fig.plot([0,1], [0,1], label=str('spikes: ')+str(self.n_spikes), width=100, height=20)
    fig.show()
    ascii_plot = True
except:
    ascii_plot = False
'''
ascii_plot = False

def asciplot_code(vm,spkcnt):
    t = [float(f) for f in vm.times]
    v = [float(f) for f in vm.magnitude]
    fig = apl.figure()
    fig.plot(t, v, label=str('spikes: ')+str(spkcnt), width=100, height=20)
    fig.show()

class InjectedCurrent:
    """Metaclass to mixin with InjectedCurrent tests."""

    required_capabilities = (ncap.ReceivesSquareCurrent,)

    default_params = dict(VmTest.default_params)
    default_params.update({'amplitude': 100*pq.pA})
    def compute_params(self):
        self.verbose = False
        self.params['injected_square_current'] = \
            self.get_injected_square_current()
        self.params['injected_square_current']['amplitude'] = \
            self.params['amplitude']


class APWidthTest(VmTest):
    """Test the full widths of action potentials at their half-maximum."""

    required_capabilities = (ncap.ProducesActionPotentials,)

    name = "AP width test"

    description = ("A test of the widths of action potentials "
                   "at half of their maximum height.")

    score_type = scores.RatioScore

    units = pq.ms
    ephysprop_name = 'Spike Half-Width'
    def __init__(self,*args,**kwargs):
        print(args)
        print(kwargs)
        super(APWidthTest,self).__init__(*args,**kwargs)
        #self.verbose = False
    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        # if get_spike_count is zero, then widths will be None
        # len of None returns an exception that is not handled
        model.inject_square_current(self.params['injected_square_current'])
        self.verbose = False
        if self.verbose is True:
            print(self.params['injected_square_current'])
        model.get_membrane_potential()
        widths = model.get_AP_widths()
        if ascii_plot:
             asciplot_code(model.vM,model.get_spike_count())

        try:
            prediction = {'mean': np.mean(widths) if len(widths) else None,
                      'std': np.std(widths) if len(widths) else None,
                      'n': len(widths)}
        except:
            prediction = None
        return prediction

    def extract_features(self, model):
        prediction = self.generate_prediction(model)
        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        #if isinstance(prediction, type(None)):
        #    score = scores.InsufficientDataScore(None)
        if prediction is None:
            return scores.InsufficientDataScore(None)
        
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APWidthTest, self).compute_score(observation,
                                                           prediction)
        return score


class InjectedCurrentAPWidthTest(InjectedCurrent, APWidthTest):
    """Tests the full widths of APs at their half-maximum
    under current injection.
    """
    def __init__(self,*args,**kwargs):
    #    self = APWidthTest
        super(InjectedCurrentAPWidthTest,self).__init__(*args,**kwargs)

    required_capabilities = (ncap.ReceivesSquareCurrent,)

    score_type = scores.ZScore

    units = pq.ms

    name = "Injected current AP width test"

    description = ("A test of the widths of action potentials "
                   "at half of their maximum height when current "
                   "is injected into cell.")

    def generate_prediction(self, model):
        #self.verbose = False
        if self.verbose is True:
            print(self.params['injected_square_current'])

        model.inject_square_current(self.params['injected_square_current'])
        model.get_membrane_potential()
        if ascii_plot:
            asciplot_code(model.vM,model.get_spike_count())
        #try:
        prediction = super(InjectedCurrentAPWidthTest, self).\
            generate_prediction(model)
        self.prediction = prediction

        #except:
        #    prediction = None
        # useful to retain inside object.

        return prediction

    def extract_features(self, model):
        prediction = self.generate_prediction(model)
        return prediction

class TotalAPAmplitudeTest(VmTest):
    """
    Test the heights (peak amplitude) of action potentials.
    """
    required_capabilities = (ncap.ProducesActionPotentials,)
    name = "AP amplitude test"
    description = ("A test of the amplitude (peak minus threshold) of "
                   "action potentials.")
    score_type = scores.ZScore
    units = pq.mV
    ephysprop_name = 'Spike Amplitude'
    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        model.inject_square_current(self.params['injected_square_current'])
        model.get_membrane_potential()
        #if ascii_plot:
        if model._backend is str("HH"):
            asciplot_code(model.vM,model.get_spike_count()) 
        height = np.max(model.get_membrane_potential())-model.get_membrane_potential().magnitude[-1]
        prediction = {'value': height[0],
                      'mean':np.mean(height) if len(height) else None,
                      'std': np.std(height) if len(height) else None,
                      'n': len(height)}



        # useful to retain inside object.
        self.prediction = prediction
        # Put prediction in a form that compute_score() can use.
        return prediction

    def extract_features(self, model):
        prediction = self.generate_prediction(model)
        return prediction

    def compute_score(self, observation, prediction):
        """Implementat sciunit.Test.score_prediction."""
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(TotalAPAmplitudeTest, self).compute_score(observation,
                                                               prediction)
        return score



class APAmplitudeTest(VmTest):
    """
    def __init__(self, *args, **kwargs):
        super(APAmplitudeTest, self).__init__(*args,**kwargs)#*args, **kwargs)
        if str('params') in kwargs.keys():
            self.params = kwargs['params']

    Test the heights (peak amplitude) of action potentials."""

    required_capabilities = (ncap.ProducesActionPotentials,)

    name = "AP amplitude test"

    description = ("A test of the amplitude (peak minus threshold) of "
                   "action potentials.")

    score_type = scores.ZScore

    units = pq.mV

    ephysprop_name = 'Spike Amplitude'

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        model.inject_square_current(self.params['injected_square_current'])
        model.get_membrane_potential()
        #if ascii_plot:
        if model._backend is str("HH"):
            asciplot_code(model.vM,model.get_spike_count())
        
            import pdb
            pdb.set_trace()
        height = np.max(model.get_membrane_potential())-model.get_AP_thresholds()
        prediction = {'value': height[0],
                      'mean':np.mean(height) if len(height) else None,
                      'std': np.std(height) if len(height) else None,
                      'n': len(height)}



        # useful to retain inside object.
        self.prediction = prediction

        # Put prediction in a form that compute_score() can use.
        return prediction

    def extract_features(self, model):
        prediction = self.generate_prediction(model)
        return prediction

    def compute_score(self, observation, prediction):
        """Implementat sciunit.Test.score_prediction."""
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APAmplitudeTest, self).compute_score(observation,
                                                               prediction)
        return score


class InjectedCurrentAPAmplitudeTest(InjectedCurrent, APAmplitudeTest):
    """Test the heights (peak amplitude) of action potentials.
    Uses current injection.

    """
    required_capabilities = (ncap.ReceivesSquareCurrent,)

    name = "Injected current AP amplitude test"

    description = ("A test of the heights (peak amplitudes) of "
                   "action potentials when current "
                   "is injected into cell.")

    def generate_prediction(self, model):
        model.inject_square_current(self.params['injected_square_current'])
        model.get_membrane_potential()
        if ascii_plot:
            asciplot_code(model.vM,model.get_spike_count())

        prediction = super(InjectedCurrentAPAmplitudeTest, self).\
            generate_prediction(model)
        # useful to retain inside object.
        self.prediction = prediction

        return prediction

    def extract_features(self, model):
        prediction = self.generate_prediction(model)
        self.prediction = prediction

        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        self.prediction = prediction

        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(InjectedCurrentAPAmplitudeTest, self).compute_score(observation,
                                                               prediction)
        return score



class APThresholdTest(VmTest):
    """Test the full widths of action potentials at their half-maximum."""

    required_capabilities = (ncap.ProducesActionPotentials,)

    name = "AP threshold test"

    description = ("A test of the membrane potential threshold at which "
                   "action potentials are produced.")

    score_type = scores.ZScore

    units = pq.mV

    ephysprop_name = 'Spike Threshold'

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        model.inject_square_current(self.params['injected_square_current'])
        model.get_membrane_potential()


        if ascii_plot:
            asciplot_code(model.vM,model.get_spike_count())

        try:
            threshes = model.get_AP_thresholds()
        except:
            threshes = None
        if type(threshes) is not type(None):
            prediction = {'mean': np.mean(threshes) if len(threshes) else None,
                      'std': np.std(threshes) if len(threshes) else None,
                      'n': len(threshes)}
        else:
            prediction = {'mean':  None,
                          'std':  None,
                          'n': 0 }
        # useful to retain inside object.
        self.prediction = prediction

        return prediction

    def extract_features(self, model):
        prediction = self.generate_prediction(model)
        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APThresholdTest, self).compute_score(observation,
                                                               prediction)
        return score


class InjectedCurrentAPThresholdTest(InjectedCurrent, APThresholdTest):
    """Test the thresholds of action potentials under current injection."""

    name = "Injected current AP threshold test"

    description = ("A test of the membrane potential threshold at which "
                   "action potentials are produced under current injection.")

    def generate_prediction(self, model):
        if 'injected_square_current' in self.params.keys():
            model.inject_square_current(self.params['injected_square_current'])

        else:
            model.inject_square_current(self.params)

        model.get_membrane_potential()
        if ascii_plot:
            asciplot_code(model.vM,model.get_spike_count())


        # useful to retain inside object.
        prediction =  super(InjectedCurrentAPThresholdTest, self).\
            generate_prediction(model)

        self.prediction = prediction

        return prediction

    def extract_features(self, model):
        prediction = self.generate_prediction(model)
        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(InjectedCurrentAPThresholdTest, self).compute_score(observation,
                                                               prediction)
        return score
