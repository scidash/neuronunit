"""Waveform neuronunit tests, e.g. testing AP waveform properties"""
from .base import np, pq, ncap, VmTest, scores, AMPL, DELAY, DURATION
try:
    import asciiplotlib as apl
    fig = apl.figure()
    fig.plot([0,1], [0,1], label=str('spikes: ')+str(self.n_spikes), width=100, height=20)
    fig.show()
    ascii_plot = True
except:
    ascii_plot = False
#import numpy

class APWidthTest(VmTest):
    """Test the full widths of action potentials at their half-maximum."""

    required_capabilities = (ncap.ProducesActionPotentials,)
    name = "AP width test"
    description = ("A test of the widths of action potentials "
                   "at half of their maximum height.")
    score_type = scores.RatioScore
    units = pq.ms
    ephysprop_name = 'Spike Half-Width'
    def __init__():
        super(VmTest,self).__init__(**args,**kwargs)
        self.verbose = False
    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        # if get_spike_count is zero, then widths will be None
        # len of None returns an exception that is not handled
        model.inject_square_current(self.params['injected_square_current'])
        if self.verbose:
            print(self.params['injected_square_current'])
        model.get_membrane_potential()
        widths = model.get_AP_widths()
        if ascii_plot:
            t = [float(f) for f in model.vM.times]
            v = [float(f) for f in model.vM.magnitude]

            fig = apl.figure()

            fig.plot(t, v, label=str('spikes: ')+str(model.get_spike_count()), width=100, height=20)
            fig.show()
        try:
            prediction = {'mean': np.mean(widths) if len(widths) else None,
                      'std': np.std(widths) if len(widths) else None,
                      'n': len(widths)}
        except:
            prediction = None
        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if isinstance(prediction, type(None)):
            score = scores.InsufficientDataScore(None)
        elif prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APWidthTest, self).compute_score(observation,
                                                           prediction)
        return score


class InjectedCurrentAPWidthTest(APWidthTest):
    """Tests the full widths of APs at their half-maximum
    under current injection.
    """

    def __init__(self, *args, **kwargs):
        super(InjectedCurrentAPWidthTest, self).__init__(**args,**kwargs)#*args, **kwargs)
        if str('params') in kwargs.keys():
            self.params = kwargs['params']

        #self.params['injected_square_current'] = {'amplitude': 100.0*pq.pA,
        #                                          'delay': DELAY,
        #                                          'duration': DURATION}

    required_capabilities = (ncap.ReceivesSquareCurrent,)
    score_type = scores.ZScore
    units = pq.ms
    name = "Injected current AP width test"
    description = ("A test of the widths of action potentials "
                   "at half of their maximum height when current "
                   "is injected into cell.")


    def generate_prediction(self, model):
        if self.verbose:
            print(self.params['injected_square_current'])

        model.inject_square_current(self.params['injected_square_current'])
        model.get_membrane_potential()
        if ascii_plot:
            t = [float(f) for f in model.vM.times]
            v = [float(f) for f in model.vM.magnitude]

            fig = apl.figure()

            fig.plot(t, v, label=str('spikes: ')+str(model.get_spike_count()), width=100, height=20)
            fig.show()

        prediction = super(InjectedCurrentAPWidthTest, self).\
            generate_prediction(model)

        return prediction


class APAmplitudeTest(VmTest):
    """Test the heights (peak amplitude) of action potentials."""

    required_capabilities = (ncap.ProducesActionPotentials,)

    name = "AP amplitude test"

    description = ("A test of the amplitude (peak minus threshold) of "
                   "action potentials.")


    score_type = scores.ZScore

    #
    units = pq.mV

    ephysprop_name = 'Spike Amplitude'

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.

        model.inject_square_current(self.params['injected_square_current'])
        model.get_membrane_potential()
        if ascii_plot:
            t = [float(f) for f in model.vM.times]
            v = [float(f) for f in model.vM.magnitude]

            fig = apl.figure()

            fig.plot(t, v, label=str('spikes: ')+str(model.get_spike_count()), width=100, height=20)
            fig.show()
        try:
            height = np.max(model.get_membrane_potential()) -float(np.min(model.get_membrane_potential()))/1000.0*model.get_membrane_potential().units #- model.get_AP_thresholds()
            prediction = {'mean':height, 'n':1, 'std':height}

        except:
            prediction = {'mean': None,
                          'std': None,
                          'n': 0}


        # Put prediction in a form that compute_score() can use.
        return prediction

    def compute_score(self, observation, prediction):
        """Implementat sciunit.Test.score_prediction."""
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APAmplitudeTest, self).compute_score(observation,
                                                               prediction)
        return score


class InjectedCurrentAPAmplitudeTest(APAmplitudeTest):
    """Test the heights (peak amplitude) of action potentials.

    Uses current injection.
    """

    def __init__(self):# *args, **kwargs):
        super(InjectedCurrentAPAmplitudeTest, self).__init__()#*args, **kwargs)
        if hasattr(self,'params'):# in .keys():
            if self.verbose:
                print(self.params)
            #self.params = kwargs['params']

        #self.params['injected_square_current'] = {'amplitude': 100.0*pq.pA,
        #                                          'delay': DELAY,
        #                                              'duration': DURATION}

    required_capabilities = (ncap.ReceivesSquareCurrent,)

    name = "Injected current AP amplitude test"

    description = ("A test of the heights (peak amplitudes) of "
                   "action potentials when current "
                   "is injected into cell.")

    def generate_prediction(self, model):
        model.inject_square_current(self.params['injected_square_current'])
        model.get_membrane_potential()
        if ascii_plot:
            t = [float(f) for f in model.vM.times]
            v = [float(f) for f in model.vM.magnitude]

            fig = apl.figure()

            fig.plot(t, v, label=str('spikes: ')+str(model.get_spike_count()), width=100, height=20)
            fig.show()
        prediction = super(InjectedCurrentAPAmplitudeTest, self).\
            generate_prediction(model)
        return prediction
    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
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
            t = [float(f) for f in model.vM.times]
            v = [float(f) for f in model.vM.magnitude]

            fig = apl.figure()

            fig.plot(t, v, label=str('spikes: ')+str(model.get_spike_count()), width=100, height=20)
            fig.show()
        try:
            threshes = model.get_AP_thresholds()
        except:
            threshes = None
        if type(threshes) is not type(None):
            prediction = {'mean': np.mean(threshes) if len(threshes) else None,
                          'std': np.std(threshes) if len(threshes) else None,
                          'n': len(threshes) if len(threshes) else 0 }
        else:
            prediction = {'mean':  None,
                          'std':  None,
                          'n': 0 }

        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APThresholdTest, self).compute_score(observation,
                                                               prediction)
        return score


class InjectedCurrentAPThresholdTest(APThresholdTest):
    """Test the thresholds of action potentials under current injection."""

    def __init__(self):#, *args, **kwargs):
        super(InjectedCurrentAPThresholdTest, self).__init__()#*args, **kwargs)
        if str('params') in kwargs.keys():
            self.params = kwargs['params']


    required_capabilities = (ncap.ReceivesSquareCurrent,)
    name = "Injected current AP threshold test"
    description = ("A test of the membrane potential threshold at which "
                   "action potentials are produced under current injection.")

    #def generate_prediction(self, model):
    #    model.inject_square_current(self.params['injected_square_current'])

    def generate_prediction(self, model):
        if 'injected_square_current' in self.params.keys():
            model.inject_square_current(self.params['injected_square_current'])

        else:
            model.inject_square_current(self.params)

        model.get_membrane_potential()
        if ascii_plot:
            t = [float(f) for f in model.vM.times]
            v = [float(f) for f in model.vM.magnitude]

            fig = apl.figure()

            fig.plot(t, v, label=str('spikes: ')+str(model.get_spike_count()), width=100, height=20)
            fig.show()
        return super(InjectedCurrentAPThresholdTest, self).\
            generate_prediction(model)

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(InjectedCurrentAPThresholdTest, self).compute_score(observation,
                                                               prediction)
        return score
