#-*- coding: utf-8 -*-
from __future__ import (print_function, division,
                        absolute_import, unicode_literals)
import numpy as np
from scipy import integrate as spint
from matplotlib import pyplot as plt



from numba import jit

class FNNeuron(Neuron):
    """FitzHugh-Naguno neuron.
    The units in this model are different from the HH ones.
    Sources:
    https://en.wikipedia.org/w/index.php?title=FitzHugh%E2%80%93Nagumo_model&oldid=828788626
    http://www.scholarpedia.org/article/FitzHugh-Nagumo_model
    """
    # TODO: add description of the parameters
    def __init__(self, I_ampl=0.85, V_0=-0.7, W_0=-0.5, a=0.7, b=0.8,
                 tau=12.5, neurondict=dict()):
        Neuron.__init__(self, I_ampl=I_ampl, **neurondict)

        # Store intial conditions
        self.V_0 = V_0
        self.W_0 = W_0

        # Store model parameters
        self.a = a
        self.b = b
        self.tau = tau

        # Units
        self.time_unit = ""
        self.V_unit = ""
        self.I_unit = ""


    def V_ddt(self, V, W, I_ext):
        """Time derivative of the potential V.
        """
        timederivative = V - np.power(V, 3)/3. - W + I_ext
        return timederivative

    def W_ddt(self, V, W):
        """Time derivative of the recovery variable W.
        """
        timederivative = (V + self.a - self.b*W)/self.tau
        return timederivative



    def _rhs(self, y, t):

        V = y[0]
        W = y[1]
        output = np.array((self.V_ddt(V, W, self.I_ext(t)),
                           self.W_ddt(V, W)))
        return output


    def solve(self, ts=None):
        """Integrate the differential equations of the system.
        The integration is made using an Euler algorithm and
        the method I_ext() is used to modelize the external current.
        Parameters
        ----------
            ts : array
                Times were the solution value is stored.

        Returns
        -------
            Vs : array
                Membrane potential at the given times.
        """
        # Simulation times
        if ts is None:
            self.ts = np.linspace(0, 1000, 1000)
        else:
            self.ts = ts

        y0 = np.array((self.V_0, self.W_0))
        sol = spint.odeint(self._rhs, y0, self.ts)
        # solve_ivp returns a lot of extra information about the solutions, but
        # we are only interested in the values of the variables, which are stored
        # in sol.y
        self.Vs = sol[:,0]
        self.Ws = sol[:,1]

        return Vs


class LinearIFNeuron(Neuron):
    """Linear integrate-and-fire neuron.
    Sources:
        http://icwww.epfl.ch/~gerstner/SPNM/node26.html
        http://www.ugr.es/~jtorres/Tema_4_redes_de_neuronas.pdf (spanish)
    """
    def __init__(
            self, I_ampl=10, V_0=-80, R=0.8, V_thr=-68.5, V_fire=20,
            V_relax=-80, relaxtime=5, firetime=2, neurondict=dict()):
        """Init method.
        Parameters
        ----------
            I_ampl : float
                External current.
            V_0 : float
                Initial value of the membrane potential.
            R : float
                Model parameter (see references).

            V_thr : float
                Voltage firing thresold.
            V_fire : float
                Voltaje firing value.
            v_relax : float
                Voltage during the relax time after the firing.
            relaxtime : float
                Relax time after firing
            firetime : float
                Fire duration.
        """
        Neuron.__init__(self, I_ampl=I_ampl, **neurondict)

        # Store initial condition
        self.V_0 = V_0

        # Store parameters
        self.R = R  # k ohmn/cm2
        self.V_thr = V_thr  # Fire threshold
        self.V_fire = V_fire  # Firing voltage
        self.V_relax = V_relax  # Firing voltage
        self.relaxtime = relaxtime  # Relax time after firing
        self.firetime = firetime  # Fire duration

        # Units
        self.I_unit = "(microA/cm2)"
        self.time_unit = "(ms)"
        self.V_unit = "(mV)"


    def fire_condition(self, V):
        """Return True if the fire condition is satisfied.
        """

    def V_ddt(self, V, I_ext):
        """Time derivative of the membrane potential.
        """
        timederivative = (-(V + 65)/self.R + I_ext)/self.C
        return timederivative


    def solve(self, ts=None, timestep=0.1):
        """Integrate the differential equations of the system.

        The integration is made using an Euler algorithm and
        the method I_ext() is used to modelize the external current.
        Parameters
        ----------
            ts : array
                Times were the solution value is stored.
        Returns
        -------
            Vs : array
                Membrane potential at the given times.
        """
        # Initialization
        t_last = 0.  # Time of the last measure
        V = self.V_0  # Present voltage

        # Create array to store the measured voltages
        Vs = np.zeros(ts.size, dtype=float)

        # Check the firing condition.
        # _neuronstate stores the state of the neuron.
        # If it is firing _neuronstate=1, if relaxing it equals 2, else
        # it is 0.
        self._neuronstate = int(V > self.V_thr)
        if self._neuronstate == 1:
            self._t_endfire = t_last + self.firetime

        for j_measure, t_measure in enumerate(ts):
            # Calculate the number of steps before the next measure
            nsteps = int((t_measure - t_last)/timestep)
            t = t_last

            for j_step in range(nsteps):
                if self._neuronstate == 0:
                    # Advance time step
                    V += self._rhs(t_last, V)*timestep

                    # Check if the firing condition is met
                    self._neuronstate = int(V > self.V_thr)
                    if self._neuronstate == 1:
                        self._t_endfire = t + self.firetime

                # Firing
                elif self._neuronstate == 1:
                    V = self.V_fire

                    # Check if the firing has ended
                    if t > self._t_endfire:
                        self._neuronstate = 2
                        self._t_endrelax = t + self.relaxtime

                # Relaxing
                elif self._neuronstate == 2:
                    V = self.V_relax

                    # Check if the relaxing time has ended
                    if t > self._t_endrelax:
                        self._neuronstate = 0

                # Update time
                t += timestep

            # Measure
            Vs[j_measure] = V
            t_last = t_measure

        return Vs


    def _rhs(self, t, y):
        """Right hand side of the system of equations to be solved.
        This functions is necessary to use scipy.integrate.
        Parameters
        ----------
            y : float
                Array with the present state of the variable
                which time derivative is to be solved, V.
            t : float
                Time variable.
        Returns
        -------
            timederivative : float
                Time derivatives of the variable.

        """
        V = y
        output = self.V_ddt(V, self.I_ext(t))
        return output


#    def solve(self, ts=None):
#        """Integrate the differential equations of the system.
#
#        """
#        # Simulation times
#        if ts is None:
#            self.ts = np.linspace(0, 1000, 1000)
#        else:
#            self.ts = ts
#
#        y0 = self.V_0
#        sol = spint.odeint(self._rhs, y0, self.ts)
#        # solve_ivp returns a lot of extra information about the solutions, but
#        # we are only interested in the values of the variables, which are stored
#        # in sol.y
#        self.Vs = sol[:,0]
#
#        return
