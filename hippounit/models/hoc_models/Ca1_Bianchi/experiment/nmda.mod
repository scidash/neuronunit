TITLE simple NMDA receptors

COMMENT
-----------------------------------------------------------------------------
     YIOTA GAVE THIS FILE TO ME (JOSE GOMEZ)
Essentially the same as /examples/nrniv/netcon/ampa.mod in the NEURON
distribution - i.e. Alain Destexhe's simple AMPA model - but with
different binding and unbinding rates and with a magnesium block.
Modified by Andrew Davison, The Babraham Institute, May 2000


       Simple model for glutamate AMPA receptors
       =========================================

 - FIRST-ORDER KINETICS, FIT TO WHOLE-CELL RECORDINGS

   Whole-cell recorded postsynaptic currents mediated by AMPA/Kainate
   receptors (Xiang et al., J. Neurophysiol. 71: 2552-2556, 1994) were used
   to estimate the parameters of the present model; the fit was performed
   using a simplex algorithm (see Destexhe et al., J. Computational Neurosci.
   1: 195-230, 1994).

 - SHORT PULSES OF TRANSMITTER (0.3 ms, 0.5 mM)

   The simplified model was obtained from a detailed synaptic model that
   included the release of transmitter in adjacent terminals, its lateral
   diffusion and uptake, and its binding on postsynaptic receptors (Destexhe
   and Sejnowski, 1995).  Short pulses of transmitter with first-order
   kinetics were found to be the best fast alternative to represent the more
   detailed models.

 - ANALYTIC EXPRESSION

   The first-order model can be solved analytically, leading to a very fast
   mechanism for simulating synapses, since no differential equation must be
   solved (see references below).



References

  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  An efficient method for
  computing synaptic conductances based on a kinetic model of receptor binding
  Neural Computation 6: 10-14, 1994.

  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. Synthesis of models for
  excitable membranes, synaptic transmission and neuromodulation using a
  common kinetic formalism, Journal of Computational Neuroscience 1:
  195-230, 1994.

-----------------------------------------------------------------------------
ENDCOMMENT



NEURON {
       POINT_PROCESS NMDA
       RANGE g, Alpha, Beta, e, gmax, ica
       USEION ca WRITE ica
       NONSPECIFIC_CURRENT  iNMDA            : i
       GLOBAL Cdur, mg, Cmax
}
UNITS {
       (nA) = (nanoamp)
       (mV) = (millivolt)
       (umho) = (micromho)
       (mM) = (milli/liter)
}

PARAMETER {
       Cmax    = 1      (mM)           : max transmitter concentration
       Cdur    = 1      (ms)           : transmitter duration (rising phase)
:       Cdur    = 10     (ms)           : transmitter duration (rising phase)
       Alpha   = 2  : 10     (/ms /mM)      : forward (binding) rate :used
:       Alpha   = 4      (/ms /mM)      : forward (binding) rate
:       Beta    = 0.0125 (/ms)          : backward (unbinding) rate :used
       Beta    = 0.027 :0.02 (/ms)            : 0.05 gives 150 ms 2P backward (unbinding) rate
:       e       = 45     (mV)           : reversal potential
       e       = 0      (mV)           : reversal potential
       mg      =1 (mM): 1      (mM)           : external magnesium concentration

}


ASSIGNED {
       v               (mV)            : postsynaptic voltage
       iNMDA           (nA)            : current = g*(v - e)
       g               (umho)          : conductance
       Rinf                            : steady state channels open
       Rtau            (ms)            : time constant of channel binding
       synon
       B
       gmax                              : magnesium block
       ica
}

STATE {Ron Roff}

INITIAL {
       Rinf = Cmax*Alpha / (Cmax*Alpha + Beta)
       Rtau = 1 / (Cmax*Alpha + Beta)
       synon = 0
}

BREAKPOINT {
       SOLVE release METHOD cnexp
       B = mgblock(v)
       g = (Ron + Roff)*1(umho) * B
:g = (Ron + Roff)*1(umho) : No Mg2+ effect

       iNMDA = g*(v - e)
       ica = 7*iNMDA/10   :(5-10 times more permeable to Ca++ than Na+ or K+, Ascher and Nowak, 1988)
:        ica = 0
       iNMDA = 3*iNMDA/10

}

DERIVATIVE release {
       Ron' = (synon*Rinf - Ron)/Rtau
       Roff' = -Beta*Roff
}

FUNCTION mgblock(v(mV)) {
       TABLE
       DEPEND mg
       FROM -140 TO 80 WITH 1000

       : from Jahr & Stevens

       mgblock = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))
}

: following supports both saturation from single input and
: summation from multiple inputs
: if spike occurs during CDur then new off time is t + CDur
: ie. transmitter concatenates but does not summate
: Note: automatic initialization of all reference args to 0 except first


NET_RECEIVE(weight, on, nspike, r0, t0 (ms)) {
       : flag is an implicit argument of NET_RECEIVE and  normally 0
       if (flag == 0) { : a spike, so turn on if not already in a Cdur pulse
               nspike = nspike + 1
               if (!on) {
                       r0 = r0*exp(-Beta*(t - t0))
                       t0 = t
                       on = 1
                       synon = synon + weight
                       state_discontinuity(Ron, Ron + r0)
                       state_discontinuity(Roff, Roff - r0)
               }
:                come again in Cdur with flag = current value of nspike
               net_send(Cdur, nspike)
      }
       if (flag == nspike) { : if this associated with last spike then turn off
               r0 = weight*Rinf + (r0 - weight*Rinf)*exp(-(t - t0)/Rtau)
               t0 = t
               synon = synon - weight
               state_discontinuity(Ron, Ron - r0)
               state_discontinuity(Roff, Roff + r0)
               on = 0
       }
gmax = weight
}
