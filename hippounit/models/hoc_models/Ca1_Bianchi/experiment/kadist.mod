TITLE K-A channel from Klee Ficker and Heinemann
: modified by Brannon and Yiota Poirazi (poirazi@LNC.usc.edu) 
: to account for Hoffman et al 1997 distal region kinetics
: used only in locations > 100 microns from the soma
:
: modified to work with CVode by Carl Gold, 8/10/03
:  Updated by Maria Markaki  12/02/03

NEURON {
	SUFFIX kad
	USEION k READ ek WRITE ik
        RANGE gkabar,gka,ik
        GLOBAL ninf,linf,taul,taun,lmin
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}


PARAMETER {    :parameters that can be entered when function is called in cell-setup   
:	gkabar = 0.008  (mho/cm2)  :suggested conductance value
	gkabar = 0      (mho/cm2)  :initialized conductance
        vhalfn = -1     (mV)       :activation half-potential
        vhalfl = -56    (mV)       :inactivation half-potential
       a0n = 0.1       (/ms)      :parameters used
       : a0l = 0.05       (/ms)      :parameters used
        zetan = -1.8    (1)        :in calculation of
        zetal = 3       (1)        :steady state values
        gmn   = 0.39    (1)        :and time constants
        gml   = 1       (1)
	lmin  = 2       (ms)
	nmin  = 0.1     (ms)
:	nmin  = 0.2     (ms)	:suggested
	pw    = -1      (1)
	tq    = -40     (mV)
	qq    = 5       (mV)
	q10   = 5                :temperature sensitivity
}


ASSIGNED {    :parameters needed to solve DE
	v               (mV)
        ek              (mV)
	celsius  	(degC)
	ik              (mA/cm2)
        ninf
        linf      
        taul            (ms)
        taun            (ms)
        gka             (mho/cm2)
}


STATE {       :the unknown parameters to be solved in the DEs 
	n l
}

: Solve qt once in initial block
LOCAL qt

INITIAL {    :initialize the following parameter using rates()
      rates(v)
	n=ninf
	l=linf
}

BREAKPOINT {
	SOLVE states METHOD cnexp
      gka = gkabar*n*l
	ik = gka*(v-ek)
}


DERIVATIVE states {     : exact when v held constant; integrates over dt step
        rates(v)          : do this here
        n' = (ninf - n)/taun
        l' = (linf - l)/taul
}



PROCEDURE rates(v (mV)) {		 :callable from hoc
	LOCAL a,qt
        qt = q10^((celsius-24)/10)       : temprature adjastment factor
        a = alpn(v)
        ninf = 1/(1 + a)		 : activation variable steady state value
        taun = betn(v)/(qt*a0n*(1+a))	 : activation variable time constant
	if (taun<nmin) {taun=nmin}	 : time constant not allowed to be less than nmin

        a = alpl(v)
        linf = 1/(1+ a)                  : inactivation variable steady state value
	taul = 0.26(ms/mV)*(v+50)               : inactivation variable time constant
	if (taul<lmin) {taul=lmin}       : time constant not allowed to be less than lmin
}


FUNCTION alpn(v(mV)) { LOCAL zeta
  zeta = zetan+pw/(1+exp((v-tq)/qq))
UNITSOFF
  alpn = exp(1.e-3*zeta*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) 
UNITSON
}

FUNCTION betn(v(mV)) { LOCAL zeta
  zeta = zetan+pw/(1+exp((v-tq)/qq))
UNITSOFF
  betn = exp(1.e-3*zeta*gmn*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) 
UNITSON
}

FUNCTION alpl(v(mV)) {
UNITSOFF
  alpl = exp(1.e-3*zetal*(v-vhalfl)*9.648e4/(8.315*(273.16+celsius))) 
UNITSON
}

FUNCTION betl(v(mV)) {
UNITSOFF
  betl = exp(1.e-3*zetal*gml*(v-vhalfl)*9.648e4/(8.315*(273.16+celsius))) 
UNITSON
}

