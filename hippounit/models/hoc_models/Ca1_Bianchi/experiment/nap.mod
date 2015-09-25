TITLE  Na persistent channel
: used in distal oblique dendrites to assist Ca spike initiation  
: 
: modified to use CVode --Carl Gold 08/12/03
:  Updated by YiotaPoirazi   26/1/05

NEURON {
	SUFFIX nap
	USEION na READ ena WRITE ina
        RANGE  gnabar,vhalf, K, ina

}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

}

PARAMETER {            
	K = 4.5            (1)      : slope of steady state variable
:	gnabar = 0.001e-2 (mho/cm2) : suggested conductance, 1 percent of the transient Na current
	gnabar = 0        (mho/cm2)
	vhalf  = -50.4    (mV)      : half potential
      
}	

ASSIGNED {
	v             (mV)
        ena           (mV)    
	ina           (mA/cm2)
        n_inf
        tau            (ms)
}

STATE { n }

BREAKPOINT {
	SOLVE states METHOD cnexp
	ina = gnabar*n*n*n*(v-ena)
}

INITIAL {
	rates(v)
	n = n_inf
}


DERIVATIVE states {
        rates(v)
        n' = (n_inf-n)/tau
}

PROCEDURE rates(v(mV)) {
	n_inf = 1 / (1 + (exp(vhalf - v)/K))
	tau =1
}



