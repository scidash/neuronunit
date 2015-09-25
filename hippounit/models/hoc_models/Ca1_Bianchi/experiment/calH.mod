TITLE Ca L-type channel with high treshold of activation
: inserted in distal dendrites to account for distally
: restricted initiation of Ca++ spikes
: uses channel conductance (not permeability)
: written by Yiota Poirazi, 1/8/00 poirazi@LNC.usc.edu

NEURON {
	SUFFIX calH
	USEION ca READ eca WRITE ica
        RANGE gcalbar, m, h
	RANGE inf, fac, tau
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}



PARAMETER {          : parameters that can be entered when function is called in cell-setup
        v               (mV)
        celsius = 34	(degC)
	dt              (ms)
        gcalbar = 0     (mho/cm2) : initialized conductance
	eca = 140       (mV)      : Ca++ reversal potential
        }

STATE {	m h }                     : unknown activation and inactivation parameters to be solved in the DEs  

ASSIGNED {
	ica (mA/cm2)
      inf[2]
	tau[2]

        
}


INITIAL {
      m = 0    : initial activation parameter value
	h = 1    : initial inactivation parameter value
	rate(v)
	
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	ica = gcalbar*m*m*m*h*(v - eca)

}



DERIVATIVE state {  
        rate(v)
        m' = (inf[0]-m)/tau[0]
	  h' = (inf[1]-h)/tau[1]

}

PROCEDURE rate(v (mV)) { :callable from hoc
       FROM i=0 TO 1 {
		tau[i] = vartau(v,i)
		inf[i] = varss(v,i)
	}

     
	
}


FUNCTION varss(v, i) {
	if (i==0) { 
             varss = 1 / (1 + exp((v+37)/(-1)))  : Ca activation 
	}
	else if (i==1) { 
             varss = 1 / (1 + exp((v+41)/(0.5))) : Ca inactivation 
	}
}

FUNCTION vartau(v, i) {
	if (i==0) {
          vartau = 3.6  : activation variable time constant
         

        }
	else if (i==1) {
:           vartau = 25   : inactivation variable time constant
           vartau = 29   : inactivation variable time constant
        }
}	

















