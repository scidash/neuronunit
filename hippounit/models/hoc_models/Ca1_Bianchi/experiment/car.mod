TITLE Ca R-type channel with medium threshold for activation
: used in distal dendritic regions, together with calH.mod, to help
: the generation of Ca++ spikes in these regions
: uses channel conductance (not permeability)
: written by Yiota Poirazi on 11/13/00 poirazi@LNC.usc.edu
:
: updated to use CVode by Carl Gold 08/10/03
:  Updated by Maria Markaki  03/12/03

NEURON {
	SUFFIX car
	USEION ca READ cai, cao WRITE ica
:	USEION Ca WRITE iCa VALENCE 2
        RANGE gcabar, m, h,ica
	RANGE inf, fac, tau
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(molar) = (1/liter)
	(mM) =	(millimolar)
	FARADAY = (faraday) (coulomb)
	R = (k-mole) (joule/degC)
}


ASSIGNED {               : parameters needed to solve DE
	ica (mA/cm2)
:	iCa (mA/cm2)
        inf[2]
	tau[2]		(ms)
        v               (mV)
        celsius 	(degC)
	   
	cai             (mM)      : initial internal Ca++ concentration
	cao             (mM)      : initial external Ca++ concentration
}


PARAMETER {              : parameters that can be entered when function is called in cell-setup
        gcabar = 0      (mho/cm2) : initialized conductance
        eca = 140       (mV)      : Ca++ reversal potential

       
}  

STATE {	
	m 
	h 
}            : unknown activation and inactivation parameters to be solved in the DEs  


INITIAL {
	rates(v)
        m = 0    : initial activation parameter value
	h = 1    : initial inactivation parameter value
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	:ecar = (1e3) * (R*(celsius+273.15))/(2*FARADAY) * log (cao/cai)
	ica = gcabar*m*m*m*h*(v - eca)

}


DERIVATIVE states {
	rates(v)
	m' = (inf[0]-m)/tau[0]
	h' = (inf[1]-h)/tau[1]
}

PROCEDURE rates(v(mV)) {LOCAL a, b :rest = -70
	FROM i=0 TO 1 {
		tau[i] = vartau(v,i)
		inf[i] = varss(v,i)
	}
}




FUNCTION varss(v(mV), i) {
	if (i==0) {
	    varss = 1 / (1 + exp((v+48.5)/(-3(mV)))) : Ca activation
	}
	else if (i==1) {
             varss = 1/ (1 + exp((v+53)/(1(mV))))    : Ca inactivation
	}
}

FUNCTION vartau(v(mV), i) (ms){
	if (i==0) {
         vartau = 50(ms)  : activation variable time constant
   :       vartau = 120(ms)  : activation variable time constant
        }
	else if (i==1) {
          vartau = 5(ms)   : inactivation variable time constant
     :      vartau = 4(ms)   : inactivation variable time constant
       }
	
}	














