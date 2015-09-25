TITLE  H-current that uses Na ions

NEURON {
	SUFFIX h
        RANGE  gbar,vhalf, K, taun, ninf, g  
	USEION na READ ena WRITE ina      
:	NONSPECIFIC_CURRENT i
}

UNITS {
	(um) = (micrometer)
	(mA) = (milliamp)
	(uA) = (microamp)
	(mV) = (millivolt)
	(pmho) = (picomho)
	(mmho) = (millimho)
}



PARAMETER {              : parameters that can be entered when function is called in cell-setup
        dt             (ms)
	v              (mV)
        ena    = 50    (mV)
        eh     = -10   (mV)
	K      = 8.5   (mV)
	gbar   = 0     (mho/cm2)  : initialize conductance to zero
	:vhalf  = -90   (mV)       : half potential
      vhalf  = -81   (mV)       : half potential
     

}	


STATE {                : the unknown parameters to be solved in the DEs
	n
}

ASSIGNED {             : parameters needed to solve DE
	ina (mA/cm2)
	ninf
	taun (ms)
	g
}

        


INITIAL {               : initialize the following parameter using states()
	states()	
	n = ninf
	g = gbar*n
	ina = g*(v-eh)
}


BREAKPOINT {
	SOLVE h METHOD derivimplicit
	g = gbar*n
	ina = g*(v-eh)  
}

DERIVATIVE h {
	states()
        n' = (ninf - n)/taun
}

PROCEDURE states() {  
 
 	if (v > -30) {
	   taun = 1
	} else {
           :taun = 2*(1/(exp((v+145)/-17.5)+exp((v+16.8)/16.5)) + 5) :h activation tau
           taun = 5*(1/(exp((v+145)/-17.5)+exp((v+16.8)/16.5)) + 5) :h activation tau


	}  
         ninf = 1 - (1 / (1 + exp((vhalf - v)/K)))                  :steady state value
}



