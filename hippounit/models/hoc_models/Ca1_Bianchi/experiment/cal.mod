TITLE l-calcium channel
: l-type calcium channel


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

	FARADAY = 96520 (coul)
	R = 8.3134 (joule/degC)
	KTOMV = .0853 (mV/degC)
}

PARAMETER {
	v (mV)
	celsius= 34	(degC)
	gcalbar=0 (mho/cm2)
	ki=.001 (mM)
	cai = 50.e-6 (mM)
	cao = 2 (mM)
     	tfa = 5
      ggk
      eca = 140	
}


NEURON {
	SUFFIX cal
	USEION ca READ cai,cao WRITE ica
        RANGE gcalbar,cai, ica, gcal, ggk
        GLOBAL minf,taum
}

STATE {
	m
}

ASSIGNED {
	ica (mA/cm2)
        gcal (mho/cm2)
        minf
        taum  (ms)
}

INITIAL {
	rate(v)
	m = minf
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	gcal = gcalbar*m*h2(cai)    
	ggk=ghk(v,cai,cao)
	ica = gcal*ggk

}

FUNCTION h2(cai(mM)) {
	h2 = ki/(ki+cai)
}


FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) {
        LOCAL nu,f
        f = KTF(celsius)/2
        nu = v/f
        ghk=-f*(1. - (ci/co)*exp(nu))*efun(nu)
}

FUNCTION KTF(celsius (DegC)) (mV) {
        KTF = ((25./293.15)*(celsius + 273.15))
}


FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}




FUNCTION alpm(v(mV)) {
	TABLE FROM -150 TO 150 WITH 200
	alpm = 0.055*(-27.01 - v)/(exp((-27.01-v)/3.8) - 1)
}


FUNCTION betm(v(mV)) {
        TABLE FROM -150 TO 150 WITH 200
        betm =0.94*exp((-63.01-v)/17)
}



DERIVATIVE state {  
        rate(v)
        m' = (minf - m)/taum
}

PROCEDURE rate(v (mV)) { :callable from hoc
        LOCAL a, b, qt
        a = alpm(v)
        taum = 1/(tfa*(a+betm(v))) : estimation of activation tau
        minf = a/(a+betm(v))       : estimation of activation steady state value

	
}
