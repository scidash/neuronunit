TITLE Channel: K_DR

COMMENT
    K delayed rectifier channel for hippocampal CA1 pyramidal neurons
ENDCOMMENT


UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
    (um) = (micrometer)
    (molar) = (1/liter)
    (mM) = (millimolar)
    (l) = (liter)
}


    
NEURON {
      

    SUFFIX K_DRS4
    USEION k READ ek WRITE ik VALENCE 1  ? reversal potential of ion is read, outgoing current is written
           
        
    RANGE gmax, gion
    
    RANGE Xinf, Xtau
    
}

PARAMETER { 
      

    gmax = 0.0090 (S/cm2)  ? default value, should be overwritten when conductance placed on cell
    
}



ASSIGNED {
      

    v (mV)
    
    celsius (degC)
          

    ? Reversal potential of k
    ek (mV)
    ? The outward flow of ion: k calculated by rate equations...
    ik (mA/cm2)
    
    
    gion (S/cm2)
    Xinf
    Xtau (ms)
    
}

BREAKPOINT { 
                        
    SOLVE states METHOD cnexp
         

    gion = gmax*((X)^4)      

    ik = gion*(v - ek)
            

}



INITIAL {
    
    ek = -80
        
    rates(v)
    X = Xinf
        
    
}
    
STATE {
    X
    
}

DERIVATIVE states {
    rates(v)
    X' = (Xinf - X)/Xtau
    
}

PROCEDURE rates(v(mV)) {  
    
    LOCAL tau, inf, temp_adj_X
        
    TABLE Xinf, Xtau
 DEPEND celsius
 FROM -100 TO 50 WITH 3000
    
    
    UNITSOFF
    temp_adj_X = 1
    
            
                
           

        
    ?      ***  Adding rate equations for gate: X  ***
        
    ? Note: Equation (and all ChannelML file values) in SI Units so need to convert v first...
    
    v = v * 0.0010   ? temporarily set v to units of equation...
            
    tau = 0.002

    ? Set correct units of tau for NEURON
    tau = tau * 1000 
    
    Xtau = tau/temp_adj_X
     
    inf = 1/(1 + exp (-(v + 0.02)/0.015))
    
    v = v * 1000   ? reset v
        
    Xinf = inf
    
    ?     *** Finished rate equations for gate: X ***
         

}


UNITSON


