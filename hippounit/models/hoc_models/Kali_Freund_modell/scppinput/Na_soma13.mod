TITLE Channel: Na_soma

COMMENT
    Na channel in soma of CA1 pyramid cell
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

    SUFFIX Na_soma13
    USEION na READ ena WRITE ina VALENCE 1  ? reversal potential of ion is read, outgoing current is written
           
        
    RANGE gmax, gion
    
    RANGE Xinf, Xtau
    
    RANGE Yinf, Ytau
    
    RANGE Zinf, Ztau
    
}

PARAMETER { 
      

    gmax = 0.05 (S/cm2)  ? default value, should be overwritten when conductance placed on cell
    
}



ASSIGNED {
      

    v (mV)
    
    celsius (degC)
          

    ? Reversal potential of na
    ena (mV)
    ? The outward flow of ion: na calculated by rate equations...
    ina (mA/cm2)
    
    
    gion (S/cm2)
    Xinf
    Xtau (ms)
    Yinf
    Ytau (ms)
    Zinf
    Ztau (ms)
    
}

BREAKPOINT { 
                        
    SOLVE states METHOD cnexp
         

    gion = gmax*((X)^3)*((Y)^1)*((Z)^1)      

    ina = gion*(v - ena)
            

}



INITIAL {
    
    ena = 55
        
    rates(v)
    X = Xinf
    Y = Yinf
    Z = Zinf
    
}
    
STATE {
    X
    Y
    Z
    
}

DERIVATIVE states {
    rates(v)
    X' = (Xinf - X)/Xtau
    Y' = (Yinf - Y)/Ytau
    Z' = (Zinf - Z)/Ztau
    
}

PROCEDURE rates(v(mV)) {  
    
    ? Note: not all of these may be used, depending on the form of rate equations
    LOCAL  alpha, beta, tau, inf, gamma, zeta, temp_adj_X, A_alpha_X, B_alpha_X, Vhalf_alpha_X, A_beta_X, B_beta_X, Vhalf_beta_X, temp_adj_Y, A_tau_Y, B_tau_Y, Vhalf_tau_Y, A_inf_Y, B_inf_Y, Vhalf_inf_Y, temp_adj_Z, A_alpha_Z, B_alpha_Z, Vhalf_alpha_Z, A_beta_Z, B_beta_Z, Vhalf_beta_Z
        
    TABLE Xinf, Xtau,Yinf, Ytau,Zinf, Ztau
 DEPEND celsius
 FROM -100 TO 50 WITH 3000
    
    
    UNITSOFF
    temp_adj_X = 1
    temp_adj_Y = 1
    temp_adj_Z = 1
    
        
    ?      ***  Adding rate equations for gate: X  ***
        
    ? Found a parameterised form of rate equation for alpha, using expression: A*exp((v-Vhalf)/B)
    A_alpha_X = 20000
    B_alpha_X = 0.01
    Vhalf_alpha_X = -0.03   
    
    ? Unit system in ChannelML file is SI units, therefore need to convert these to NEURON quanities...
    
    A_alpha_X = A_alpha_X * 0.0010   ? 1/ms
    B_alpha_X = B_alpha_X * 1000   ? mV
    Vhalf_alpha_X = Vhalf_alpha_X * 1000   ? mV
          
                     
    alpha = A_alpha_X * exp((v - Vhalf_alpha_X) / B_alpha_X)
    
    
    ? Found a parameterised form of rate equation for beta, using expression: A*exp((v-Vhalf)/B)
    A_beta_X = 20000
    B_beta_X = -0.00818182
    Vhalf_beta_X = -0.03   
    
    ? Unit system in ChannelML file is SI units, therefore need to convert these to NEURON quanities...
    
    A_beta_X = A_beta_X * 0.0010   ? 1/ms
    B_beta_X = B_beta_X * 1000   ? mV
    Vhalf_beta_X = Vhalf_beta_X * 1000   ? mV
          
                     
    beta = A_beta_X * exp((v - Vhalf_beta_X) / B_beta_X)
    
    Xtau = 1/(temp_adj_X*(alpha + beta))
    Xinf = alpha/(alpha + beta)
          
       
    
    ?     *** Finished rate equations for gate: X ***
    

    ?      ***  Adding rate equations for gate: Y  ***
    
    ? Note: Equation (and all ChannelML file values) in SI Units so need to convert v first...
    
    v = v * 0.0010   ? temporarily set v to units of equation...
            
    tau = (1/((300*( exp (0.2*(v + 0.036)/(-0.005)))) + (300*( exp ((0.2 - 1)*(v + 0.036)/(-0.005))))) + 0.0001)

    ? Set correct units of tau for NEURON
    tau = tau * 1000 
    
    Ytau = tau/temp_adj_Y
     
    inf = 1/(1 + exp (-(v + 0.036)/(-0.003)))
    
    v = v * 1000   ? reset v
        
    Yinf = inf
    
    ?     *** Finished rate equations for gate: Y ***

        
    ?      ***  Adding rate equations for gate: Z  ***
    
    ? Note: Equation (and all ChannelML file values) in SI Units so need to convert v first...
    
    v = v * 0.0010   ? temporarily set v to units of equation...
            
    alpha = (1+0.7*( exp ((v+0.03)/0.002)))/(1+( exp ((v+0.03)/0.002)))*(1+(exp (450*(v+0.045))))/(5*(exp (90*(v+0.045))) + 0.002*(exp (450*(v+0.045))))
        
    ? Set correct units of alpha for NEURON
    alpha = alpha * 0.0010 
    
    beta = (1+(exp (450*(v+0.045))))/(5*(exp (90*(v+0.045))) + 0.002*(exp (450*(v+0.045)))) - (1+0.7*( exp ((v+0.03)/0.002)))/(1+( exp ((v+0.03)/0.002)))*(1+(exp (450*(v+0.045))))/(5*(exp (90*(v+0.045))) + 0.002*(exp (450*(v+0.045))))
        
    ? Set correct units of beta for NEURON
    beta = beta * 0.0010 
    
    v = v * 1000   ? reset v
        
    Ztau = 1/(temp_adj_Z*(alpha + beta))
    Zinf = alpha/(alpha + beta)
    
    ?     *** Finished rate equations for gate: Z ***

}


UNITSON

