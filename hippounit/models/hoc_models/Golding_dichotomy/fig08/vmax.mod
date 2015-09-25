NEURON {
    SUFFIX vmax
    RANGE vm
}

ASSIGNED {
       v (millivolt)
       vm (millivolt)
}

INITIAL {
    vm = v
}

BREAKPOINT { 
   if (v>vm) { vm=v }
}
