NEURON {
    SUFFIX vmax2
    RANGE vm
    RANGE vap
    RANGE vt
    RANGE vtm
    RANGE vtl
    RANGE vth
    RANGE tsave
}

PARAMETER {
       vth = 0 (millivolt/ms)
       delay = 1 (ms)
       vtscale = 0.1 (1)
}

ASSIGNED {
       dt (ms)
       v (millivolt)
       vap (millivolt)
       vlast (millivolt)
       vm (millivolt)
       vt (millivolt/ms)
       vtm (millivolt/ms)
       vtl (millivolt/ms)
       tsave (ms)
}

INITIAL {
    vm = v
    vlast = v
    vap = 0 (millivolt)
    vt = 0 (millivolt/ms)
    vtl = 0 (millivolt/ms)
    vtm = 0 (millivolt/ms)
    tsave = 0 (ms)
}

BREAKPOINT { 
   if (v>vm) { vm=v }
   if (dt>0 (ms)) {
      vt=(v-vlast)/dt
      if (vt>vth && vtm<vth) {
            vap=vlast+(v-vlast)*(vth-vtl)/(vt-vtl)
            tsave=t
      }
      if (t>2*delay && vt>vtm) { vtm = vt }
      vtl=vt
      vlast=v
   }
}



