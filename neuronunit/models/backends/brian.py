
class brianBackend(Backend):
    """Used for generation of code for PyNN, with simulation using NEURON"""

    backend = 'brian'
    try:
        from brian2.library.IF import Izhikevich, ms
        eqs=Izhikevich(a=0.02/ms,b=0.2/ms)
        print(eqs)

    except:
        import os
        os.system('pip install brian2')
        #from brian2.library.IF import Izhikevich, ms
        #eqs=Izhikevich(a=0.02/ms,b=0.2/ms)
        #print(eqs)
