
import julia
from julia import Main
j = jj = julia.Julia()
j.eval("using SpikingNeuralNetworks")
j.eval("using NSGAIII")
Main.include("julia_run.jl")

#print(Main.varinfo())
#print(MU,NGEN)
j.eval("MU = 3")
j.eval("NGEN = 1")

jj.eval("""
function zzz(in_genes)
    jDOOM=py"DO.OM"
    fitnesses,pop,dtcpop = jDOOM.update_deap_pop(in_genes,jDOOM.tests,jDOOM.td)
    @show (fitnesses)
    @show (dtcpop)
    @show (pop)

    return pop
end
""")
j.eval("nsga(MU, NGEN, init, zzz)")#, fplot = plot_pop)")
#j.eval("using JLD")

#Main.inject_square_current(current)
