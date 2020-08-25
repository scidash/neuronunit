using NSGAII
using vOptGeneric, GLPKMathProgInterface
using Random: bitrand
using LinearAlgebra: dot
using PyPlot

function plot_pop(P)
    clf()
    non_dom = filter(x ->x.rank == 1, P)
    feasible = filter(x -> x.CV == 0 && x.rank > 1, P)
    unfeasible = filter(x -> x.CV > 0, P)
    p = plot3D(map(x -> x.y[1], feasible), map(x -> x.y[2], feasible), map(x -> x.y[3], feasible), "bo", markersize=1)
    p = plot3D(map(x -> x.y[1], unfeasible), map(x -> x.y[2], unfeasible), map(x -> x.y[3], unfeasible), "rx", markersize=3)
    p = plot3D(map(x -> x.y[1], non_dom), map(x -> x.y[2], non_dom), map(x -> x.y[3], non_dom), "go", markersize=1)
    ax = gca()
    ax[:set_xlim]([-95, 110])
    ax[:set_ylim]([-60, 45])
    ax[:set_zlim]([-50, 110])
    show()
    sleep(0.1)
end


m = vModel(solver = GLPKSolverMIP())

@variable(m, 0 <=x[1:5] <= 10)
@variable(m, 0 <= δ[1:3] <= 1, Int)

@addobjective(m, Max, dot([17,-12,-12,-19,-6], x) + dot([-73, -99, -81], δ))
@addobjective(m, Max, dot([2,-6,0,-12,13], x) + dot([-61,-79,-53], δ))
@addobjective(m, Max, dot([-20,7,-16,0,-1], x) + dot([-72,-54,-79], δ))

@constraint(m, sum(δ) <= 1)
@constraint(m, -x[2] + 6x[5] + 25δ[1] <= 52)
@constraint(m, -x[1] + 18x[4] + 18x[5] + 8δ[2] <= 77)
@constraint(m, 7x[4] + 9x[5] + 19δ[3] <= 66)
@constraint(m, 16x[1] + 20x[5] <= 86)
@constraint(m, 13x[2] + 7x[4] <= 86)


#print(m)

#res = 
nsga(500, 50, m, fplot=plot_pop);

solve(m, method=:lex)
println("\nRésolution lexico-graphique : ")
for i = 1:2:6
    println("x = $(getvalue(x, i)) , δ = $(getvalue(δ, i))")
    println("z = $(getY_N(m)[i])")
end

println()
println("Meilleur individu sur le premier objectif")
x1 = sort(res, by = ind -> ind.y[1])[end];
println("x = $(x1.pheno[1:5]) , δ = $(x1.pheno[6:8])")
println("z = $(x1.y)")
println("CV : $(x1.CV)")

println("Meilleur individu sur le deuxième objectif")
x2 = sort(res, by = ind -> ind.y[2])[end];
println("x = $(x2.pheno[1:5]) , δ = $(x2.pheno[6:8])")
println("z = $(x2.y)")
println("CV : $(x2.CV)")

println("Meilleur individu sur le troisième objectif")
x3 = sort(res, by = ind -> ind.y[3])[end];
println("x = $(x3.pheno[1:5]) , δ = $(x3.pheno[6:8])")
println("z = $(x3.y)")
println("CV : $(x3.CV)")




println("\n Résolution en partant des solutions lex-optimales")

seed = [vcat(getvalue(x, i), getvalue(δ, i)) for i=1:2:6]

res = nsga(500, 50, m, fplot=plot_pop, seed=seed);

println()
println("Meilleur individu sur le premier objectif")
x1 = sort(res, by = ind -> ind.y[1])[end];
println("x = $(x1.pheno[1:5]) , δ = $(x1.pheno[6:8])")
println("z = $(x1.y)")
println("CV : $(x1.CV)")

println("Meilleur individu sur le deuxième objectif")
x2 = sort(res, by = ind -> ind.y[2])[end];
println("x = $(x2.pheno[1:5]) , δ = $(x2.pheno[6:8])")
println("z = $(x2.y)")
println("CV : $(x2.CV)")

println("Meilleur individu sur le troisième objectif")
x3 = sort(res, by = ind -> ind.y[3])[end];
println("x = $(x3.pheno[1:5]) , δ = $(x3.pheno[6:8])")
println("z = $(x3.y)")
println("CV : $(x3.CV)")
