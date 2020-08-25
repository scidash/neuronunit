mutable struct indiv{X, N, Y}

    x::X
    y::NTuple{N, Y}
    normalized_y::Vector{Float64}
    rank::Int
    CV::Float64
    dom_count::Int
    dom_list::Vector{indiv{X,N,Y}}
    ref_point::Int
    distance_ref::Float64

    indiv(x::X, y::NTuple{N, Y}, cv) where {X,N,Y} = new{X, N, Y}(x, y, zeros(N), 0, cv, 0, indiv{X,N,Y}[], 0, 0.)
end
indiv(x, z::Function, fCV::Function) = indiv(x, z(x), fCV(x))


function dominates(a::indiv, b::indiv)
    a.CV != b.CV && return a.CV < b.CV

    res = false
    for i in eachindex(a.y)
        a.y[i] > b.y[i] && return false
        a.y[i] < b.y[i] && (res=true)
    end
    res
end
⋖(a::indiv, b::indiv) = dominates(a, b)

import Base.==; ==(a::indiv, b::indiv) = a.x == b.x
import Base.hash; hash(a::indiv) = hash(a.x)
import Base.show; show(io::IO, ind::indiv) = print(io, "ind($(ind.x) : $(ind.y) | rank : $(ind.rank))")

function show(io::IO, ind::indiv{X, Y}) where {X<:AbstractArray{Bool}, Y}
    genotype = String([xx ? '1' : '0' for xx in ind.x])
    if VERSION >= v"0.7-"
        print(io, "ind(") ; show(IOContext(io, :limit=>true, :compact=>true), genotype) ; print(io, " : $(ind.y) | rank : $(ind.rank))")
    else
        print(io, "ind(") ; show(IOContext(io, limit=true, compact=true), genotype) ; print(io, " : $(ind.y) | rank : $(ind.rank))")
    end
end 

function eval!(indiv::indiv, z::Function, fCV::Function)
    indiv.y = z(indiv.x)
    indiv.CV = fCV(indiv.x)
    indiv
end