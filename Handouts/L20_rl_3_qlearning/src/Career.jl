module Career

using Distributions
using QuantEcon
using PlotlyBase
using Colors
using Interpolations
using BasisMatrices
using LinearAlgebra
using Parameters

export NealDSDC, NealCSDC, McCallBasic,
       bellman_operator, bellman_operator!, get_greedy,
       vfi, bellman_Q_operator!, bellman_Q_operator, qfi,
       plot_vf, plot_policy, plot_distributions,
       State, Action, EpsilonGreedy, SoftMax,
       Sarsa, Sarsaλ, QLearning, QLearningλ, DoubleQLearning, get_action,
       update!, learn

import Base: ==

# --------- #
# Utilities #
# --------- #

# generate quadrature weights and nodes for known distributions
qnw(d::LogNormal, N::Int=51) = qnwlogn(N, d.μ, d.σ^2)
qnw(d::Normal, N::Int=51) = qnwnorm(N, d.μ, d.σ^2)
qnw(d::Uniform, N::Int=51) = qnwunif(N, d.a, d.b)
qnw(d::Beta, N::Int=51) = qnwbeta(N, d.α, d.β)
qnw(d::Gamma, N::Int=51) = qnwgamma(N, d.α, d.θ)
const QuadDist = Union{LogNormal,Normal,Uniform,Beta,Gamma}

# ------------ #
# DP functions #
# ------------ #

abstract type AbstractModel{NS,NA} end

function bellman_operator(m::AbstractModel{NS}, v::Array{T,NS}=v_init(m);
                                ret_policy=false) where T where NS
    out = similar(v)
    bellman_operator!(m, v, out, ret_policy=ret_policy)
    return out
end

bellman_Q_operator(m::AbstractModel, q::Array=q_init(m)) =
    bellman_Q_operator!(m, q, similar(q))

get_greedy!(m::AbstractModel{NS}, v::Array{T,NS}, out::Array{T,NS}) where T where NS =
    bellman_operator!(m, v, out, ret_policy=true)

get_greedy(m::AbstractModel{NS}, v::Array{T,NS})  where T where NS =
    bellman_operator(m, v, ret_policy=true)

function vfi(wp::AbstractModel; verbose::Bool=false,
             max_iter::Int=5000, err_tol=1e-8)
    v = v_init(wp)
    func(x) = bellman_operator(wp, x)
    compute_fixed_point(func, v; verbose=verbose, max_iter=max_iter,
                        err_tol=err_tol)
end

function qfi(m::AbstractModel; verbose::Bool=false,
             max_iter::Int=5000, err_tol=1e-8)
    q = q_init(m)
    func(x) = bellman_Q_operator(m, x)
    compute_fixed_point(func, q; verbose=verbose, max_iter=max_iter,
                        err_tol=err_tol)
end

include("neal.jl")

end  # module
