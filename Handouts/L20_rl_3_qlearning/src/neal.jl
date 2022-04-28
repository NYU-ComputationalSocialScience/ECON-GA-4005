"Reference: quant-econ.net/jl/carrer.html"
abstract type NealModel <: AbstractModel{2,1} end

# ---------------------- #
# Continuous State Model #
# ---------------------- #

# in this variation of the model draws for ϵ and θ are still i.i.d, but
# are now drawn from the a continuous-valued distribution
struct NealCSDC <: NealModel
    β::Float64
    F::Beta
    G::Beta

    # stuff to not re-compute
    F_mean::Float64
    G_mean::Float64

    # grids for discretized version
    θ::Vector{Float64}
    ϵ::Vector{Float64}
    F_probs::Vector{Float64}
    G_probs::Vector{Float64}
end

function plot_distributions(m::NealCSDC)
    Plot([
            bar(m.θ, m.F_probs, name="θ ~ F"),
            bar(m.ϵ, m.G_probs, name="ϵ ~ G"),
    ])
end

function NealCSDC(;β::Real=0.95,
                   F::Beta=Beta(5.0, 5.0),
                   G::Beta=Beta(1.2, 1.2),
                   Nθ::Int=31,
                   Nϵ::Int=31)
    # build 1d quadrature weights/nodes, then combine, then chop low
    # probability events
    θ, F_probs = qnw(F, Nθ)
    ϵ, G_probs = qnw(G, Nϵ)

    NealCSDC(β, F, G, dot(θ, F_probs), dot(ϵ, G_probs), θ, ϵ, F_probs, G_probs)
end

# NOTE: In NealDSDC we were free to choose a grid for θ and ϵ, here we aren't.
#       The grid we chose was on [0, 5]. That is the support of Beta distribution
#       so here we multiply all samples on [0, 1] by 5
reward(m::NealCSDC, s::Tuple{Float64,Float64}) = 5*s[1] + 5*s[2]

# NOTE: m = NealCSDC; abs(mean(m.F) - m.F_mean) > 1e-5
#       returns true. That's not good..

v_init(m::NealCSDC) = fill(0.0, length(m.θ), length(m.ϵ))
q_init(m::NealCSDC) = fill(0.0, 3, length(m.θ), length(m.ϵ))

function bellman_operator!(m::NealCSDC, v::AbstractArray, out::AbstractArray;
                           ret_policy=false)

    Ev_ϵ = v*m.G_probs
    # new life. This is a function of the distribution parameters and is
    # always constant. No need to recompute it in the loop
    v3 = (5*m.G_mean + 5*m.F_mean + m.β .*
          m.F_probs' * Ev_ϵ)[1]  # don't need 1 element array

    for j=1:length(m.ϵ)
        for i=1:length(m.θ)
            # stay put
            v1 = 5*m.θ[i] + 5*m.ϵ[j] + m.β * v[i, j]

            # new job
            v2 = 5*m.θ[i] .+ 5*m.G_mean .+ m.β .*Ev_ϵ[i]

            if ret_policy
                if v1 >= max(v2, v3)
                    action = 1
                elseif v2 >= max(v1, v3)
                    action = 2
                else
                    action = 3
                end
                out[i, j] = action
            else
                out[i, j] = max(v1, v2, v3)
            end
        end
    end
    out
end

function bellman_Q_operator!(m::NealCSDC, q::AbstractArray, out::AbstractArray;
                             ret_policy=false)
    v = squeeze(maximum(q, 1), 1)

    Ev_ϵ = v*m.G_probs
    # new life. This is a function of the distribution parameters and is
    # always constant. No need to recompute it in the loop
    v3 = (5*m.G_mean + 5*m.F_mean + m.β .*
          m.F_probs' * Ev_ϵ)[1]  # don't need 1 element array

    for j=1:length(m.ϵ)
        for i=1:length(m.θ)
            # stay put
            v1 = 5*m.θ[i] + 5*m.ϵ[j] + m.β * v[i, j]

            # new job
            v2 = 5*m.θ[i] .+ 5*m.G_mean .+ m.β .*Ev_ϵ[i]

            out[1, i, j] = v1
            out[2, i, j] = v2
            out[3, i, j] = v3
        end
    end
    out
end

function Base.step(m::NealCSDC, s::Tuple{Float64,Float64}, a::Int)
    if a == 1  # stay put
        sp = s
    elseif a == 2
        # new job
        ϵp = rand(m.G)
        sp = (s[1], ϵp)
    elseif a == 3
        # new career
        θp = rand(m.G)
        ϵp = rand(m.G)
        sp = (θp, ϵp)
    end

    return sp, reward(m, sp)
end

# ------------- #
# DiscreteModel #
# ------------- #

struct NealDSDC <: NealModel
    beta::Float64
    N::Int
    B::Float64
    θ::Vector{Float64}
    ϵ::Vector{Float64}
    F_dist::BetaBinomial
    G_dist::BetaBinomial
    F_probs::Vector{Float64}
    G_probs::Vector{Float64}
    F_drv::DiscreteRV{Vector{Float64},Vector{Float64}}
    G_drv::DiscreteRV{Vector{Float64},Vector{Float64}}
    F_mean::Float64
    G_mean::Float64
end

function NealDSDC(;beta::Real=0.95, B::Real=5.0, N::Real=50,
                             F_a::Real=1, F_b::Real=1, G_a::Real=10,
                             G_b::Real=10)
    theta = range(0, stop=B, length=N)
    epsilon = copy(theta)
    F_dist = BetaBinomial(N-1, F_a, F_b)
    G_dist = BetaBinomial(N-1, G_a, G_b)
    F_probs::Vector{Float64} = pdf(F_dist)
    G_probs::Vector{Float64} = pdf(G_dist)
    F_mean = sum(theta .* F_probs)
    G_mean = sum(epsilon .* G_probs)
    F_drv = DiscreteRV(F_probs)
    G_drv = DiscreteRV(G_probs)
    NealDSDC(beta, N, B, theta, epsilon, F_dist, G_dist, F_probs,
                        G_probs, F_drv, G_drv, F_mean, G_mean)
end


function plot_distributions(m::NealDSDC)
    Plot([
        bar(x=m.ϵ, y=m.G_probs, name="ϵ ~ G"),
        bar(x=m.θ, y=m.F_probs, name="θ ~ F"),
    ])
end

v_init(m::NealDSDC) = fill(100.0, m.N, m.N)
q_init(m::NealDSDC) = fill(100.0, 3, m.N, m.N)
rand_action(m::NealModel, s=missing) = Action(rand(1:3))

function bellman_operator!(m::NealDSDC, v::Array, out::Array;
                           ret_policy=false)
    # new life. This is a function of the distribution parameters and is
    # always constant. No need to recompute it in the loop
    v3 = (m.G_mean + m.F_mean + m.beta .* (m.F_probs' * v * m.G_probs))

    for j=1:m.N
        for i=1:m.N
            # stay put
            v1 = m.θ[i] + m.ϵ[j] + m.beta * v[i, j]

            # new job
            v2 = m.θ[i] + m.G_mean + m.beta * dot(v[i, :], m.G_probs)

            if ret_policy
                if v1 >= max(v2, v3)
                    action = 1
                elseif v2 >= max(v1, v3)
                    action = 2
                else
                    action = 3
                end
                out[i, j] = action
            else
                out[i, j] = max(v1, v2, v3)
            end
        end
    end
end

function bellman_Q_operator!(m::NealDSDC, q::Array, out::Array)
    # get optimal value function
    v = dropdims(maximum(q, dims=1), dims=1)

    # new life. This is a function of the distribution parameters and is
    # always constant. No need to recompute it in the loop
    v3 = m.G_mean + m.F_mean + m.beta * (m.F_probs' * v * m.G_probs)

    for j=1:m.N
        for i=1:m.N
            # stay put
            v1 = m.θ[i] + m.ϵ[j] + m.beta * v[i, j]

            # new job
            v2 = m.θ[i] + m.G_mean + m.beta * dot(v[i, :], m.G_probs)

            out[1, i, j] = v1
            out[2, i, j] = v2
            out[3, i, j] = v3
        end
    end
    out
end

function Base.step(m::NealDSDC, s, a)
    if a == 1      # stay put
        sp = s
    elseif a == 2  # new job
        sp = NealState(s.θ, draw(m.G_drv))
    elseif a == 3  # new career
        sp = NealState(draw(wp.F_drv), draw(wp.G_drv))
    end

    return sp, m.θ[sp.θ] + m.ϵ[sp.ϵ]
end

# -------- #
# Plotting #
# -------- #

function plot_vf(m::NealModel=NealDSDC(), v=vfi(m))
    tg, eg = meshgrid(m.θ, m.ϵ)

    s = surface(x=tg, y=eg, z=v', colorscale="Viridis")
    l = Layout(xaxis_title="theta", yaxis_title="epsilon",
               scene_camera_eye=attr(x=-1.6, y=-1.55, z=0.66))
    Plot(s, l)
end

function plot_policy(m::NealModel=NealDSDC(),
                     pol=get_greedy(m, vfi(m)))
    c_new_life = RGB(180/255, 255/255, 229/255)
    c_new_job = RGB(255/255, 229/255, 180/255)
    c_stay = RGB(255/255, 180/255, 206/255)
    cs = [(0, c_stay), (1/3, c_stay),
          (1/3, c_new_job), (2/3, c_new_job),
          (2/3, c_new_life), (1, c_new_life)]

    cb = attr(autotick=false, tick0=1, dtick=1)

    ann = [attr(text="new life", x=2.5, y=2.5, showarrow=false),
           attr(text="new job", x=4.5, y=2.5, textangle=270, showarrow=false),
           attr(text="stay put", x=4.5, y=4.5, showarrow=false)]

    layout = Layout(annotations=ann, width=500, height=500,
                    xaxis_title="theta", yaxis_title="epsilon")

    Plot(heatmap(x=m.θ, y=m.ϵ, z=pol,
                 colorscale=cs,
                 showscale=false,
                 colorbar=cb),
         layout)
end

#=
function plot_policy(m::NealCSDC=NealCSDC(),
                     vf::Matrix{Float64}=vfi(m))
    # we assume we have a matrix `vf` where
    # vf[i,j] = v(m.θ[i], m.ϵ[j])
    # we want to plot on a more continuous looking grid
    θf = linspace(extrema(m.θ)..., 300)
    ϵf = linspace(extrema(m.ϵ)..., 300)


    c_new_life = RGB(180/255, 255/255, 229/255)
    c_new_job = RGB(255/255, 229/255, 180/255)
    c_stay = RGB(255/255, 180/255, 206/255)
    cs = [(0, c_stay), (1/3, c_stay),
          (1/3, c_new_job), (2/3, c_new_job),
          (2/3, c_new_life), (1, c_new_life)]
    cb = attr(autotick=false, tick0=1, dtick=1)

    ann = [attr(text="new life", x=2.5, y=2.5, showarrow=false),
           attr(text="new job", x=4.5, y=2.5, textangle=270, showarrow=false),
           attr(text="stay put", x=4.5, y=4.5, showarrow=false)]

    layout = Layout(annotations=ann, width=500, height=500,
                    xaxis_title="theta", yaxis_title="epsilon")

    Plot(heatmap(x=θf, y=ϵf, z=pol,
                 colorscale=cs,
                 showscale=false,
                 colorbar=cb),
         layout)
end
=#

# ------------- #
# RL Algorithms #
# ------------- #

struct State
    θ::Int
    ϵ::Int
end

reward(wp::NealDSDC, s::State) = wp.θ[s.θ] + wp.ϵ[s.ϵ]

rand_state(wp::NealDSDC) = State(draw(wp.F_drv), draw(wp.G_drv))

struct Action
    i::Int
    Action(i::Int) = i > 0 && i < 4 ?
        new(i) :
        error("i  must be in {1, 2, 3}")
end

Base.convert(::Type{Action}, i::Integer) = Action(convert(Int, i))

==(a::Action, i::Int) = a.i == i

function Base.step(wp::NealDSDC, s::State, a::Action)
    if a == 1  # stay put
        sp = s
    elseif a == 2
        # new job
        sp = State(s.θ, draw(wp.G_drv))
    elseif a == 3
        # new career
        sp = rand_state(wp)
    end

    return sp, reward(wp, sp)

end

Base.step(wp::NealDSDC, s::State, i::Int) = step(wp, s, Action(i))

# wrap 3d array to make setindex! and getindex natural
struct QFunction
    q::Array{Float64,3}
end

Base.setindex!(Q::QFunction, v::Real, s::State, a::Action) =
    setindex!(Q.q, v, a.i, s.θ, s.ϵ)

Base.getindex(Q::QFunction, s::State, a::Action) =
    getindex(Q.q, a.i, s.θ, s.ϵ)

Base.view(Q::QFunction, s::State, ::Colon) = view(Q.q, :, s.θ, s.ϵ)

function get_greedy(Q::QFunction, s::State)
    s1 = s.θ
    s2 = s.ϵ

    the_max = -Inf
    arg_max = 0
    for i in 1:3
        q_i = Q.q[i, s1, s2]
        if q_i > the_max
            the_max = q_i
            arg_max = i
        elseif q_i == the_max # break ties randomly
            foo = rand(Bool)
            if foo
                the_max = q_i
                arg_max = i
            end
        end
    end
    return Action(arg_max)
end

# get policy rule, given the QFunction
get_greedy(wp::NealDSDC, Q::QFunction) =
    get_greedy(wp, dropdims(maximum(Q.q, dims=1), dims=1))

#=----------------------------------------------+
|                                               |
|                  Algorithms                   |
|                                               |
+----------------------------------------------=#

abstract type AbstractActor end
abstract type AbstractAlgorithm end
abstract type AbstractQAlgorithm <: AbstractAlgorithm end

get_greedy(wp::NealDSDC, algo::AbstractQAlgorithm) = get_greedy(wp, algo.Q)

mutable struct EpsilonGreedy <: AbstractActor
    ϵ0::Float64
    ϵT::Float64
    T::Int

    # internal state
    t::Int
    ϵt::Float64
    Δϵ::Float64

    function EpsilonGreedy(e0, eT, T)
        Δϵ = (eT - e0) / (T-1)
        new(e0, eT, T, 0, e0, Δϵ)
    end
end

EpsilonGreedy() = EpsilonGreedy(1.0, 0.02, 1_000_000)

PlotlyBase.Plot(eg::EpsilonGreedy) =
    Plot(scatter(x=[0, eg.T, 2*eg.T], y=[eg.ϵ0, eg.ϵT, eg.ϵT], mode="lines"),
         Layout(xaxis_title="t", yaxis_title="ϵ", title="Epsilon greedy policy"))

function get_action(wp::AbstractModel, algo::AbstractAlgorithm,
                    eg::EpsilonGreedy, sp)
    eg.t += 1

    exploring = rand() < eg.ϵt
    ap = exploring ? rand_action(wp, sp) : get_greedy(algo, sp)

    if eg.t < eg.T
        eg.ϵt += eg.Δϵ
    end
    return ap, exploring
end

struct SoftMax <: AbstractActor end

function get_action(wp::NealDSDC, algo::AbstractQAlgorithm, eg::SoftMax, sp::State)
    # apply softmax function
    exp_Q_sp = exp(view(algo.Q, sp, :))
    exp_Q_sp ./= sum(exp_Q_sp)

    # create distribution and draw from it
    ap = Action(draw(DiscreteRV(exp_Q_sp)))

    # see if we explored
    exploring = indmax(exp_Q_sp) != ap.i

    ap, exploring
end

# Barto & Sutton Algorithm in figure 6.5
mutable struct Sarsa <: AbstractQAlgorithm
    Q::QFunction
    s::State
    a::Action
    α::Float64
end

function Sarsa(wp::NealDSDC, α::Float64=0.5)
    s = rand_state(wp)
    Sarsa(QFunction(zeros(3, length(wp.θ), length(wp.ϵ))),
          rand_state(wp),
          rand_action(wp, s),
          α)
end

get_greedy(algo::Sarsa, s) = get_greedy(algo.Q, s)

function update!(wp::NealDSDC, actor::AbstractActor, algo::Sarsa)
    # use current state and action to do transition
    sp, r = step(wp, algo.s, algo.a)

    # get next action using actor's strategy
    ap = get_action(wp, algo, actor, sp)[1]

    # Apply TD update
    algo.Q[algo.s, algo.a] = (1-algo.α)*algo.Q[algo.s, algo.a] +
                             algo.α*(r + wp.beta * algo.Q[sp, ap])

    # step forward in time
    algo.s = sp
    algo.a = ap
end

# Barto & Sutton Algorithm in figure 7.13
mutable struct Sarsaλ <: AbstractQAlgorithm
    Q::QFunction
    E::QFunction
    s::State
    a::Action
    α::Float64
    λ::Float64
end

function Sarsaλ(wp::NealDSDC, α::Float64=0.1, λ::Float64=0.8)
    Sarsaλ(QFunction(zeros(3, length(wp.θ), length(wp.ϵ))),
           QFunction(zeros(3, length(wp.θ), length(wp.ϵ))),
           rand_state(wp),
           rand_action(wp),
           α,
           λ)
end

get_greedy(algo::Sarsaλ, s) = get_greedy(algo.Q, s)

function update!(wp::NealDSDC, actor::AbstractActor, algo::Sarsaλ)
    # use current state and action to do transition
    sp, r = step(wp, algo.s, algo.a)

    # get next action using actor's strategy
    ap = get_action(wp, algo, actor, sp)[1]

    # construct TD(0) error
    δ = r + wp.beta*algo.Q[sp, ap] - algo.Q[algo.s, algo.a]

    # update eligibility trace for action chosen today
    # NOTE: I'm using dutch traces here
    algo.E[algo.s, algo.a] = (1-algo.α)*algo.E[algo.s, algo.a] + 1.0

    βλ = wp.beta * algo.λ
    αδ = algo.α * δ

    # update Q and E
    @inbounds @simd for ix in eachindex(algo.Q.q)
        algo.Q.q[ix] += αδ*algo.E.q[ix]
        algo.E.q[ix] *= βλ
    end

    # step forward in time
    algo.s = sp
    algo.a = ap
end


# Barto & Sutton Algorithm in figure 6.7
mutable struct QLearning <: AbstractQAlgorithm
    Q::QFunction
    s::State
    α::Float64
end

function QLearning(wp::NealDSDC, α::Float64=0.1)
    QLearning(QFunction(zeros(3, length(wp.θ), length(wp.ϵ))),
              rand_state(wp),
              α)
end

get_greedy(algo::QLearning, s) = get_greedy(algo.Q, s)

function update!(wp::NealDSDC, actor::AbstractActor, algo::QLearning)
    # choose a
    a = get_action(wp, algo, actor, algo.s)[1]

    # use current state and action to do transition
    sp, r = step(wp, algo.s, a)

    # ap is greedy in Q
    ap = get_greedy(algo, sp)

    # Apply TD update
    algo.Q[algo.s, a] = (1-algo.α)*algo.Q[algo.s, a] +
                           algo.α*(r + wp.beta * algo.Q[sp, ap])

    # step forward in time
    algo.s = sp
end

# Barto & Sutton Algorithm in figure 6.12
mutable struct DoubleQLearning <: AbstractQAlgorithm
    Q::QFunction  # this is just so actors aren't confused. It will always be Q1+Q2
    Q1::QFunction
    Q2::QFunction
    s::State
    α::Float64
end

function DoubleQLearning(wp::NealDSDC, α::Float64=0.5)
    Q1 = QFunction(zeros(3, length(wp.θ), length(wp.ϵ)))
    Q2 = QFunction(zeros(3, length(wp.θ), length(wp.ϵ)))
    Q = QFunction(Q1.q + Q2.q)
    DoubleQLearning(Q,
                    Q1,
                    Q2,
                    rand_state(wp),
                    α)
end

get_greedy(algo::DoubleQLearning, s) = get_greedy(algo.Q, s)

function update!(wp::NealDSDC, actor::AbstractActor, algo::DoubleQLearning)
    # choose a
    a = get_action(wp, algo, actor, algo.s)[1]

    # use current state and action to do transition
    sp, r = step(wp, algo.s, a)

    if rand() < 0.5
        # update Q1 using Q2 to evaluate
        ap = get_greedy(algo.Q1, sp)
        algo.Q1[algo.s, a] += algo.α*(r + wp.beta*algo.Q2[sp, ap] - algo.Q1[algo.s, a])
    else
        # update Q2 using Q1 to evaluate
        ap = get_greedy(algo.Q2, sp)
        algo.Q2[algo.s, a] += algo.α*(r + wp.beta*algo.Q1[sp, ap] - algo.Q2[algo.s, a])
    end

    #make sure Q is Q1 + Q2
    algo.Q[algo.s, a] = algo.Q1[algo.s, a] + algo.Q2[algo.s, a]

    # step forward in time
    algo.s = sp
end

# Barto & Sutton Algorithm in figure 7.16
mutable struct QLearningλ <: AbstractQAlgorithm
    Q::QFunction
    E::QFunction
    s::State
    a::Action
    α::Float64
    λ::Float64
end

function QLearningλ(wp::NealDSDC, α::Float64=0.1, λ::Float64=0.8)
    Sarsaλ(QFunction(zeros(3, length(wp.θ), length(wp.ϵ))),
           QFunction(zeros(3, length(wp.θ), length(wp.ϵ))),
           rand_state(wp),
           rand_action(wp),
           α,
           λ)
end

get_greedy(algo::QLearningλ, s) = get_greedy(algo.Q, s)

function update!(wp::NealDSDC, actor::AbstractActor, algo::QLearningλ)
    # use current state and action to do transition
    sp, r = step(wp, algo.s, algo.a)

    # get next action using epsilon-greedy strategy
    astar = get_greedy(Q, sp)
    ap, exploring = get_action(wp, algo, actor, sp)

    # construct TD(0) error
    δ = r + wp.beta*algo.Q[sp, astar] - algo.Q[algo.s, algo.a]

    # update eligibility trace for chosen action
    algo.E[algo.s, algo.a] = algo.E[algo.s, algo.a] + 1.0

    αδ = algo.α*δ

    # update Q
    @inbounds @simd for ix in eachindex(algo.Q.q)
        algo.Q.q[ix] += αδ*algo.E.q[ix]
    end

    βλ = wp.beta * algo.λ

    # if we explore, set the eligibility trace vector to zero
    if exploring
        fill!(algo.E.q, 0.0)
    else
        # otherwise, decay as usual
        @inbounds @simd for ix in eachindex(algo.E.q)
            algo.E.q[ix] *= βλ
        end
    end

    # step forward in time
    algo.s = sp
end

function learn(wp::NealDSDC, algo::AbstractAlgorithm,
               actor::AbstractActor; maxit::Int=20_000_000,
               should_plot::Bool=false, plot_skip::Int=10_000)
    if should_plot
        p = plot_policy(wp, get_greedy(wp, algo))
        relayout!(p, title="After 0 iterations")
        display(p)
        sleep(1.0)  # give plot plenty of time to display
    end

    N = zeros(Int, length(wp.θ), length(wp.ϵ))
    for it in 1:maxit
        update!(wp, actor, algo)

        N[algo.s.θ, algo.s.ϵ] += 1

        # update plot
        if should_plot && it % plot_skip == 0
            restyle!(p, z=(get_greedy(wp, algo),))
            relayout!(p, title="After $(it) iterations")
            sleep(0.1)
        end
    end
    algo, N
end

#=----------------------------------------------+
|                                               |
|         Continuous State algorithms           |
|                                               |
+----------------------------------------------=#

abstract type AbstractOptimizer end

struct SGD <: AbstractOptimizer
    α::Float64
end

# Barto & Sutton Algorithm in figure 7.13
# TODO: this is bad form. I should separate the representation
#       of Q from the algorithm
struct CSarsa <: AbstractAlgorithm
    # algorithm parameters
    α::Float64

    # internal fields
    θ::Matrix{Float64}         # coefficient matrix, one column for each action
    s::Tuple{Float64,Float64}  # state
    a::Int                     # action
    p1::LinParams              # parameters for θ
    p2::LinParams              # parameters for ϵ
end

function CSarsa(wp::NealCSDC, α::Float64=0.05)
    p1 = LinParams(wp.θ, 0)
    p2 = LinParams(wp.ϵ, 0)

    θ = zeros(length(wp.θ) * length(wp.ϵ), 3)
    s = (rand(wp.F), rand(wp.G))
    a = rand(1:3)
    CSarsa(α, θ, s, a, p1, p2)
end

function get_greedy(algo::CSarsa, s::Tuple{Float64,Float64})
    Φθ = BasisMatrices.evalbase(SplineSparse, algo.p1, [s[1]])
    Φs = BasisMatrices.evalbase(SplineSparse, algo.p2, [s[2]])
    Φ = row_kron(Φs, Φθ)
    qs = Φ*algo.θ

    out = 0
    the_max = -Inf
    for i in 1:length(qs)
        q_i = qs[i]
        if q_i > the_max
            the_max = q_i
            out = i
        elseif q_i == the_max # break ties randomly
            foo = rand(Bool)
            if foo
                the_max = q_i
                out = i
            end
        end
    end
    out
end

function eval_q_and_grads(algo::CSarsa, s::Tuple{Float64,Float64}, a::Int)
    Φθ = BasisMatrices.evalbase(SplineSparse, algo.p1, s[1])
    Φs = BasisMatrices.evalbase(SplineSparse, algo.p2, s[2])
    Φ = row_kron(Φs, Φθ)
    q = (Φ*view(algo.θ, :, a))[1]  # the `*` will return a 1 elment vector
    q, Φ
end

# batch versions
function get_greedy(algo::CSarsa, s::Matrix{Float64})
    Φθ = BasisMatrices.evalbase(SplineSparse, algo.p1, view(s, :, 1))
    Φs = BasisMatrices.evalbase(SplineSparse, algo.p2, view(s, :, 2))
    Φ = row_kron(Φs, Φθ)
    q = Φ * algo.θ
    inds = mapslices(indmax, q, 2)
    vec(inds)
end

function eval_q_and_grads(algo::CSarsa, s::Matrix{Float64}, a::Vector{Int})
    Φθ = BasisMatrices.evalbase(SplineSparse, algo.p1, view(s, :, 1))
    Φs = BasisMatrices.evalbase(SplineSparse, algo.p2, view(s, :, 2))
    Φ = row_kron(Φs, Φθ)
    q = Φ * algo.θ

    out = Array(Float64, length(a))
    for i in eachindex(out)
        out[i] = q[i, a[i]]
    end
    out, Φ
end

function update!(m::NealCSDC, algo::CSarsa, actor::AbstractActor,
                 opt::SGD)
    # use current state and action to do transition
    sp, r = step(m, algo.s, algo.a)

    # get next action using actor's strategy
    ap = get_action(m, algo, actor, sp)[1]

    # get TD(0) return
    target = r + m.β * eval_q_and_grads(algo, sp, ap)[1]
    q, grad = eval_q_and_grads(algo, algo.s, algo.a)

    # update column of coef vector corresponding to this action
    # (all other grads) are 0
    # NOTE: the loops below do the same operation as the line below, but in a
    #       _much_ more efficient way
    # algo.θ[:, a] += opt.α * (target - q) * vec(full(grad))

    @inbounds for chunk in 1:grad.n_chunks
        first_row = grad.cols[CompEcon.col_ix(grad, 1, chunk)]
        for n in 1:grad.chunk_len
            θ_row = first_row + (n-1)
            val_ind = CompEcon.val_ix(grad, 1, chunk, n)
            algo.θ[θ_row, algo.a] += opt.α * (target - q) * grad.vals[val_ind]
        end
    end

    # step time forward
    algo.s = sp
    algo.a = ap
end

function learn(m::NealCSDC, algo::AbstractAlgorithm=CSarsa(m),
               actor::AbstractActor=EpsilonGreedy(),
               opt::AbstractOptimizer=SGD(0.1); maxit::Int=20_000_000,
               should_plot::Bool=false, plot_skip::Int=10_000)
    for it in 1:maxit
        update!(m, algo, actor, opt)
    end
    algo
end

#=
m = NealCSDC()
algo = CSarsa(m)
actor = EpsilonGreedy()
opt = SGD(1.0)
=#
