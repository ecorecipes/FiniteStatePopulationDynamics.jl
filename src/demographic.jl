"""
Demographic (finite-population) stochasticity for finite-state continuous-time
dynamics, realized as a continuous-time Markov jump process via the
`StructuredPopulationCore` reaction IR and reference `gillespie` realizer.
"""

# ----------------------------------------------------------------------------
# Reaction problem
# ----------------------------------------------------------------------------

"""
    FiniteStateReactionProblem(reactions, u0, tspan; p=nothing)

A demographic (integer-count) problem defined by an explicit
`DemographicReactionSystem`. This is the general, fully-controlled route: each
reaction carries its own stoichiometry, so movement (`-eᵢ + eⱼ`), birth (`+eⱼ`)
and death (`-eᵢ`) are represented exactly. Solve with `solve(prob, Demographic())`.
"""
struct FiniteStateReactionProblem{R, U, T<:Real, P} <: AbstractFiniteStateDynamicsProblem
    reactions::R
    u0::U
    tspan::Tuple{T, T}
    p::P
end

function _require_integer_counts(u0; name::AbstractString)
    u0i = Vector{Int}(undef, length(u0))
    for i in eachindex(u0)
        xi = round(Int, u0[i])
        isapprox(u0[i], xi; atol = 1e-8, rtol = 0) || throw(ArgumentError(
            "exact demographic solves require integer initial counts; " *
            "$(name)[$i] = $(u0[i]) is not integer-valued"))
        u0i[i] = xi
    end
    return u0i
end

function FiniteStateReactionProblem(reactions::DemographicReactionSystem, u0, tspan; p = nothing)
    u0i = _require_integer_counts(u0; name = "u0")
    length(u0i) == reactions.n_states || throw(DimensionMismatch(
        "u0 length $(length(u0i)) does not match reaction system size $(reactions.n_states)"))
    T = promote_type(typeof(float(tspan[1])), typeof(float(tspan[2])))
    return FiniteStateReactionProblem(reactions, u0i, (T(tspan[1]), T(tspan[2])), p)
end

"""
    FiniteStateDemographicSolution

Result of a demographic (jump-process) solve: event/grid times `t` and integer
count vectors `u`.
"""
struct FiniteStateDemographicSolution{U} <: AbstractProjectionSolution
    t::Vector{Float64}
    u::U
    retcode::Symbol
end

function Base.show(io::IO, sol::FiniteStateDemographicSolution)
    print(io, "FiniteStateDemographicSolution(", length(sol.t), " points, retcode=",
        sol.retcode, ")")
end

# `generator_reactions` lives in StructuredPopulationCore (shared with the
# continuous-state backend) and is re-exported by this package.

# ----------------------------------------------------------------------------
# Solve
# ----------------------------------------------------------------------------

_demographic_grid(tspan, saveat::Number) = collect(tspan[1]:saveat:tspan[2])
_demographic_grid(tspan, saveat::AbstractVector) = collect(float.(saveat))

# Piecewise-constant sampling of an event trajectory onto a time grid.
function _sample_on_grid(ts, us, grid)
    out = Vector{eltype(us)}(undef, length(grid))
    idx = 1
    for g in eachindex(grid)
        tg = grid[g]
        while idx < length(ts) && ts[idx + 1] <= tg
            idx += 1
        end
        out[g] = us[idx]
    end
    return out
end

function _demographic_retcode(sys, ts, us, tspan, p, max_events::Int)
    length(ts) == max_events + 1 || return :Success
    ts[end] < float(tspan[2]) || return :Success
    return total_propensity(sys, us[end], p, ts[end]) > 0 ? :MaxIters : :Success
end

"""
    solve(prob::FiniteStateReactionProblem, ::Demographic; rng, saveat=nothing, max_events)

Realize one continuous-time Markov jump trajectory via Gillespie's direct method.
Returns a [`FiniteStateDemographicSolution`]; with `saveat` (a step or a vector of
times) the trajectory is sampled piecewise-constantly onto that grid, otherwise
the raw event times/states are returned.
"""
function CommonSolve.solve(prob::FiniteStateReactionProblem, ::Demographic;
        rng::AbstractRNG = Random.default_rng(), saveat = nothing,
        max_events::Int = 1_000_000)
    ts, us = gillespie(rng, prob.reactions, prob.u0, prob.tspan;
        p = prob.p, max_events = max_events)
    retcode = _demographic_retcode(prob.reactions, ts, us, prob.tspan, prob.p, max_events)
    if saveat === nothing
        return FiniteStateDemographicSolution(collect(float.(ts)), us, retcode)
    end
    grid = _demographic_grid(prob.tspan, saveat)
    return FiniteStateDemographicSolution(collect(float.(grid)),
        _sample_on_grid(ts, us, grid), retcode)
end

"""
    solve(prob::FiniteStateGeneratorProblem, ::Demographic; rng, saveat, max_events)

Demographic realization of a generator problem: builds reactions from the
constant generator (and constant `source`) via [`generator_reactions`] and runs
the jump process. Requires a constant generator matrix and a `nothing`/constant
`source`.
"""
function CommonSolve.solve(prob::FiniteStateGeneratorProblem, ::Demographic; kwargs...)
    prob.generator isa AbstractMatrix || throw(ArgumentError(
        "Demographic solve of a FiniteStateGeneratorProblem requires a constant generator " *
        "matrix; got $(typeof(prob.generator)). Build a FiniteStateReactionProblem for " *
        "time-varying or explicitly-structured demographic dynamics."))
    src = prob.source
    (src === nothing || src isa AbstractVector) || throw(ArgumentError(
        "Demographic solve requires `source` to be nothing or a constant vector; got $(typeof(src))."))
    reactions = generator_reactions(prob.generator; source = src)
    rprob = FiniteStateReactionProblem(reactions, prob.u0, prob.tspan; p = prob.p)
    return solve(rprob, Demographic(); kwargs...)
end

"""
    demographic_ensemble(prob; n_reps=100, saveat=1.0, rng=Random.default_rng())

Run `n_reps` independent demographic realizations on a common `saveat` grid and
return `(totals, sols)` where `totals` is `(n_grid × n_reps)` of total population
sizes (consumable by `quasi_extinction`).
"""
function demographic_ensemble(prob::AbstractFiniteStateDynamicsProblem;
        n_reps::Int = 100, saveat = 1.0, rng::AbstractRNG = Random.default_rng())
    sols = [solve(prob, Demographic(); rng = rng, saveat = saveat) for _ in 1:n_reps]
    grid = sols[1].t
    totals = Matrix{Float64}(undef, length(grid), n_reps)
    for (r, s) in enumerate(sols)
        @inbounds for g in eachindex(grid)
            totals[g, r] = sum(s.u[g])
        end
    end
    return totals, sols
end
