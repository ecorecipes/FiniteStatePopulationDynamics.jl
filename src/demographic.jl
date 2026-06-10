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

function FiniteStateReactionProblem(reactions::DemographicReactionSystem, u0, tspan; p = nothing)
    u0i = round.(Int, u0)
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

# ----------------------------------------------------------------------------
# Generator -> reactions
# ----------------------------------------------------------------------------

# Propensity for a first-order reaction with rate coefficient `coef` acting on
# state `i`: a fresh closure per call avoids loop-capture issues.
_linear_propensity(coef, i) = (n, p, t) -> coef * n[i]

"""
    generator_reactions(G; source=nothing)

Build a `DemographicReactionSystem` from an `n × n` generator `G` whose
conditional mean reproduces `dn/dt = G·n` exactly. Off-diagonal entries
`G[j,i] > 0` become **migration** reactions `i → j` (`-eᵢ + eⱼ`); each state's
net column sum becomes a self **birth** (`+eᵢ`, if positive) or **death**
(`-eᵢ`, if negative); a constant nonnegative `source` vector adds immigration.

The mean is exact for any generator. The *fluctuation* structure follows this
migration-plus-net-birth/death convention, which is the correct continuous-time
Markov chain when off-diagonal flows are genuine movements (e.g. epidemic or
physiological-condition transitions). For models where off-diagonal entries are
fecundity (a parent persists while producing offspring), build the reactions
explicitly via [`FiniteStateReactionProblem`] so the stoichiometry is right.
"""
function generator_reactions(G::AbstractMatrix; source = nothing)
    n = size(G, 1)
    size(G, 2) == n || throw(DimensionMismatch("generator must be square; got $(size(G))"))
    reactions = DemographicReaction[]
    for i in 1:n, j in 1:n
        i == j && continue
        r = G[j, i]
        if r > 0
            push!(reactions, DemographicReaction(_linear_propensity(r, i), n, i => -1, j => +1))
        elseif r < 0
            throw(ArgumentError(
                "off-diagonal generator entry G[$j,$i] = $r is negative; not a valid rate"))
        end
    end
    for i in 1:n
        cs = sum(@view G[:, i])
        if cs > 0
            push!(reactions, DemographicReaction(_linear_propensity(cs, i), n, i => +1))
        elseif cs < 0
            push!(reactions, DemographicReaction(_linear_propensity(-cs, i), n, i => -1))
        end
    end
    if source !== nothing
        src = collect(source)
        length(src) == n || throw(DimensionMismatch(
            "source length $(length(src)) does not match generator size $n"))
        for i in 1:n
            si = src[i]
            if si > 0
                push!(reactions, DemographicReaction(float(si), n, i => +1))  # immigration
            elseif si < 0
                throw(ArgumentError("source[$i] = $si < 0 is not a valid immigration rate"))
            end
        end
    end
    return DemographicReactionSystem(n, reactions)
end

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
    if saveat === nothing
        return FiniteStateDemographicSolution(collect(float.(ts)), us, :Success)
    end
    grid = _demographic_grid(prob.tspan, saveat)
    return FiniteStateDemographicSolution(collect(float.(grid)),
        _sample_on_grid(ts, us, grid), :Success)
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
