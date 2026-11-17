# Resilience operators: `ResilienceCore.linearise` for finite-state generator problems.

# How a generator or source term depends on the state and on time: :none, :constant, :time or
# :state. A callable is probed numerically, because the (u, p, t) signature is used for terms
# that ignore `u` as well as for density-dependent ones.
function _dependence(value, term, u, p, t)
    term === nothing && return :none
    term isa AbstractArray && return :constant
    v0 = value(term, u, p, t)
    if applicable(term, u, p, t) || applicable(term, u, t, p)
        value(term, 2 .* u .+ 1, p, t) == v0 || return :state
    end
    for δ in (0.371, 1.913, 7.27)
        value(term, u, p, t + δ) == v0 || return :time
    end
    return :constant
end

_generator_at(prob, u, p, t) =
    _materialize_generator_matrix(_materialize_generator_value(prob.generator, u, p, t), length(u))
_generator_value(term, u, p, t) = _materialize_generator_value(term, u, p, t)
_source_value(term, u, p, t) = _source_vector(term, u, p, t)

"""
    linearise(prob::FiniteStateGeneratorProblem, u = nothing; p = prob.p, t = first(prob.tspan),
              period = nothing, steps = 100, tol = 1e-8, check = true)

The resilience operator of a finite-state generator problem `du/dt = G u + b` (see
ResilienceCore.jl), chosen by how `G` and the source `b` depend on the state and on time:

- **Constant `G`, no source:** `Generator(G)`, a density-independent model. Its `growth_rate` is
  the spectral abscissa of `G`; its `resilience`, `Re λ₁ − Re λ₂`, is the rate of convergence to
  the stable stage structure.
- **Constant `G` and constant source `b`** (immigration, recruitment from outside):
  `ContinuousJacobian(G, u*)` at the equilibrium `u* = −G⁻¹ b` (`affine_equilibrium`). `G` must be
  stable.
- **`G(t)` periodic in time:** pass `period = T`. The result is a `PeriodicOperator` of `steps`
  matrix exponentials over one period from `t` (`periodic_generator`), with per-step metrics
  per `Δt = T / steps`. A source does not change it: perturbations of any solution evolve under
  `G(t)` alone.
- **`G` or `b` depending on the state** (density dependence): pass the equilibrium `u`. The
  problem is lowered with `to_ode_problem` and linearised there, giving a `ContinuousJacobian`;
  this needs ForwardDiff to be loaded. `tol` and `check` are passed on.
- **`normalize = true`** (proportions): without a source, still `Generator(G)`. The normalised
  dynamics do not grow, and their rate of convergence to the stable structure is
  `resilience(Generator(G))`. With a source the normalised model is non-linear and takes the
  state-dependent route.

A time-dependent term without `period` throws an `ArgumentError`. `u` is used only by the
state-dependent route. Delay and stochastic reaction problems have no single operator; see
their `linearise` methods.
"""
function ResilienceCore.linearise(prob::FiniteStateGeneratorProblem, u = nothing; p = prob.p,
                                  t = first(prob.tspan), period = nothing, steps::Integer = 100,
                                  tol = 1e-8, check = true)
    u0 = Vector{Float64}(something(u, prob.u0))
    gdep = _dependence(_generator_value, prob.generator, u0, p, t)
    sdep = _dependence(_source_value, prob.source, u0, p, t)
    hassource = sdep != :none && !(sdep == :constant && all(iszero, _source_vector(prob.source, u0, p, t)))
    if gdep == :state || sdep == :state || (prob.normalize && hassource)
        u === nothing && throw(ArgumentError(
            "this model is density-dependent (its generator or source depends on the state, or it is " *
            "normalised with a source): pass the equilibrium to linearise at, as `linearise(prob, u)`"))
        return ResilienceCore.linearise(to_ode_problem(prob), u0; p = p, t = t, tol = tol, check = check)
    end
    if period !== nothing
        prob.normalize && throw(ArgumentError(
            "a normalised model with a periodic generator is non-linear; linearise its un-normalised generator"))
        return ResilienceCore.periodic_generator(τ -> _generator_at(prob, u0, p, t + τ), period; steps = steps)
    end
    (gdep == :time || sdep == :time) && throw(ArgumentError(
        "the generator or source depends on time: pass `period = T` for a periodic model, or give a " *
        "constant generator"))
    G = _generator_at(prob, u0, p, t)
    hassource || return ResilienceCore.Generator(G)
    b = _source_vector(prob.source, u0, p, t)
    return ResilienceCore.ContinuousJacobian(G, ResilienceCore.affine_equilibrium(G, b))
end

"""
    linearise(prob::DelayFiniteStateProblem, args...; kwargs...)

Not defined: with delayed terms the characteristic equation is transcendental, with infinitely many
roots, so there is no matrix whose spectrum gives the stability. Expand each delay into a chain of
stages (a linear chain trick) and linearise the resulting `FiniteStateGeneratorProblem`, or use
simulation-based measures of recovery.
"""
ResilienceCore.linearise(::DelayFiniteStateProblem, args...; kwargs...) = throw(ArgumentError(
    "a delay problem has no single operator: its characteristic equation has infinitely many roots. " *
    "Expand the delays into chains of stages and linearise the resulting FiniteStateGeneratorProblem, " *
    "or use simulation-based measures of recovery"))

"""
    linearise(prob::FiniteStateReactionProblem, args...; kwargs...)

Not defined: a reaction problem is a stochastic jump process with no single operator. Its mean
follows the deterministic generator problem with the same rates; linearise that, or use
trajectory-based measures (recovery times, variability) on simulated realisations.
"""
ResilienceCore.linearise(::FiniteStateReactionProblem, args...; kwargs...) = throw(ArgumentError(
    "a reaction problem is a stochastic jump process with no single operator: linearise the deterministic " *
    "FiniteStateGeneratorProblem of its mean, or use trajectory-based measures on simulated realisations"))
