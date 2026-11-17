# FiniteStatePopulationDynamics.jl

Finite-state, continuous-time population dynamics in Julia.

This package is the deterministic generator-based sibling of
`MatrixProjectionModels.jl`: it takes the same discrete-stage state space and
lifts it into continuous time via infinitesimal generator matrices and delay
generator terms. Broad stochastic continuous-time semantics are intentionally
deferred to future extension packages.

## Problem types

```@docs
FiniteStateGeneratorProblem
DelayFiniteStateProblem
DelayGeneratorTerm
```

## SciML lowering and solving

```@docs
to_ode_problem
to_dde_problem
```

Use `CommonSolve.solve` with a SciML solver (e.g. `Tsit5()`, `MethodOfSteps`)
to integrate the lowered problem.

## Resilience operators

`linearise` (from [ResilienceCore.jl](https://github.com/ecorecipes/ResilienceCore.jl), re-exported
here) turns a generator problem into the operator whose stability and transient metrics apply: a
`Generator` for a constant generator, a `ContinuousJacobian` at the equilibrium when there is a
constant source, a `PeriodicOperator` for a periodic generator, and the Jacobian at a given
equilibrium for a density-dependent model.

```julia
using FiniteStatePopulationDynamics, ResilienceCore

G = [-0.5 0.1; 0.3 -0.2]
prob = FiniteStateGeneratorProblem(G, DiscreteDomain([:juvenile, :adult]), [1.0, 1.0], (0.0, 10.0);
                                   source = [0.2, 0.0])
J = linearise(prob)              # ContinuousJacobian at u* = −G⁻¹ b
resilience(J), reactivity(J)
```

```@docs
linearise
```

## Events and callbacks

```@docs
state_index
set_state!
add_to_state!
scale_state!
transfer_state!
scheduled_event
periodic_event
threshold_event
combine_callbacks
```

## Structure traits

```@docs
AbstractFiniteStateDynamicsStructure
SimpleFiniteStateStructure
GeneralFiniteStateStructure
```
