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
