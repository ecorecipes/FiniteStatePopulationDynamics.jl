# FiniteStatePopulationDynamics.jl

Finite-state, continuous-time population dynamics in Julia. This package is the
deterministic generator-based sibling of
[MatrixProjectionModels.jl](https://github.com/ecorecipes/MatrixProjectionModels.jl):
it takes the same discrete-stage state space and lifts it into continuous time
via infinitesimal generator matrices and delay generator terms.

Broad stochastic continuous-time semantics are intentionally deferred — they
should arrive as extension packages layered on top of the deterministic problem
types here, rather than widening the core backend.

## Features

- **Generator problems**: `FiniteStateGeneratorProblem` — deterministic
  infinitesimal generator (Q-matrix) dynamics, lowered to ODEs.
- **Delay generator terms**: `DelayGeneratorTerm` + `DelayFiniteStateProblem`
  for finite-state continuous-time systems with maturation delays, lowered to
  DDEs.
- **SciML lowering**: `to_ode_problem`, `to_dde_problem`, and
  `CommonSolve.solve` dispatch into the SciML solver stack.
- **Events and callbacks**: domain-aware state mutators (`set_state!`,
  `add_to_state!`, `scale_state!`, `transfer_state!`) and convenience builders
  (`scheduled_event`, `periodic_event`, `threshold_event`,
  `combine_callbacks`).
- **Categorical lowering target**: integrates with
  [CategoricalPopulationDynamics.jl](https://github.com/ecorecipes/CategoricalPopulationDynamics.jl)
  via a dedicated weakdep extension so valued/labelled categorical nets can
  lower directly to a finite-state continuous-time problem.

## Installation

This package is not yet registered in the Julia General registry. Install
directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/FiniteStatePopulationDynamics.jl")
```

## Quick start

```julia
using FiniteStatePopulationDynamics, OrdinaryDiffEq

# 3-state generator: state 1 -> 2 -> 3 at rates 0.5 and 0.3
Q = [-0.5  0.0  0.0;
      0.5 -0.3  0.0;
      0.0  0.3  0.0]
u0 = [100.0, 0.0, 0.0]

prob = FiniteStateGeneratorProblem(Q, u0, (0.0, 20.0))
sol  = solve(prob, Tsit5())
```

## Related

- [StructuredPopulationCore.jl](https://github.com/ecorecipes/StructuredPopulationCore.jl)
  — shared abstractions (domains, state/time semantics, analysis primitives)
- [MatrixProjectionModels.jl](https://github.com/ecorecipes/MatrixProjectionModels.jl)
  — discrete-state, discrete-time matrix projection models
- [IntegralProjectionModels.jl](https://github.com/ecorecipes/IntegralProjectionModels.jl)
  — continuous-state, discrete-time integral projection models
- [ContinuousStatePopulationDynamics.jl](https://github.com/ecorecipes/ContinuousStatePopulationDynamics.jl)
  — continuous-state, continuous-time dynamics
- [CategoricalPopulationDynamics.jl](https://github.com/ecorecipes/CategoricalPopulationDynamics.jl)
  — compositional categorical front-end that lowers to the finite-state backend
- [PhysiologicallyBasedDemographicModels.jl](https://github.com/ecorecipes/PhysiologicallyBasedDemographicModels.jl)
  — application-level PBDM reference suite
