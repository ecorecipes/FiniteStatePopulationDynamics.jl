"""
    FiniteStatePopulationDynamics

Finite-state, continuous-time population dynamics backends.

This package is the deterministic generator-based sibling to
`MatrixProjectionModels.jl`. Broad stochastic continuous-time semantics are
intentionally deferred; future stochastic support should be introduced as
extension packages layered on top of these finite-state continuous-time
structures and problem types.
"""
module FiniteStatePopulationDynamics

using CommonSolve
using LinearAlgebra
using Random
using SciMLBase
using StructuredPopulationCore

export AbstractFiniteStateDynamicsStructure
export SimpleFiniteStateStructure, GeneralFiniteStateStructure
export AbstractFiniteStateDynamicsProblem
export DelayGeneratorTerm
export FiniteStateGeneratorProblem, DelayFiniteStateProblem
export remake
export to_ode_problem, to_dde_problem
export solve

export AbstractProjectionStructure
export AbstractTimeSemantics, DiscreteTime, ContinuousTime
export AbstractStateSemantics, FiniteState, ContinuousState
export DirectIteration, EigenAnalysis
export AbstractStateDomain, DiscreteDomain, n_states

# Demographic stochasticity (continuous-time Markov jump process)
export Demographic
export DemographicReaction, DemographicReactionSystem, gillespie
export FiniteStateReactionProblem, FiniteStateDemographicSolution
export generator_reactions, demographic_ensemble

"""
    AbstractFiniteStateDynamicsStructure

Shared supertype for finite-state continuous-time backend structure descriptors.
Deterministic generator problem types and SciML lowering build on this trait.
"""
abstract type AbstractFiniteStateDynamicsStructure <: AbstractProjectionStructure end

"""
    SimpleFiniteStateStructure

Single finite-state component structure.
"""
struct SimpleFiniteStateStructure <: AbstractFiniteStateDynamicsStructure end

"""
    GeneralFiniteStateStructure

Multi-component or otherwise generalized finite-state structure.
"""
struct GeneralFiniteStateStructure <: AbstractFiniteStateDynamicsStructure end

"""
    AbstractFiniteStateDynamicsProblem

Supertype for finite-state continuous-time generator formulations.
"""
abstract type AbstractFiniteStateDynamicsProblem end

# Problem types
include("problems.jl")

# SciML lowering and solve interface
include("sciml_interface.jl")
include("solve.jl")
include("events.jl")
include("demographic.jl")
using CommonSolve: solve

export state_index, set_state!, add_to_state!, scale_state!, transfer_state!
export scheduled_event, periodic_event, threshold_event, combine_callbacks

end # module
