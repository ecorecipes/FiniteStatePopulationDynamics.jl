"""
    FiniteStatePopulationDynamics

Finite-state, continuous-time population dynamics backends.

This package is the continuous-time finite-state sibling to
`MatrixProjectionModels.jl`. It provides deterministic generator/delay problem
types lowered to SciML ODE/DDE problems, event/callback helpers, and exact
demographic stochasticity for finite-population continuous-time Markov jump
processes.
"""
module FiniteStatePopulationDynamics

using CommonSolve
using LinearAlgebra
using Random
using SciMLBase
using StructuredPopulationCore
import SciMLBase: remake

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
