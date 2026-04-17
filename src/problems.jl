"""
Continuous-time generator problem types for finite-state population dynamics.
"""

"""
    FiniteStateGeneratorProblem(structure, generator, domain, u0, tspan; p, source, normalize)

Continuous-time finite-state dynamics driven by an infinitesimal generator. The
generator may be a constant matrix or a callable returning one.
"""
struct FiniteStateGeneratorProblem{
        S<:AbstractFiniteStateDynamicsStructure,
        G,
        Dom,
        U,
        T<:Real,
        P,
        B,
        C} <: AbstractFiniteStateDynamicsProblem
    structure::S
    generator::G
    domain::Dom
    u0::U
    tspan::Tuple{T, T}
    p::P
    source::B
    normalize::Bool
    callbacks::C
end

"""
    DelayFiniteStateProblem(structure, generator, delay_terms, domain, u0, history, tspan;
                            p, source, normalize)

Delay problem with an instantaneous generator plus delayed linear
contributions over a finite state space.
"""
struct DelayFiniteStateProblem{
        S<:AbstractFiniteStateDynamicsStructure,
        G,
        D,
        Dom,
        U,
        H,
        T<:Real,
        P,
        B,
        C} <: AbstractFiniteStateDynamicsProblem
    structure::S
    generator::G
    delay_terms::Vector{D}
    domain::Dom
    u0::U
    history::H
    tspan::Tuple{T, T}
    p::P
    source::B
    normalize::Bool
    callbacks::C
end

function _finite_state_dimension(domain)
    if applicable(n_states, domain)
        return n_states(domain)
    elseif domain isa AbstractDict
        return sum(n_states(d) for d in values(domain))
    elseif domain isa Integer
        return Int(domain)
    else
        return nothing
    end
end

function _validate_finite_state(domain, u0)
    dim = _finite_state_dimension(domain)
    dim === nothing && return
    length(u0) == dim || throw(DimensionMismatch(
        "initial state length $(length(u0)) does not match domain size $(dim)"))
end

function _finite_state_numeric_type(u0, tspan)
    promote_type(Float64, eltype(u0), typeof(float(tspan[1])), typeof(float(tspan[2])))
end

function FiniteStateGeneratorProblem(structure::AbstractFiniteStateDynamicsStructure,
        generator, domain, u0, tspan;
        p = nothing, source = nothing, normalize = false, callbacks = nothing)
    u0_vec = collect(u0)
    isempty(u0_vec) && throw(ArgumentError("u0 must contain at least one state value"))
    _validate_finite_state(domain, u0_vec)
    T = _finite_state_numeric_type(u0_vec, tspan)
    return FiniteStateGeneratorProblem(structure, generator, domain, T.(u0_vec),
        (T(tspan[1]), T(tspan[2])), p, source, normalize, callbacks)
end

function FiniteStateGeneratorProblem(generator, domain, u0, tspan; kwargs...)
    FiniteStateGeneratorProblem(SimpleFiniteStateStructure(), generator, domain, u0, tspan; kwargs...)
end

function DelayFiniteStateProblem(structure::AbstractFiniteStateDynamicsStructure,
        generator, delay_terms::AbstractVector{<:DelayGeneratorTerm},
        domain, u0, history, tspan;
        p = nothing, source = nothing, normalize = false, callbacks = nothing)
    u0_vec = collect(u0)
    isempty(u0_vec) && throw(ArgumentError("u0 must contain at least one state value"))
    _validate_finite_state(domain, u0_vec)
    T = _finite_state_numeric_type(u0_vec, tspan)
    converted_terms = [DelayGeneratorTerm(T(term.lag), term.operator) for term in delay_terms]
    return DelayFiniteStateProblem(structure, generator, converted_terms, domain, T.(u0_vec),
        history, (T(tspan[1]), T(tspan[2])), p, source, normalize, callbacks)
end

function DelayFiniteStateProblem(generator, delay_terms, domain, u0, history, tspan; kwargs...)
    DelayFiniteStateProblem(SimpleFiniteStateStructure(), generator, delay_terms, domain, u0, history, tspan; kwargs...)
end

function remake(prob::FiniteStateGeneratorProblem;
        structure = prob.structure,
        generator = prob.generator,
        domain = prob.domain,
        u0 = prob.u0,
        tspan = prob.tspan,
        p = prob.p,
        source = prob.source,
        normalize = prob.normalize,
        callbacks = prob.callbacks)
    FiniteStateGeneratorProblem(structure, generator, domain, u0, tspan;
        p = p, source = source, normalize = normalize, callbacks = callbacks)
end

function remake(prob::DelayFiniteStateProblem;
        structure = prob.structure,
        generator = prob.generator,
        delay_terms = prob.delay_terms,
        domain = prob.domain,
        u0 = prob.u0,
        history = prob.history,
        tspan = prob.tspan,
        p = prob.p,
        source = prob.source,
        normalize = prob.normalize,
        callbacks = prob.callbacks)
    DelayFiniteStateProblem(structure, generator, delay_terms, domain, u0, history, tspan;
        p = p, source = source, normalize = normalize, callbacks = callbacks)
end

function Base.show(io::IO, prob::FiniteStateGeneratorProblem)
    print(io, "FiniteStateGeneratorProblem(", typeof(prob.structure).name.name,
        ", tspan=", prob.tspan, ")")
end

function Base.show(io::IO, prob::DelayFiniteStateProblem)
    print(io, "DelayFiniteStateProblem(", typeof(prob.structure).name.name,
        ", ", length(prob.delay_terms), " delays, tspan=", prob.tspan, ")")
end
