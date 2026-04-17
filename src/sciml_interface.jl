"""
SciML ecosystem interface for finite-state continuous-time models.
"""

function generator_matrix(value::AbstractMatrix)
    return value
end

function generator_matrix(value)
    throw(ArgumentError(
        "generator must materialize to an AbstractMatrix, got $(typeof(value))"))
end

function _materialize_generator_matrix(value, n::Int)
    matrix = generator_matrix(value)
    size(matrix) == (n, n) || throw(DimensionMismatch(
        "generator matrix has size $(size(matrix)); expected ($n, $n)"))
    return matrix
end

function _materialize_generator_value(generator, u, p, t)
    if generator isa AbstractMatrix
        return generator
    elseif applicable(generator, u, p, t)
        return generator(u, p, t)
    elseif applicable(generator, u, t, p)
        return generator(u, t, p)
    elseif applicable(generator, p, t)
        return generator(p, t)
    elseif applicable(generator, t, p)
        return generator(t, p)
    else
        throw(ArgumentError(
            "generator $(typeof(generator)) is neither a matrix nor callable as (u, p, t)"))
    end
end

function _source_vector(source, u, p, t)
    source === nothing && return zeros(eltype(u), length(u))
    value = if source isa AbstractVector
        source
    elseif applicable(source, u, p, t)
        source(u, p, t)
    elseif applicable(source, u, t, p)
        source(u, t, p)
    elseif applicable(source, p, t)
        source(p, t)
    elseif applicable(source, t, p)
        source(t, p)
    else
        throw(ArgumentError(
            "source $(typeof(source)) is neither a vector nor callable as (u, p, t)"))
    end
    length(value) == length(u) || throw(DimensionMismatch(
        "source has length $(length(value)); expected $(length(u))"))
    return collect(eltype(u).(value))
end

function _lagged_state(history, p, t, n::Int, T::Type)
    value = history(p, t)
    length(value) == n || throw(DimensionMismatch(
        "history returned length $(length(value)); expected $(n)"))
    return collect(T.(value))
end

function _delay_term_matrix(term::DelayGeneratorTerm, u, history, p, t)
    op = term.operator
    value = if op isa AbstractMatrix
        op
    elseif applicable(op, u, history, p, t, term.lag)
        op(u, history, p, t, term.lag)
    elseif applicable(op, u, history, p, t)
        op(u, history, p, t)
    elseif applicable(op, u, p, t, term.lag)
        op(u, p, t, term.lag)
    elseif applicable(op, u, p, t)
        op(u, p, t)
    elseif applicable(op, p, t, term.lag)
        op(p, t, term.lag)
    elseif applicable(op, p, t)
        op(p, t)
    else
        throw(ArgumentError(
            "delay operator $(typeof(op)) is neither a matrix nor callable with supported signatures"))
    end
    return value
end

function _apply_finite_state_normalization!(du, u)
    total = sum(u)
    total > zero(total) || return
    growth = sum(du) / total
    du .-= growth .* u
end

"""
    to_ode_problem(prob::FiniteStateGeneratorProblem)

Convert a finite-state generator problem to a `SciMLBase.ODEProblem`.
"""
function to_ode_problem(prob::FiniteStateGeneratorProblem)
    function generator_ode!(du, u, p, t)
        G = _materialize_generator_matrix(
            _materialize_generator_value(prob.generator, u, p, t),
            length(u))
        mul!(du, G, u)
        du .+= _source_vector(prob.source, u, p, t)
        prob.normalize && _apply_finite_state_normalization!(du, u)
        return nothing
    end
    return SciMLBase.ODEProblem(generator_ode!, prob.u0, prob.tspan, prob.p)
end

"""
    to_dde_problem(prob::DelayFiniteStateProblem)

Convert a finite-state delay generator problem to a `SciMLBase.DDEProblem`.
"""
function to_dde_problem(prob::DelayFiniteStateProblem)
    lags = [term.lag for term in prob.delay_terms]
    function generator_dde!(du, u, h, p, t)
        G = _materialize_generator_matrix(
            _materialize_generator_value(prob.generator, u, p, t),
            length(u))
        mul!(du, G, u)
        for term in prob.delay_terms
            lagged_u = _lagged_state(h, p, t - term.lag, length(u), eltype(u))
            delayed = _delay_term_matrix(term, u, h, p, t)
            if delayed isa AbstractVector
                length(delayed) == length(u) || throw(DimensionMismatch(
                    "delay term vector has length $(length(delayed)); expected $(length(u))"))
                du .+= eltype(u).(delayed)
            else
                Aτ = _materialize_generator_matrix(delayed, length(u))
                du .+= Aτ * lagged_u
            end
        end
        du .+= _source_vector(prob.source, u, p, t)
        prob.normalize && _apply_finite_state_normalization!(du, u)
        return nothing
    end
    return SciMLBase.DDEProblem(generator_dde!, prob.u0, prob.history, prob.tspan, prob.p;
        constant_lags = lags)
end
