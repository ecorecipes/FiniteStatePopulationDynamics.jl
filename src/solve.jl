function CommonSolve.solve(prob::AbstractFiniteStateDynamicsProblem; kwargs...)
    throw(ArgumentError(
        "Finite-state continuous-time dynamics problems require an explicit SciML time-stepping algorithm. " *
        "Use to_ode_problem/to_dde_problem or call solve(prob, alg) with a SciML algorithm."))
end

function CommonSolve.solve(prob::FiniteStateGeneratorProblem, ::DirectIteration; kwargs...)
    throw(ArgumentError("DirectIteration is not defined for FiniteStateGeneratorProblem; use a SciML ODE algorithm instead."))
end

function CommonSolve.solve(prob::FiniteStateGeneratorProblem, ::EigenAnalysis; kwargs...)
    throw(ArgumentError("EigenAnalysis is not defined for FiniteStateGeneratorProblem; use to_ode_problem or a SciML ODE solve instead."))
end

function CommonSolve.solve(prob::DelayFiniteStateProblem, ::DirectIteration; kwargs...)
    throw(ArgumentError("DirectIteration is not defined for DelayFiniteStateProblem; use a SciML DDE algorithm instead."))
end

function CommonSolve.solve(prob::DelayFiniteStateProblem, ::EigenAnalysis; kwargs...)
    throw(ArgumentError("EigenAnalysis is not defined for DelayFiniteStateProblem; use to_dde_problem or a SciML DDE solve instead."))
end

function _merge_callbacks(prob_cb, user_cb)
    if prob_cb === nothing
        return user_cb
    elseif user_cb === nothing
        return prob_cb
    else
        return SciMLBase.CallbackSet(prob_cb, user_cb)
    end
end

function CommonSolve.solve(prob::FiniteStateGeneratorProblem, alg; kwargs...)
    user_cb = get(kwargs, :callback, nothing)
    merged = _merge_callbacks(prob.callbacks, user_cb)
    rest = Base.structdiff(NamedTuple(kwargs), NamedTuple{(:callback,)})
    if merged === nothing
        return SciMLBase.solve(to_ode_problem(prob), alg; rest...)
    end
    return SciMLBase.solve(to_ode_problem(prob), alg; callback = merged, rest...)
end

function CommonSolve.solve(prob::DelayFiniteStateProblem, alg; kwargs...)
    user_cb = get(kwargs, :callback, nothing)
    merged = _merge_callbacks(prob.callbacks, user_cb)
    rest = Base.structdiff(NamedTuple(kwargs), NamedTuple{(:callback,)})
    if merged === nothing
        return SciMLBase.solve(to_dde_problem(prob), alg; rest...)
    end
    return SciMLBase.solve(to_dde_problem(prob), alg; callback = merged, rest...)
end
