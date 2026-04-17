"""
Event/callback helpers for finite-state continuous-time dynamics.

These wrap `SciMLBase.DiscreteCallback` / `SciMLBase.ContinuousCallback` with
domain-aware state accessors so users can write events in terms of stage names
rather than numeric indices.
"""

"""
    state_index(domain, label)

Return the 1-based integer index of `label` in a `DiscreteDomain`. Accepts any
`AbstractString`/`Symbol` label and throws `KeyError` if not found. Integer
labels are returned as-is.
"""
function state_index(domain, label::Integer)
    1 <= label <= _finite_state_dimension(domain) ||
        throw(BoundsError(domain, label))
    return Int(label)
end

function state_index(domain::DiscreteDomain, label::Symbol)
    idx = findfirst(==(label), domain.labels)
    idx === nothing && throw(KeyError(label))
    return idx
end

function state_index(domain::DiscreteDomain, label::AbstractString)
    return state_index(domain, Symbol(label))
end

"""
    set_state!(integrator, domain, label, value)

Assign `integrator.u[state_index(domain, label)] = value` during an event.
"""
function set_state!(integrator, domain, label, value)
    integrator.u[state_index(domain, label)] = value
    return integrator
end

"""
    add_to_state!(integrator, domain, label, amount)

Add `amount` to `integrator.u[state_index(domain, label)]` during an event.
"""
function add_to_state!(integrator, domain, label, amount)
    integrator.u[state_index(domain, label)] += amount
    return integrator
end

"""
    scale_state!(integrator, domain, label, factor)

Multiply `integrator.u[state_index(domain, label)]` by `factor`.
"""
function scale_state!(integrator, domain, label, factor)
    integrator.u[state_index(domain, label)] *= factor
    return integrator
end

"""
    transfer_state!(integrator, domain, from_label, to_label, amount)

Move `amount` from one stage to another, clamped to the source stage's
current value.
"""
function transfer_state!(integrator, domain, from_label, to_label, amount)
    from = state_index(domain, from_label)
    to = state_index(domain, to_label)
    actual = min(integrator.u[from], amount)
    integrator.u[from] -= actual
    integrator.u[to] += actual
    return integrator
end

"""
    scheduled_event(times, affect!; save_positions = (true, true))

Build a `SciMLBase.DiscreteCallback` that fires whenever the solver passes
through (or exactly hits) any time in `times`. `affect!(integrator)` is called
at each such time. `times` may be a scalar or an iterable of times.

The callback uses `tstops` semantics — callers should also pass `tstops = times`
to `solve` to guarantee each event time is exactly visited.
"""
function scheduled_event(times, affect!; save_positions = (true, true))
    time_set = Set(Float64.(collect(Iterators.flatten((times,)))))
    condition(u, t, integrator) = t in time_set
    return SciMLBase.DiscreteCallback(condition, affect!;
        save_positions = save_positions)
end

"""
    periodic_event(period, affect!; t0 = 0.0, save_positions = (true, true))

Build a `SciMLBase.DiscreteCallback` that fires on a regular schedule
`t0, t0+period, t0+2*period, ...`. Callers should pass matching `tstops` to
`solve` so the events are not skipped by adaptive time stepping.
"""
function periodic_event(period::Real, affect!; t0::Real = 0.0,
        save_positions = (true, true))
    period > 0 || throw(ArgumentError("period must be positive"))
    t0f = float(t0)
    periodf = float(period)
    function condition(u, t, integrator)
        t < t0f && return false
        rel = (t - t0f) / periodf
        rounded = round(rel)
        return abs(rel - rounded) < 1e-12 && rounded >= 0
    end
    return SciMLBase.DiscreteCallback(condition, affect!;
        save_positions = save_positions)
end

"""
    threshold_event(condition, affect!; save_positions = (true, true), kwargs...)

Build a `SciMLBase.ContinuousCallback` that fires when the scalar-valued
`condition(u, t, integrator)` crosses zero. Extra kwargs are forwarded to
`ContinuousCallback` (e.g., `rootfind`, `interp_points`).
"""
function threshold_event(condition, affect!; save_positions = (true, true), kwargs...)
    return SciMLBase.ContinuousCallback(condition, affect!;
        save_positions = save_positions, kwargs...)
end

"""
    combine_callbacks(cbs...)

Combine any number of SciML callbacks into a single `CallbackSet`. `nothing`
entries are skipped. Returns `nothing` when all arguments are `nothing`.
"""
function combine_callbacks(cbs...)
    filtered = Any[cb for cb in cbs if cb !== nothing]
    isempty(filtered) && return nothing
    length(filtered) == 1 && return filtered[1]
    return SciMLBase.CallbackSet(filtered...)
end
