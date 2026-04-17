using Test
using FiniteStatePopulationDynamics
using OrdinaryDiffEq
using SciMLBase

@testset "Event helpers" begin
    domain = DiscreteDomain([:juvenile, :adult])

    @testset "state_index" begin
        @test state_index(domain, :juvenile) == 1
        @test state_index(domain, :adult) == 2
        @test state_index(domain, "adult") == 2
        @test state_index(domain, 1) == 1
        @test_throws KeyError state_index(domain, :missing)
        @test_throws BoundsError state_index(domain, 3)
    end

    @testset "state mutators" begin
        # Build a minimal integrator-like struct
        mutable struct _FakeIntegrator{U}
            u::U
        end
        integ = _FakeIntegrator([1.0, 2.0])

        set_state!(integ, domain, :juvenile, 5.0)
        @test integ.u == [5.0, 2.0]

        add_to_state!(integ, domain, :adult, 1.5)
        @test integ.u == [5.0, 3.5]

        scale_state!(integ, domain, :juvenile, 0.5)
        @test integ.u == [2.5, 3.5]

        transfer_state!(integ, domain, :juvenile, :adult, 1.0)
        @test integ.u == [1.5, 4.5]

        # Transfer clamps to source
        transfer_state!(integ, domain, :juvenile, :adult, 10.0)
        @test integ.u[1] == 0.0
        @test integ.u[2] == 6.0
    end
end

@testset "Scheduled events integrate with solve" begin
    domain = DiscreteDomain([:juvenile, :adult])
    G = [-0.2 0.0; 0.1 -0.1]

    harvest_time = 5.0
    affect! = integ -> scale_state!(integ, domain, :adult, 0.5)
    cb = scheduled_event(harvest_time, affect!)

    prob = FiniteStateGeneratorProblem(G, domain, [10.0, 10.0], (0.0, 10.0);
        callbacks = cb)
    sol = solve(prob, Tsit5(); tstops = [harvest_time])

    before = sol(harvest_time - 1e-6)
    after = sol(harvest_time + 1e-6)
    @test after[2] ≈ 0.5 * before[2] atol = 1e-3
end

@testset "Periodic pulse fires repeatedly" begin
    domain = DiscreteDomain([:adult])
    G = reshape([-0.1], 1, 1)
    pulse_times = 1.0:1.0:4.0

    affect! = integ -> add_to_state!(integ, domain, :adult, 5.0)
    cb = periodic_event(1.0, affect!; t0 = 1.0)

    prob = FiniteStateGeneratorProblem(G, domain, [1.0], (0.0, 5.0);
        callbacks = cb)
    sol = solve(prob, Tsit5(); tstops = collect(pulse_times))

    # Compare at t=5 vs. unpulsed baseline
    prob_no_pulse = FiniteStateGeneratorProblem(G, domain, [1.0], (0.0, 5.0))
    sol_base = solve(prob_no_pulse, Tsit5())

    @test sol.u[end][1] > sol_base.u[end][1] + 10.0
end

@testset "Threshold event fires on state crossing" begin
    domain = DiscreteDomain([:population])
    G = reshape([0.5], 1, 1)

    fired = Ref(false)
    function condition(u, t, integrator)
        return u[1] - 5.0
    end
    function affect!(integrator)
        fired[] = true
        set_state!(integrator, domain, :population, 5.0)
    end
    cb = threshold_event(condition, affect!)

    prob = FiniteStateGeneratorProblem(G, domain, [1.0], (0.0, 10.0);
        callbacks = cb)
    sol = solve(prob, Tsit5())

    @test fired[]
    @test sol.u[end][1] ≈ 5.0 * exp(0.5 * (sol.t[end] - sol.t[findfirst(t -> t >= log(5.0) / 0.5, sol.t)])) rtol = 0.2
end

@testset "User callback merges with problem callback" begin
    domain = DiscreteDomain([:a, :b])
    G = zeros(2, 2)

    prob_affect_calls = Ref(0)
    user_affect_calls = Ref(0)

    prob_cb = scheduled_event(1.0, integ -> (prob_affect_calls[] += 1))
    user_cb = scheduled_event(2.0, integ -> (user_affect_calls[] += 1))

    prob = FiniteStateGeneratorProblem(G, domain, [1.0, 1.0], (0.0, 3.0);
        callbacks = prob_cb)
    sol = solve(prob, Tsit5(); callback = user_cb, tstops = [1.0, 2.0])

    @test prob_affect_calls[] == 1
    @test user_affect_calls[] == 1
end

@testset "combine_callbacks" begin
    @test combine_callbacks(nothing, nothing) === nothing
    cb = scheduled_event(1.0, integ -> nothing)
    @test combine_callbacks(nothing, cb) === cb
    @test combine_callbacks(cb, nothing) === cb
    combined = combine_callbacks(cb, cb)
    @test combined isa SciMLBase.CallbackSet
end

@testset "remake preserves and overrides callbacks" begin
    domain = DiscreteDomain([:a, :b])
    G = zeros(2, 2)
    cb1 = scheduled_event(1.0, integ -> nothing)
    cb2 = scheduled_event(2.0, integ -> nothing)

    prob = FiniteStateGeneratorProblem(G, domain, [1.0, 1.0], (0.0, 3.0);
        callbacks = cb1)
    @test prob.callbacks === cb1

    prob2 = FiniteStatePopulationDynamics.remake(prob)
    @test prob2.callbacks === cb1

    prob3 = FiniteStatePopulationDynamics.remake(prob; callbacks = cb2)
    @test prob3.callbacks === cb2
end
