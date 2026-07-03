import DelayDiffEq
import OrdinaryDiffEq

using SciMLBase: DDEProblem, ODEProblem, remake as sciml_remake

@testset "Finite-state generator problems" begin
    domain = DiscreteDomain([:juvenile, :adult])

    @testset "FiniteStateGeneratorProblem to ODEProblem" begin
        G = [-0.5 0.2;
              0.1 -0.3]
        u0 = [1.0, 0.5]
        prob = FiniteStateGeneratorProblem(G, domain, u0, (0.0, 2.0); source = [0.1, 0.0])
        odeprob = to_ode_problem(prob)

        @test odeprob isa ODEProblem
        du = zeros(2)
        odeprob.f(du, odeprob.u0, odeprob.p, 0.0)
        @test du ≈ G * u0 .+ [0.1, 0.0]

        prob2 = remake(prob; tspan = (0.0, 3.0))
        @test prob2.tspan == (0.0, 3.0)
        @test sciml_remake(prob; tspan = (0.0, 4.0)).tspan == (0.0, 4.0)
    end

    @testset "FiniteStateGeneratorProblem with callable generator" begin
        u0 = [1.0, 0.5]
        prob = FiniteStateGeneratorProblem(
            (u, p, t) -> [-p.decay * (1 + t) 0.0; p.decay 0.0],
            domain,
            u0,
            (0.0, 1.0);
            p = (decay = 0.2,),
            source = (u, p, t) -> [0.0, 0.05],
        )

        odeprob = to_ode_problem(prob)
        du = zeros(2)
        odeprob.f(du, odeprob.u0, odeprob.p, 0.5)
        expected = [-0.3 0.0; 0.2 0.0] * u0 .+ [0.0, 0.05]
        @test du ≈ expected
    end

    @testset "Finite-state normalization" begin
        u0 = [1.0, 2.0]
        prob = FiniteStateGeneratorProblem([0.2 0.0; 0.0 -0.1], domain, u0, (0.0, 1.0); normalize = true)
        odeprob = to_ode_problem(prob)
        du = zeros(2)
        odeprob.f(du, odeprob.u0, odeprob.p, 0.0)
        @test sum(du) ≈ 0.0 atol = 1e-10
    end

    @testset "DelayFiniteStateProblem to DDEProblem" begin
        G = [-0.5 0.0;
              0.2 -0.1]
        delay = DelayGeneratorTerm(1.0, [0.0 0.4;
                                         0.0 0.0])
        history(p, t) = [2.0, 1.0]
        u0 = [1.0, 0.5]

        prob = DelayFiniteStateProblem(G, [delay], domain, u0, history, (0.0, 2.0); source = [0.1, 0.0])
        ddeprob = to_dde_problem(prob)

        @test ddeprob isa DDEProblem
        du = zeros(2)
        ddeprob.f(du, ddeprob.u0, ddeprob.h, ddeprob.p, 0.5)
        @test du ≈ G * u0 .+ [0.0 0.4; 0.0 0.0] * history(nothing, -0.5) .+ [0.1, 0.0]

        prob2 = remake(prob; tspan = (0.0, 4.0))
        @test prob2.tspan == (0.0, 4.0)
    end

    @testset "Finite-state solve wrappers" begin
        odeprob = FiniteStateGeneratorProblem(
            [-0.2 0.0;
              0.0 -0.1],
            domain,
            [1.0, 2.0],
            (0.0, 1.0),
        )
        odesol = solve(odeprob, OrdinaryDiffEq.Tsit5(); saveat = 1.0, reltol = 1e-8, abstol = 1e-10)
        @test odesol.u[end] ≈ [exp(-0.2), 2.0 * exp(-0.1)] rtol = 1e-6

        delayprob = DelayFiniteStateProblem(
            [-0.2 0.0;
              0.0 -0.1],
            [DelayGeneratorTerm(0.5, zeros(2, 2))],
            domain,
            [1.0, 2.0],
            (p, t) -> [1.0, 2.0],
            (0.0, 1.0),
        )
        ddesol = solve(delayprob,
            DelayDiffEq.MethodOfSteps(OrdinaryDiffEq.Tsit5());
            saveat = 1.0,
            reltol = 1e-8,
            abstol = 1e-10)
        @test ddesol.u[end] ≈ [exp(-0.2), 2.0 * exp(-0.1)] rtol = 1e-6
    end

    @testset "Finite-state solve dispatch errors" begin
        prob = FiniteStateGeneratorProblem([-0.2 0.0; 0.0 -0.1], domain, [1.0, 1.0], (0.0, 1.0))
        dprob = DelayFiniteStateProblem([-0.2 0.0; 0.0 -0.1], [DelayGeneratorTerm(1.0, zeros(2, 2))],
            domain, [1.0, 1.0], (p, t) -> [1.0, 1.0], (0.0, 1.0))

        @test_throws ArgumentError solve(prob)
        @test_throws ArgumentError solve(prob, DirectIteration())
        @test_throws ArgumentError solve(dprob, EigenAnalysis())
    end

    @testset "Finite-state domain validation" begin
        @test_throws DimensionMismatch FiniteStateGeneratorProblem(zeros(2, 2), domain, [1.0], (0.0, 1.0))
    end
end
