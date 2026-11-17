import ForwardDiff
import OrdinaryDiffEq
import ResilienceCore
using LinearAlgebra

@testset "linearise: resilience operators" begin
    domain = DiscreteDomain([:juvenile, :adult])
    Q = [-0.5 0.1;
          0.3 -0.2]

    @testset "constant generator → Generator" begin
        op = linearise(FiniteStateGeneratorProblem(Q, domain, [1.0, 1.0], (0.0, 1.0)))
        @test op isa ResilienceCore.Generator
        @test ResilienceCore.operator(op) == Q
        @test ResilienceCore.growth_rate(op) ≈ maximum(real, eigvals(Q))
        r = sort(real.(eigvals(Q)); rev = true)
        @test ResilienceCore.resilience(op) ≈ r[1] - r[2]
        # a callable that depends on neither state nor time is constant too
        @test ResilienceCore.operator(linearise(FiniteStateGeneratorProblem((u, p, t) -> Q, domain, [1.0, 1.0],
                                                                            (0.0, 1.0)))) == Q
        # normalised (proportions): still the generator
        @test linearise(FiniteStateGeneratorProblem(Q, domain, [0.5, 0.5], (0.0, 1.0); normalize = true)) isa
              ResilienceCore.Generator
        # a zero source is no source
        @test linearise(FiniteStateGeneratorProblem(Q, domain, [1.0, 1.0], (0.0, 1.0); source = zeros(2))) isa
              ResilienceCore.Generator
    end

    @testset "constant generator and source → Jacobian at u*" begin
        b = [0.2, 0.0]
        prob = FiniteStateGeneratorProblem(Q, domain, [1.0, 1.0], (0.0, 200.0); source = b)
        J = linearise(prob)
        @test J isa ResilienceCore.ContinuousJacobian
        @test ResilienceCore.operator(J) == Q
        u = J.equilibrium
        @test Q * u + b ≈ zeros(2) atol = 1e-12
        @test ResilienceCore.isstable(J)
        sol = OrdinaryDiffEq.solve(to_ode_problem(prob), OrdinaryDiffEq.Tsit5(); reltol = 1e-10, abstol = 1e-12)
        @test sol.u[end] ≈ u rtol = 1e-6                     # the solution settles there
        # unstable generator: no attracting equilibrium
        @test_throws ArgumentError linearise(FiniteStateGeneratorProblem([0.1 0.0; 0.2 -0.3], domain, [1.0, 1.0],
                                                                         (0.0, 1.0); source = b))
    end

    @testset "periodic generator → PeriodicOperator" begin
        T = 2.0
        f(t) = 1 + 0.5 * sin(π * t / T)                      # ∫₀ᵀ f = T + T/π
        exact = exp(Q * (T + T / π))
        prob = FiniteStateGeneratorProblem((p, t) -> f(t) * Q, domain, [1.0, 1.0], (0.0, T))
        M(k) = ResilienceCore.operator(ResilienceCore.monodromy(linearise(prob; period = T, steps = k)))
        e = [norm(M(k) - exact) for k in (10, 20, 40, 80)]
        @test all(e[i + 1] < e[i] for i in 1:3)
        @test norm(M(400) - exact) < 1e-5
        @test linearise(prob; period = T, steps = 7) isa ResilienceCore.PeriodicOperator
        @test ResilienceCore.period(linearise(prob; period = T, steps = 7)) == 7
        # the (u, p, t) form that ignores u is recognised as time-dependent, not density-dependent
        prob_upt = FiniteStateGeneratorProblem((u, p, t) -> f(t) * Q, domain, [1.0, 1.0], (0.0, T))
        @test ResilienceCore.operator(ResilienceCore.monodromy(linearise(prob_upt; period = T, steps = 40))) ≈ M(40)
        # time dependence without a period is refused
        @test_throws ArgumentError linearise(prob)
    end

    @testset "density-dependent → SciML route" begin
        m, a, μ, β = 0.4, 0.05, 0.3, 2.0           # maturation slowed by adults; recruits from outside
        G(u, p, t) = [-(m + a * u[2]) 0.0; m -μ]
        c = a * m / μ                               # β = m u₁ + c u₁²
        u1 = (-m + sqrt(m^2 + 4c * β)) / (2c)
        ustar = [u1, m * u1 / μ]
        prob = FiniteStateGeneratorProblem(G, domain, [1.0, 1.0], (0.0, 1.0); source = [β, 0.0])
        J = linearise(prob, ustar)
        @test J isa ResilienceCore.ContinuousJacobian
        ode = to_ode_problem(prob)
        rhs(u) = (du = similar(u); ode.f(du, u, ode.p, 0.0); du)
        h = 1e-6
        Jfd = reduce(hcat, [(rhs(ustar .+ h .* (1:2 .== j)) .- rhs(ustar .- h .* (1:2 .== j))) ./ 2h for j in 1:2])
        @test ResilienceCore.operator(J) ≈ Jfd rtol = 1e-6
        @test ResilienceCore.isstable(J)
        @test_throws ArgumentError linearise(prob)               # needs the equilibrium
        @test_throws ArgumentError linearise(prob, [1.0, 1.0])   # not an equilibrium
    end

    @testset "no single operator" begin
        delay = DelayGeneratorTerm(1.0, [0.0 0.4; 0.0 0.0])
        dprob = DelayFiniteStateProblem(Q, [delay], domain, [1.0, 0.5], (p, t) -> [1.0, 0.5], (0.0, 2.0))
        @test_throws ArgumentError linearise(dprob)
        sys = DemographicReactionSystem(1, [DemographicReaction((n, p, t) -> 0.5 * n[1], 1, 1 => +1)])
        @test_throws ArgumentError linearise(FiniteStateReactionProblem(sys, [10], (0.0, 1.0)))
    end
end
