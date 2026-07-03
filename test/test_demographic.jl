using Random
using LinearAlgebra
using StructuredPopulationCore: quasi_extinction

_mean(xs) = sum(xs) / length(xs)

@testset "Demographic stochasticity (finite-state CTMC)" begin
    rng = Random.Xoshiro(7)

    @testset "migration CTMC: mean tracks exp(Gt)·n0, total conserved" begin
        G = [-0.8 0.4; 0.8 -0.4]          # conservative A <-> B
        n0 = [100, 0]
        prob = FiniteStateGeneratorProblem(G, 2, n0, (0.0, 2.0))
        grid = 0.0:0.5:2.0
        reps = 4000
        acc = [zeros(2) for _ in grid]
        conserved = true
        for _ in 1:reps
            s = solve(prob, Demographic(); rng=rng, saveat=grid)
            for g in eachindex(grid)
                acc[g] .+= s.u[g]
            end
            all(sum(u) == 100 for u in s.u) || (conserved = false)
        end
        @test conserved                                   # migration conserves total N
        for (g, t) in enumerate(grid)
            @test isapprox(acc[g] ./ reps, exp(G .* t) * n0; rtol=0.05, atol=1.0)
        end
    end

    @testset "explicit birth-death reactions: mean and extinction" begin
        b, d = 0.6, 0.9                                   # subcritical
        sys = DemographicReactionSystem(1, [
            DemographicReaction((n, p, t) -> b * n[1], 1, 1 => +1),
            DemographicReaction((n, p, t) -> d * n[1], 1, 1 => -1),
        ])
        prob = FiniteStateReactionProblem(sys, [40], (0.0, 3.0))
        reps = 3000
        finals = [sum(solve(prob, Demographic(); rng=rng).u[end]) for _ in 1:reps]
        @test isapprox(_mean(finals), 40 * exp((b - d) * 3.0); rtol=0.07)

        prob2 = FiniteStateReactionProblem(sys, [40], (0.0, 30.0))
        totals, _ = demographic_ensemble(prob2; n_reps=400, saveat=1.0, rng=rng)
        @test size(totals, 1) == 31
        @test quasi_extinction(totals; threshold=1.0).prob_extinct > 0.7
    end

    @testset "exact solves validate integer initial counts" begin
        sys = DemographicReactionSystem(1, [DemographicReaction(1.0, 1, 1 => +1)])
        prob = FiniteStateReactionProblem(sys, [10.0], (0.0, 0.01))
        @test prob.u0 == [10]
        @test solve(prob, Demographic(); rng=rng).u[1] == [10]
        @test_throws ArgumentError FiniteStateReactionProblem(sys, [10.25], (0.0, 0.01))
    end

    @testset "retcode reports max_events truncation" begin
        sys = DemographicReactionSystem(1, [DemographicReaction(1000.0, 1, 1 => +1)])
        prob = FiniteStateReactionProblem(sys, [1], (0.0, 1.0))
        sol = solve(prob, Demographic(); rng=rng, max_events=1)
        @test sol.retcode == :MaxIters
        @test length(sol.t) == 2
    end

    @testset "generator_reactions reproduces dn/dt = G·n in mean (with growth)" begin
        G = [0.2 0.5; 0.3 -0.7]                           # net growth (dominant eigenvalue > 0)
        n0 = [20, 20]
        prob = FiniteStateGeneratorProblem(G, 2, n0, (0.0, 1.5))
        reps = 5000
        acc = zeros(2)
        for _ in 1:reps
            acc .+= solve(prob, Demographic(); rng=rng, saveat=[1.5]).u[end]
        end
        @test isapprox(acc ./ reps, exp(G .* 1.5) * n0; rtol=0.06, atol=1.0)
    end

    @testset "errors" begin
        cprob = FiniteStateGeneratorProblem((u, p, t) -> [-0.1 0.0; 0.0 -0.1], 2,
            [10, 10], (0.0, 1.0))
        @test_throws ArgumentError solve(cprob, Demographic())          # callable generator
        @test_throws ArgumentError generator_reactions([-0.5 -0.2; 0.0 -0.3])  # negative rate
    end
end
