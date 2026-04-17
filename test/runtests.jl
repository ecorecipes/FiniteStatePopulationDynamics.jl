using Test
using FiniteStatePopulationDynamics

@testset "FiniteStatePopulationDynamics" begin
    @test SimpleFiniteStateStructure() isa AbstractFiniteStateDynamicsStructure
    @test GeneralFiniteStateStructure() isa AbstractFiniteStateDynamicsStructure
    @test FiniteState() isa AbstractStateSemantics
    @test ContinuousTime() isa AbstractTimeSemantics
    @test n_states(DiscreteDomain([:juvenile, :adult])) == 2
    include("test_generators.jl")
    include("test_events.jl")
end
