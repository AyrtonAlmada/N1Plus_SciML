using Test
using LinearAlgebra
using DifferentialEquations
using SciMLSensitivity
using Optimization
using OptimizationOptimJL
using ForwardDiff
using Plots

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(REPO_ROOT, "src"))

using N1PlusSciML

@testset "N1Plus-SciML" begin
    net = make_demo_network()

    @test size(net.L) == (3, 3)
    @test size(net.incidence) == (3, 3)
    @test length(net.susceptance) == 3
    @test length(net.inertia) == 3
    @test length(net.damping) == 3

    α = 0.72
    Lf = fault_laplacian(net, α)

    @test size(Lf) == size(net.L)
    @test isapprox(Lf, Lf'; atol=1e-12)

    τ = 0.5
    tspan = (0.0, 1.0)
    saveat = 0.02
    x0 = zeros(6)
    p_true = [0.72, 0.35]

    sol = simulate(
        net,
        p_true;
        tspan=tspan,
        τ=τ,
        x0=x0,
        saveat=saveat,
    )

    @test SciMLBase.successful_retcode(sol)
    @test length(sol.t) > 1
    @test size(Array(sol), 1) == 6

    reference = phase_trajectory(sol, 3)

    @test size(reference, 1) == 3
    @test size(reference, 2) == length(sol.t)

    p_test = [0.40, 0.80]

    loss_value = trajectory_loss(
        net,
        p_test,
        reference;
        tspan=tspan,
        τ=τ,
        x0=x0,
        saveat=saveat,
    )

    @test isfinite(loss_value)
    @test loss_value >= 0

    gradient = ForwardDiff.gradient(
        p -> trajectory_loss(
            net,
            p,
            reference;
            tspan=tspan,
            τ=τ,
            x0=x0,
            saveat=saveat,
        ),
        p_test,
    )

    @test length(gradient) == 2
    @test all(isfinite, gradient)

    # The true parameters should reproduce the reference trajectory.
    true_loss = trajectory_loss(
        net,
        p_true,
        reference;
        tspan=tspan,
        τ=τ,
        x0=x0,
        saveat=saveat,
    )

    @test true_loss < 1e-12
end
