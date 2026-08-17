# N1Plus_SciML_Demo_v1.jl
#
# Differentiable calibration of a reduced transmission-grid model.
#
# This example is intentionally self-contained. It generates a reference
# trajectory, differentiates a trajectory loss through an ODE solve, and
# estimates the model parameters with L-BFGS.

using LinearAlgebra
using DifferentialEquations
using SciMLSensitivity
using Optimization
using OptimizationOptimJL
using ForwardDiff
using Plots

# When this file is run directly from the repository root, the source
# directory is added to LOAD_PATH so that the local module can be loaded.
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(REPO_ROOT, "src"))

using N1PlusSciML

"""
    run_sciml_demo()

Run the complete calibration example and return a named tuple containing
the reference trajectory, optimized parameters, losses, gradient, and
SciML solution.
"""
function run_sciml_demo()
    net = make_demo_network()

    # Simulation settings.
    τ = 0.5
    tspan = (0.0, 1.0)
    saveat = 0.01

    # Initial phase angles and frequencies.
    x0 = zeros(2 * length(net.inertia))

    # Parameters used to generate the reference trajectory.
    p_true = [0.72, 0.35]

    # Deliberately different initial guess for the calibration.
    p_initial = [0.40, 0.80]

    reference_solution = simulate(
        net,
        p_true;
        tspan=tspan,
        τ=τ,
        x0=x0,
        saveat=saveat,
    )

    reference = phase_trajectory(reference_solution, length(net.inertia))

    loss(p) = trajectory_loss(
        net,
        p,
        reference;
        tspan=tspan,
        τ=τ,
        x0=x0,
        saveat=saveat,
    )

    # ForwardDiff is used for the parameter gradient. The ODE solve itself
    # is differentiated with the SciMLSensitivity machinery.
    initial_gradient = ForwardDiff.gradient(loss, p_initial)

    println("N1Plus-SciML differentiable calibration")
    println("----------------------------------------")
    println("True parameters       = ", p_true)
    println("Initial parameters    = ", p_initial)
    println("Initial loss          = ", loss(p_initial))
    println("Initial gradient      = ", initial_gradient)

    optf = OptimizationFunction(
        (p, _) -> loss(p),
        Optimization.AutoForwardDiff(),
    )

    optprob = OptimizationProblem(optf, p_initial)

    result = Optimization.solve(
        optprob,
        OptimizationOptimJL.LBFGS();
        maxiters=100,
        abstol=1e-10,
        reltol=1e-10,
    )

    p_est = Vector(result.u)
    final_loss = loss(p_est)

    calibrated_solution = simulate(
        net,
        p_est;
        tspan=tspan,
        τ=τ,
        x0=x0,
        saveat=saveat,
    )

    calibrated = phase_trajectory(
        calibrated_solution,
        length(net.inertia),
    )

    # Relative squared error at the fault-clearing time.
    kτ = argmin(abs.(collect(reference_solution.t) .- τ))
    reference_at_clear = reference[:, kτ]
    calibrated_at_clear = calibrated[:, kτ]

    rse_clear =
        sum(abs2, calibrated_at_clear .- reference_at_clear) /
        (sum(abs2, reference_at_clear) + eps(Float64))

    println()
    println("Estimated parameters = ", p_est)
    println("Final loss           = ", final_loss)
    println("Fault-clearing time  = ", τ)
    println("RSE at clearing      = ", rse_clear)

    # Plot the phase-angle trajectories.
    plt = plot(
        xlabel="Time [s]",
        ylabel="Phase angle",
        title="N1Plus-SciML: Reference vs. Calibrated Trajectory",
        legend=:best,
    )

    for i in 1:size(reference, 1)
        plot!(
            plt,
            reference_solution.t,
            reference[i, :],
            label="Reference θ$i",
            lw=2,
        )
        plot!(
            plt,
            calibrated_solution.t,
            calibrated[i, :],
            label="Calibrated θ$i",
            ls=:dash,
            lw=2,
        )
    end

    display(plt)

    return (
        network=net,
        reference_solution=reference_solution,
        calibrated_solution=calibrated_solution,
        reference=reference,
        calibrated=calibrated,
        p_true=p_true,
        p_initial=p_initial,
        p_estimated=p_est,
        initial_gradient=initial_gradient,
        initial_loss=loss(p_initial),
        final_loss=final_loss,
        clearing_time=τ,
        clearing_rse=rse_clear,
        plot=plt,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_sciml_demo()
end
