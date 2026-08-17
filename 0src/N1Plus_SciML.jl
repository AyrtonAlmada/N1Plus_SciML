###############################################################
# N1Plus_SciML.jl
#
# Differentiable calibration example for the N1Plus power-grid
# dynamic contingency framework.
#
# This file keeps the physics-based swing-equation viewpoint of
# N1Plus and adds a small SciML workflow:
#
#     physical model -> ODEProblem -> sensitivity -> optimization
#
# The example is intentionally self-contained. It does not require
# the IEEE/Israeli-grid data files, so it can be run immediately
# after installing the packages listed below.
#
# Required packages
# -----------------
# ] add DifferentialEquations SciMLSensitivity Optimization
#       OptimizationOptimJL ForwardDiff Plots LinearAlgebra
#
# Main demonstration
# ------------------
# run_sciml_demo()
#
# The demonstration generates a reference trajectory with known
# parameters and then estimates:
#
#     alpha : fault/line-strength scaling
#     sigma : amplitude of unresolved forcing
#
# from the trajectory alone.
#
# This is a minimal "learning-to-model" SciML example connected
# directly to the N1Plus research problem.
###############################################################

using LinearAlgebra
using DifferentialEquations
using SciMLSensitivity
using Optimization
using OptimizationOptimJL
using ForwardDiff
using Plots

###############################################################
# 1. NETWORK MODEL
###############################################################

"""
    weighted_laplacian(n, edges, susceptance)

Construct the weighted graph Laplacian

    L = B' * diag(beta) * B

for a transmission network.

`edges` is a vector of `(from, to)` bus pairs using 1-based indexing.
"""
function weighted_laplacian(
    n::Int,
    edges::Vector{<:Tuple},
    susceptance::AbstractVector
)
    B = zeros(eltype(susceptance), length(edges), n)

    for (k, (i, j)) in enumerate(edges)
        B[k, i] = 1
        B[k, j] = -1
    end

    return B' * Diagonal(susceptance) * B
end


"""
    faulted_laplacian(L, edges, fault_line, alpha)

Return a Laplacian in which the susceptance of one line is scaled
by `alpha`.

For example, `alpha = 2/3` represents the partial line-strength
reduction used in the original N1Plus single-phase-fault model.
"""
function faulted_laplacian(
    L::AbstractMatrix,
    edges::Vector{<:Tuple},
    fault_line::Int,
    alpha::Real
)
    n = size(L, 1)

    i, j = edges[fault_line]

    # Laplacian contribution of the selected line.
    e = zeros(eltype(L), n)
    e[i] = 1
    e[j] = -1

    # Changing beta -> alpha*beta changes L by
    # (alpha - 1) beta e e'.
    #
    # The demo stores L and uses a unit-normalized line contribution.
    # The physical-data version below allows the actual susceptance
    # to be supplied explicitly.
    return L + (alpha - 1) * (e * e')
end


"""
    make_swing_matrices(M, D, L)

Construct the first-order linear swing-equation system

    d/dt [omega; theta]
        = [ -M^-1 D   -M^-1 L ] [omega; theta]
          [    I          0   ] [omega; theta]
          + [M^-1 P; 0].

The explicit form is used in the SciML demonstration because it
allows DifferentialEquations.jl to work directly with the state.
"""
function make_swing_matrices(
    M::AbstractMatrix,
    D::AbstractMatrix,
    L::AbstractMatrix
)
    n = size(L, 1)

    Minv = inv(M)

    A = zeros(eltype(L), 2n, 2n)
    A[1:n, 1:n] = -Minv * D
    A[1:n, n+1:2n] = -Minv * L
    A[n+1:2n, 1:n] = I(n)

    return A
end


###############################################################
# 2. DIFFERENTIABLE SWING DYNAMICS
###############################################################

"""
    unresolved_forcing(t)

A deterministic forcing realization used for the demonstration.

Using a fixed realization is intentional: it makes the calibration
problem reproducible and differentiable. In the full N1Plus model,
this term can be replaced by a stochastic process.
"""
function unresolved_forcing(t)
    return [
        sin(2π * 0.7 * t),
        cos(2π * 0.4 * t),
        sin(2π * 1.1 * t)
    ]
end


"""
    build_problem(p; ...)

Create a DifferentialEquations.jl ODEProblem for the N1Plus-style
linearized swing dynamics.

Parameters
----------
p[1] = alpha
    Scaling of the faulted-line contribution.

p[2] = sigma
    Amplitude of unresolved forcing.

The forcing is deliberately simple so that the example is easy
to inspect and reproduce.
"""
function build_problem(
    p;
    tspan = (0.0, 5.0),
    saveat = 0.02,
    fault_line = 2
)
    alpha = p[1]
    sigma = p[2]

    # Small 3-bus demonstration network.
    n = 3

    edges = [(1, 2), (2, 3), (1, 3)]
    beta = [4.0, 3.0, 2.0]

    L = weighted_laplacian(n, edges, beta)

    # Apply the fault parameter to one line using its actual
    # susceptance contribution.
    i, j = edges[fault_line]
    e = zeros(eltype(p), n)
    e[i] = 1
    e[j] = -1

    Lfault = L + (alpha - 1) * beta[fault_line] * (e * e')

    M = Diagonal([4.0, 3.0, 2.5])
    D = Diagonal([0.8, 0.7, 0.6])

    P = [0.25, -0.10, -0.15]

    Minv = inv(M)

    function rhs!(du, u, p, t)
        α = p[1]
        σ = p[2]

        # Reconstruct the faulted Laplacian from the current
        # optimization parameter. This is what makes the forward
        # simulation a differentiable function of α.
        Lα = L + (α - 1) * beta[fault_line] * (e * e')

        ω = @view u[1:n]
        θ = @view u[n+1:2n]

        du[1:n] .= Minv * (
            P - D * ω - Lα * θ + σ * unresolved_forcing(t)
        )

        du[n+1:2n] .= ω

        return nothing
    end

    # Start from the equilibrium of the pre-fault network.
    θ0 = pinv(L) * P
    u0 = vcat(zeros(eltype(p), n), θ0)

    return ODEProblem(rhs!, u0, tspan, p)
end


###############################################################
# 3. FORWARD SIMULATION
###############################################################

"""
    simulate_phase(p; ...)

Solve the differentiable swing equation and return the phase
trajectory.

The output is an `n × nt` matrix, where each row corresponds to
one bus.
"""
function simulate_phase(
    p;
    tspan = (0.0, 5.0),
    saveat = 0.02
)
    prob = build_problem(p; tspan=tspan, saveat=saveat)

    # Tsit5 is appropriate for this small non-stiff demonstration.
    #
    # SciMLSensitivity provides the sensitivity machinery used by
    # SciML when differentiating objectives through the ODE solve.
    sol = solve(
        prob,
        Tsit5();
        saveat=saveat,
        abstol=1e-9,
        reltol=1e-9,
        sensealg=ForwardDiffSensitivity()
    )

    n = 3
    return Array(sol)[n+1:2n, :]
end


###############################################################
# 4. LEARNING / PARAMETER IDENTIFICATION
###############################################################

"""
    trajectory_loss(p, reference)

Mean-squared error between the model trajectory and a reference
trajectory.

This is the SciML learning objective:

    p* = argmin_p ||theta_model(p) - theta_reference||².
"""
function trajectory_loss(p, reference)
    predicted = simulate_phase(p)

    return sum(abs2, predicted .- reference) / length(reference)
end


"""
    fit_parameters(reference; initial=[...])

Estimate the model parameters from a reference trajectory.

The optimization is performed through the differentiable ODE
simulation rather than by treating the simulator as a black box.
"""
function fit_parameters(
    reference;
    initial = [0.80, 0.05]
)
    optf = OptimizationFunction(
        (p, _) -> trajectory_loss(p, reference),
        Optimization.AutoForwardDiff()
    )

    optprob = OptimizationProblem(
        optf,
        initial
    )

    # BFGS uses gradients supplied by ForwardDiff. Since the loss
    # contains a differentiable ODE solve, the resulting gradient
    # propagates through the dynamics.
    result = Optimization.solve(
        optprob,
        BFGS();
        maxiters=100,
        abstol=1e-8,
        reltol=1e-8
    )

    return result
end


###############################################################
# 5. COMPLETE WORKING EXAMPLE
###############################################################

"""
    run_sciml_demo()

Run the complete self-contained SciML example.

The reference system is generated with

    alpha_true = 2/3
    sigma_true = 0.10

and the optimizer starts from a deliberately different point.

The function prints the recovered parameters and produces a plot
comparing the reference and fitted trajectories.
"""
function run_sciml_demo()
    println()
    println("============================================================")
    println(" N1Plus + SciML: differentiable parameter identification")
    println("============================================================")

    α_true = 2 / 3
    σ_true = 0.10

    p_true = [α_true, σ_true]
    p_initial = [0.90, 0.02]

    println("True parameters:")
    println("    alpha = ", α_true)
    println("    sigma = ", σ_true)

    println()
    println("Generating reference trajectory...")

    reference = simulate_phase(p_true)

    println("Reference trajectory generated.")
    println()
    println("Estimating parameters through the ODE solve...")

    result = fit_parameters(
        reference;
        initial=p_initial
    )

    p_est = result.u

    println()
    println("Optimization finished.")
    println("Initial parameters:")
    println("    alpha = ", p_initial[1])
    println("    sigma = ", p_initial[2])

    println()
    println("Estimated parameters:")
    println("    alpha = ", p_est[1])
    println("    sigma = ", p_est[2])

    println()
    println("Final trajectory loss = ", result.objective)

    fitted = simulate_phase(p_est)

    t = range(0.0, 5.0; length=size(reference, 2))

    plt = plot(
        t,
        reference[1, :];
        linewidth=2,
        label="Reference — bus 1",
        xlabel="Time [s]",
        ylabel="Phase angle",
        title="N1Plus SciML parameter identification"
    )

    plot!(
        plt,
        t,
        fitted[1, :];
        linewidth=2,
        linestyle=:dash,
        label="Fitted model — bus 1"
    )

    display(plt)

    return (
        result=result,
        true_parameters=p_true,
        estimated_parameters=p_est,
        reference=reference,
        fitted=fitted,
        plot=plt
    )
end


###############################################################
# 6. OPTIONAL: CONNECTION TO REAL N1PLUS DATA
###############################################################

"""
    calibrate_swing_parameters(M, D, L, P, reference, initial)

Template for extending the demonstration to an actual N1Plus
network.

The intended workflow is:

1. Obtain a high-fidelity trajectory from ParaEMT/EMT or another
   validated simulator.
2. Construct the corresponding reduced swing-equation model.
3. Replace the small demonstration network in `build_problem`.
4. Pass the measured/reference trajectory to the same
   `fit_parameters` idea.
5. Estimate physically meaningful parameters such as fault
   severity, damping, inertia, or unresolved forcing amplitude.

This function is intentionally left as a template because the
mapping between the user's actual data columns and the reduced
state vector depends on the particular grid dataset.
"""
function calibration_workflow_description()
    println("""
    Real-data extension
    --------------------

    High-fidelity trajectory
            |
            v
    reference = theta_EMT(t)
            |
            v
    differentiable swing model
            |
            v
    Optimization.jl
            |
            v
    calibrated parameters
    """)
end


###############################################################
# 7. OPTIONAL NETWORK UTILITIES
###############################################################

"""
    line_phase_differences(theta, edges)

Compute phase differences across all transmission lines.

`theta` must have size `n × nt`.
"""
function line_phase_differences(theta, edges)
    result = Matrix{eltype(theta)}(undef, length(edges), size(theta, 2))

    for (k, (i, j)) in enumerate(edges)
        result[k, :] .= theta[i, :] .- theta[j, :]
    end

    return result
end


"""
    overload_indicator(theta, edges, theta_max)

Compute the cumulative overload indicator

    S = sum_k ∫ 1{|theta_i-theta_j| > theta_max} dt

using a uniform time grid.

This is a compact SciML-compatible analogue of the overload
indicators used in the original N1Plus framework.
"""
function overload_indicator(
    theta,
    edges,
    theta_max,
    dt
)
    differences = line_phase_differences(theta, edges)

    indicator = abs.(differences) .> theta_max

    return sum(indicator) * dt
end


###############################################################
# 8. SCRIPT ENTRY POINT
###############################################################

if abspath(PROGRAM_FILE) == @__FILE__
    run_sciml_demo()
end
