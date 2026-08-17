# N1Plus-SciML

## Repository Structure

``` text
N1Plus-SciML/
│
├── LICENSE
├── Project.toml
├── README.md
│
├── 0src/
│   └── N1PlusSciML.jl
│
├── 1examples/
│   └── N1Plus_SciML_Demo_v1.jl
│
├── 2test/
│   └── runtests.jl
│
└── 3figures/
    └── calibration_example.png
```
------------------------------------------------------------------------

## Differentiable Calibration of a Physics-Based Power-Grid Surrogate

This repository contains a compact **Scientific Machine Learning (SciML)** implementation of the **N1Plus** power-grid surrogate model.

The example demonstrates how a physics-based dynamical model can be embedded in a differentiable simulation and optimization workflow. The implementation is connected to the research project: **Differentiable Hybrid Surrogates for Learning, Optimization, and Control of Power-Grid Dynamics**

The computational workflow is:

``` text
Reference trajectory
        |
        v
Physics-based N1Plus model
        |
        v
Differentiable ODE simulation
        |
        v
Trajectory loss
        |
        v
Automatic differentiation
        |
        v
Gradient-based optimization
        |
        v
Calibrated surrogate
```

------------------------------------------------------------------------

## Scientific Motivation

High-fidelity electromagnetic transient (EMT) simulations provide detailed information about transmission-grid dynamics, but their computational cost makes repeated simulation expensive.

The N1Plus framework provides a reduced representation of transmission-grid dynamics based on swing-equation structure and network topology. In the original research, the surrogate was calibrated against high-fidelity EMT simulations for dynamic contingency assessment.

This repository takes the next step by making the reduced dynamical model **differentiable**.

The model parameters are exposed to an optimization layer,

$$
p \longrightarrow x(t;p) \longrightarrow \mathcal{L}(p),
$$

and automatic differentiation provides

$$
\nabla_p \mathcal{L}.
$$

This creates a simple physics-based SciML workflow in which known physical structure is retained while model parameters are learned from trajectory data.

------------------------------------------------------------------------

## What the Example Demonstrates

`N1Plus_SciML_Demo_v1.jl` demonstrates:

1.  Construction of a reduced transmission-network Laplacian;
2.  Representation of a transmission-line fault as a structured perturbation;
3.  Definition of reduced swing dynamics;
4.  Differentiable ODE simulation;
5.  Construction of a dimensionless trajectory loss;
6.  Sensitivity calculation through the ODE solve;
7.  Automatic differentiation with `ForwardDiff.jl`;
8.  Parameter optimization with `Optimization.jl` and L-BFGS;
9.  Comparison of reference and calibrated phase-angle trajectories;
10. Evaluation of relative squared error at fault clearing.

------------------------------------------------------------------------

## Requirements

-   Julia 1.10 or later
-   `LinearAlgebra`
-   `DifferentialEquations`
-   `SciMLSensitivity`
-   `Optimization`
-   `OptimizationOptimJL`
-   `ForwardDiff`
-   `Plots`

------------------------------------------------------------------------

## Installation

### Install the required packages

Run:

``` julia
using Pkg

Pkg.add([
    "DifferentialEquations",
    "SciMLSensitivity",
    "Optimization",
    "OptimizationOptimJL",
    "ForwardDiff",
    "Plots"
])
```

### Repository environment

If `Project.toml` and `Manifest.toml` are included, the preferred
approach is:

``` julia
using Pkg

Pkg.activate(".")
Pkg.instantiate()
```

This ensures that the repository uses its own reproducible Julia
environment.

------------------------------------------------------------------------

## Example Usage

Run:

``` julia
include("N1Plus_SciML_Demo_v1.jl")
```

Then:

``` julia
result = run_sciml_demo()
```

The example reports:

-   True/reference parameters;
-   Initial parameter estimates;
-   Initial calibration gradient;
-   Estimated parameters;
-   Initial and final loss;
-   Fault-clearing time;
-   Fault-clearing relative squared error.

It also produces a reference-versus-calibrated phase-angle plot.

------------------------------------------------------------------------

## Example Calibration

The demonstration generates a synthetic reference trajectory using

``` julia
p_true = [0.72, 0.35]
```

and starts the calibration from

``` julia
p_initial = [0.40, 0.80]
```

A representative run recovers approximately

``` text
True parameters        = [0.72, 0.35]
Initial parameters     = [0.4, 0.8]
Estimated parameters   ≈ [0.72005, 0.35000]
```

The fault-clearing time is

``` text
τ = 0.5 s
```

### Important interpretation

The synthetic reference trajectory is generated from the same model class used during calibration. Therefore, the very small final error demonstrates **differentiable parameter identification**, not independent physical validation.

Independent validation against ParaEMT/EMT trajectories is a separate research step.

------------------------------------------------------------------------

## Differentiable Calibration

The calibration problem is

$$
p^* = \mathop{\text{argmin}}\limits_{p}\{\mathcal{L}(p)\}.
$$

The demonstration uses a normalized trajectory loss,

$$
\mathcal{L}(p) = \frac{\mathbb{E}\left[\left\|\theta(\tau;p)-\theta^{\mathrm{ref}}(\tau)\right\|_2^2\right]}{\mathbb{E}\left[\left\|\theta^{\mathrm{ref}}(\tau)\right\|_2^2\right]}.
$$

The dynamical model is

$$
\frac{dx}{dt}=f(x,p,t).
$$

Thus,

$$
p\mapsto f(x,p,t) \mapsto x(t;p) \mapsto \mathcal{L}(p) \mapsto \nabla_p\mathcal{L}.
$$

`SciMLSensitivity.jl` provides sensitivity analysis through the differential-equation solve, `ForwardDiff.jl` computes derivatives of the loss, and `Optimization.jl` with `OptimizationOptimJL.LBFGS()` performs the parameter calibration.

------------------------------------------------------------------------

## Power-Grid Model

The example uses a small reduced transmission network.

A line fault is represented by the structured Laplacian perturbation

$$
L_{\mathrm{fault}}=L-\alpha b aa^\top,
$$

where $L$ is the nominal network Laplacian, $b$ is the faulted-line susceptance, $a$ is the incidence vector, and $\alpha$ controls the effective fault strength.

The state is

$$
x=\begin{bmatrix}
\theta \\
\omega
\end{bmatrix},
$$

with phase angles $\theta$ and frequency-related states $\omega$.

The dynamics are

$$
\begin{cases}
\dot{\theta}=\omega, \\
M\dot{\omega}=P-L(t)\theta-D\omega+\sigma q(t).
\end{cases}
$$

During the fault interval,

$$
0\leq t<\tau,
$$

the perturbed topology is used. After clearing,

$$
t\geq\tau,
$$

the nominal topology is restored.

------------------------------------------------------------------------

## Relation to the N1Plus Research

In the full research workflow, high-fidelity EMT simulations provide reference trajectories while the N1Plus model provides a computationally efficient reduced representation.

The corresponding calibration problem is

$$
(\alpha^{opt},\sigma^{opt})=\mathop{\text{argmin}}\limits_{\alpha,\sigma}\frac{\mathbb{E}\left[\left\|\theta^{\mathrm{EMT}}(\tau)-\theta^{\mathrm{N1Plus}}_{(\alpha,\sigma)}(\tau)\right\|^2\right]}{\mathbb{E}\left[\left\|\theta^{\mathrm{EMT}}(\tau)\right\|^2\right]}.
$$

The present repository demonstrates the differentiable computational machinery needed to solve this type of problem.

------------------------------------------------------------------------

## Connection to Scientific Machine Learning

The implementation follows the SciML paradigm of combining mechanistic models with differentiable data-driven computation.

A natural extension is a hybrid model of the form

$$
\dot{x} = f_{\mathrm{physics}}(x,u;\vartheta)+f_{\mathrm{learned}}(x,\mathcal{G},u;\phi)+f_{\mathrm{stochastic}}(x;\psi),
$$

where $\vartheta$ contains physical parameters, $\phi$ parameterizes a learned correction, $\mathcal{G}$ represents network topology, and $\psi$ parameterizes unresolved stochastic effects.

The physical model remains explicit while the learned component represents systematic model discrepancy.

The same differentiable representation can subsequently support:

1.  **Learning to Model**: identify unresolved dynamics from high-fidelity data.
2.  **Learning to Optimize**: embed the surrogate in dynamic-security optimization.
3.  **Learning to Control**: use the surrogate for constrained predictive control.

------------------------------------------------------------------------

## Connection to Dr. Drgoňa's SciML Research

The repository is designed as a power-grid testbed for the broader SciML workflow developed in Dr. Drgoňa's research.

The connection is through differentiable dynamical models used as reusable computational components for:

-   System identification;
-   Constrained optimization;
-   Differentiable predictive control;
-   Physics-informed modeling;
-   Modular Scientific Machine Learning.

NeuroMANCER provides an open-source framework combining differentiable dynamical models with optimization and control. This repository explores the same general computational pattern using a transmission-grid surrogate.

The current implementation focuses on **differentiable model calibration**. The proposed research extends this foundation toward a single hybrid power-grid model that can be reused for learning, optimization, and control.

------------------------------------------------------------------------

## Reproducibility

After cloning the repository:

``` julia
using Pkg

Pkg.activate(".")
Pkg.instantiate()
```

Then:

``` julia
include("N1Plus_SciML_Demo_v1.jl")

result = run_sciml_demo()
```
------------------------------------------------------------------------

## Future Extensions

Natural extensions include:

-   Replacing the synthetic reference with ParaEMT data;
-   Calibrating directly against EMT phase-angle trajectories;
-   Introducing stochastic differential-equation forcing;
-   Learning a graph-based model-discrepancy term;
-   Incorporating multiple fault locations;
-   Calibrating across multiple operating points;
-   Adding dynamic-security constraints;
-   Embedding the surrogate into differentiable optimization;
-   Using the surrogate for model-based predictive control.

These extensions correspond to the broader research direction of developing differentiable hybrid surrogates for learning, optimization and control of power-grid dynamics.

------------------------------------------------------------------------

## Citation

If you use this code in your research, please cite the following publications.

### N1Plus Dynamic Contingency Assessment

``` bibtex
@article{almada2026real,
  title={Real-Time Dynamic N-1 Screening: Identifying High-Risk Lines and Transformers After Common Faults},
  author={Almada, Ayrton and Pagnier, Laurent and Goldshtein, Igal and Kazi, Saif R and others},
  journal={arXiv preprint arXiv:2602.12293},
  year={2026}
}
```

### Scientific Machine Learning and NeuroMANCER

``` bibtex
@article{drgona2023neuromancer,
  title={NeuroMANCER: Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularizations},
  author={Drgona, Jan and Tuor, Aaron and Koch, James and Shapiro, Madelyn and Jacob, Bruno and Vrabie, Draguna},
  year={2023}
}
```

The NeuroMANCER reference is included because the repository follows a related SciML philosophy: differentiable dynamical models are treated as computational components that can be calibrated from data and subsequently embedded in optimization and control workflows.

------------------------------------------------------------------------
