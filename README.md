# N1Plus + SciML demonstration

This small example connects the N1Plus dynamic-contingency model to the **SciML** ecosystem.

The demonstration:
- represents a transmission fault as a Laplacian perturbation;
- solves the linearized swing equations with `DifferentialEquations.jl`;
- generates synthetic PMU-like observations;
- differentiates a trajectory loss through the ODE solve with `ForwardDiff.jl`;
- estimates inertia and damping scaling parameters; and
- evaluates a simple dynamic contingency indicator.

## Installation

Use a clean Julia environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Then:

```julia
include("N1Plus\\\_SciML.jl")
using .N1PlusSciML

result = run\\\_example()
```

The code intentionally does not use `Optimization.jl` or `OptimizationOptimJL`. The parameter update is a short gradient-descent loop so that the differentiable simulation is easy to see. `DifferentialEquations.jl` provides the ODE problem/solver interface and `ForwardDiff.jl` provides forward-mode automatic differentiation.
