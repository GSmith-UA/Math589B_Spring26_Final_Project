// dynamics.h — Hamiltonian ODE system
// Read DESIGN.md §1 before implementing.
//
// State vector: z = [theta, phi, lambda1, lambda2]
// Parameter:    alpha (damping, default 0.1)

#pragma once
#include <array>

using State = std::array<double, 4>;
using Matrix4 = std::array<std::array<double, 4>, 4>;

// Hamilton's equations: dz/dt = forwardDynamics(z)
// See DESIGN.md §1 for the full expressions.
State forwardDynamics(const State& z, double alpha);

// For backward integration: dz/dt = -forwardDynamics(z)
State backwardDynamics(const State& z, double alpha);

// Effective Hamiltonian — should be ~0 along optimal trajectories
// Use for conservation monitoring (see DESIGN.md §7)
double effectiveHamiltonian(const State& z, double alpha);

// Analytical Jacobian of forwardDynamics at z
// Used for re-linearization in the continuation algorithm (DESIGN.md §4)
// and for eigendecomposition to seed the stable manifold (DESIGN.md §3)
Matrix4 computeJacobian(const State& z, double alpha);
