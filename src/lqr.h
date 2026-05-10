// lqr.h — LQR solution for small-angle regime
// Read DESIGN.md §6 before implementing.
//
// Linearization: sin(θ)≈θ, cos(θ)≈1
// System: A = [0 1; 1 -alpha], B = [0; 1]
// Cost:   Q = I_2x2, R = 1
// Solves the algebraic Riccati equation for P.

#pragma once
#include <array>

using Matrix2 = std::array<std::array<double, 2>, 2>;
using Vector2 = std::array<double, 2>;

// Solve the 2x2 algebraic Riccati equation.
// Returns P such that: AᵀP + PA - PBR⁻¹BᵀP + Q = 0
// alpha: damping parameter
Matrix2 solveLQRRiccati(double alpha);

// Estimate costate via LQR: lambda = P * [theta, phi]
// Use when |(theta, phi)| is small (see DESIGN.md §6)
Vector2 estimateCostateViaLQR(double theta, double phi, const Matrix2& P);

// Check if a point is in the small-angle regime
// threshold: typically 0.1 (see DESIGN.md §9)
bool isSmallAngle(double theta, double phi, double threshold = 0.1);
