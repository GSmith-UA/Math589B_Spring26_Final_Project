// utils.h — Angle wrapping, distance metrics, math helpers
// Read DESIGN.md §2 before implementing.

#pragma once
#include <cmath>

// Wrap angle to [0, 2π)
double wrapTo2Pi(double theta);

// Signed shortest arc from 'from' to 'to' on S¹
// Returns value in (-π, π]
double shortestArc(double from, double to);

// Wrapped distance between two angles (always positive, in [0, π])
double wrappedThetaDistance(double a, double b);

// Euclidean distance in (θ,φ) space with wrapped θ
double wrappedStateDist(double theta1, double phi1,
                        double theta2, double phi2);
