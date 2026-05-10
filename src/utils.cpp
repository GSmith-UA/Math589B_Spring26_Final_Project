#include "utils.h"
#include <cmath>

static constexpr double TWO_PI = 2.0 * M_PI;

double wrapTo2Pi(double theta) {
    double result = std::fmod(theta, TWO_PI);
    if (result < 0.0) result += TWO_PI;
    return result;
}

double shortestArc(double from, double to) {
    // atan2(sin(Δ), cos(Δ)) maps any Δ into (-π, π]
    return std::atan2(std::sin(to - from), std::cos(to - from));
}

double wrappedThetaDistance(double a, double b) {
    // Reduce |a-b| mod 2π then take the shorter of the two arcs.
    // fmod handles inputs outside [0,2π) correctly.
    double diff = std::fmod(std::abs(a - b), TWO_PI);
    return std::min(diff, TWO_PI - diff);
}

double wrappedStateDist(double theta1, double phi1,
                        double theta2, double phi2) {
    double dtheta = wrappedThetaDistance(theta1, theta2);
    double dphi   = phi1 - phi2;
    return std::sqrt(dtheta * dtheta + dphi * dphi);
}
