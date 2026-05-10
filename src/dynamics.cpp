#include "dynamics.h"
#include <cmath>

State forwardDynamics(const State& z, double alpha) {
    double th = z[0], phi = z[1], l1 = z[2], l2 = z[3];
    
    State dz;
    dz[0] = phi;
    dz[1] = sin(th) - alpha * phi - l2 * cos(th) * cos(th);
    dz[2] = -(sin(th) + l2 * cos(th) + l2 * l2 * sin(th) * cos(th));
    dz[3] = -(phi + l1 - alpha * l2);
    
    return dz;
}

State backwardDynamics(const State& z, double alpha) {
    State fwd = forwardDynamics(z, alpha);
    for (double& val : fwd) val = -val;
    return fwd;
}

double effectiveHamiltonian(const State& z, double alpha) {
    double th = z[0], phi = z[1], l1 = z[2], l2 = z[3];
    
    return (1 - cos(th)) + 0.5 * phi * phi - 0.5 * l2 * l2 * cos(th) * cos(th) + 
           l1 * phi + l2 * (sin(th) - alpha * phi);
}

Matrix4 computeJacobian(const State& z, double alpha) {
    double th = z[0], phi = z[1], l1 = z[2], l2 = z[3];
    double s = sin(th), c = cos(th);
    
    Matrix4 J = {{{0}}};
    
    // Row 0: d(theta_dot)/dz
    J[0][1] = 1.0;
    
    // Row 1: d(phi_dot)/dz
    J[1][0] = c * (1.0 + 2.0 * l2 * s);
    J[1][1] = -alpha;
    J[1][3] = -c * c;
    
    // Row 2: d(lambda1_dot)/dz
    J[2][0] = -c + l2 * s - l2 * l2 * (c*c - s*s);
    J[2][3] = -c - 2.0 * l2 * s * c;
    
    // Row 3: d(lambda2_dot)/dz
    J[3][1] = -1.0;
    J[3][2] = -1.0;
    J[3][3] = alpha;
    
    return J;
}