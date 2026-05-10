#include "rk4.h"
#include <cmath>

// Single RK4 step: z_{n+1} = RK4(z_n, h, f)
State rk4Step(const State& z, double h,
              const std::function<State(const State&)>& f) {
    State k1 = f(z);
    
    State z_temp;
    for (int i = 0; i < 4; ++i) z_temp[i] = z[i] + (h / 2.0) * k1[i];
    State k2 = f(z_temp);
    
    for (int i = 0; i < 4; ++i) z_temp[i] = z[i] + (h / 2.0) * k2[i];
    State k3 = f(z_temp);
    
    for (int i = 0; i < 4; ++i) z_temp[i] = z[i] + h * k3[i];
    State k4 = f(z_temp);
    
    State z_new;
    for (int i = 0; i < 4; ++i) {
        z_new[i] = z[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    return z_new;
}

// Integrate from t=0 to t=T with fixed step h
// Returns trajectory as vector of states at uniform times
std::vector<State> integrateRK4(const State& z0, double T, double h,
                                 const std::function<State(const State&)>& f) {
    std::vector<State> trajectory;
    trajectory.push_back(z0);
    
    State z_current = z0;
    double t = 0.0;
    int max_steps = 10000000; // safety net
    int step = 0;
    
    while (t < T - 1e-14 && step < max_steps) {
        double h_step = std::min(h, T - t);
        z_current = rk4Step(z_current, h_step, f);
        trajectory.push_back(z_current);
        t += h_step;
        step++;
    }
    return trajectory;
}

// Negate a state (used to flip forwardDynamics sign for backward integration)
State negateState(const State& z) {
    State neg_z;
    for (int i = 0; i < 4; ++i) {
        neg_z[i] = -z[i];
    }
    return neg_z;
}