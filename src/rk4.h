// rk4.h — Fixed-step RK4 integrator
// Read DESIGN.md §7 before implementing.
//
// IMPORTANT: Fixed step size is deliberate for future GPU compatibility.
// Do NOT add adaptive stepping. See DESIGN.md §7 "CUDA NOTE".

#pragma once
#include "dynamics.h"
#include <vector>
#include <functional>

// Single RK4 step: z_{n+1} = RK4(z_n, h, f)
State rk4Step(const State& z, double h,
              const std::function<State(const State&)>& f);

// Integrate from t=0 to t=T with fixed step h
// Returns trajectory as vector of states at uniform times
// f is the RHS function (use lambda capturing alpha)
// For backward integration: pass f = [&](const State& z){ return backwardDynamics(z,alpha); }
std::vector<State> integrateRK4(const State& z0, double T, double h,
                                 const std::function<State(const State&)>& f);

// Negate a state (used to flip forwardDynamics sign for backward integration)
State negateState(const State& z);
