"""
Hodgkin-Huxley Neuron Simulation
--------------------------------
A from-scratch implementation of the 1952 Hodgkin-Huxley model using Euler's Method.

This script simulates the action potential of a giant squid axon. It demonstrates:
1. The interaction of voltage-gated Sodium (Na) and Potassium (K) channels.
2. The role of Gating Variables (m, n, h).
3. The 'Refractory Period' phenomenon (why a neuron cannot fire twice instantly).

Author: K Samar
Based on: Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current
          and its application to conduction and excitation in nerve.
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Hardware Constants (Bio-Physical Properties)
# ==========================================

# Membrane Capacitance (C_m)
# Represents the lipid bilayer which acts as a dielectric.
C_m = 1.0     # uF/cm^2

# Maximum Conductances (g_max)
# Represents the total number of channels available per area if 100% were open.
g_Na_max = 120.0  # mS/cm^2 (Sodium)
g_K_max  = 36.0   # mS/cm^2 (Potassium)
g_L      = 0.3    # mS/cm^2 (Leak - mostly Chloride ions)

# Nernst Potentials (Equilibrium Voltages)
# The voltage at which the chemical force equals the electrical force.
E_Na = 50.0       # mV (Sodium pushes IN to make cell positive)
E_K  = -77.0      # mV (Potassium pushes OUT to make cell negative)
E_L  = -54.387    # mV (Leak maintains resting potential)

# ==========================================
# 2. Simulation Setup (Time & Arrays)
# ==========================================

time_duration = 50.0   # Total simulation time (ms)
dt = 0.01              # Time step (ms)
t = np.arange(0, time_duration + dt, dt)

# Initialize State Variable Arrays
V = np.zeros(len(t))   # Membrane Voltage
n = np.zeros(len(t))   # Potassium Activation Gate Probability
m = np.zeros(len(t))   # Sodium Activation Gate Probability
h = np.zeros(len(t))   # Sodium Inactivation Gate Probability

# ==========================================
# 3. Helper Functions (Rate Constants)
# ==========================================
# These empirical formulas were determined by curve-fitting experimental data
# from the original 1952 paper. They describe how fast gates open/close
# as a function of Voltage.

def alpha_n(V):
    """Opening rate for K+ channels."""
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

def beta_n(V):
    """Closing rate for K+ channels."""
    return 0.125 * np.exp(-(V + 65) / 80)

def alpha_m(V):
    """Opening rate for Na+ channels."""
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

def beta_m(V):
    """Closing rate for Na+ channels."""
    return 4.0 * np.exp(-(V + 65) / 18)

def alpha_h(V):
    """Opening rate for Na+ Inactivation gate (Recovery)."""
    return 0.07 * np.exp(-(V + 65) / 20)

def beta_h(V):
    """Closing rate for Na+ Inactivation gate (Inactivation)."""
    return 1 / (1 + np.exp(-(V + 35) / 10))

# ==========================================
# 4. Initial Conditions (Steady State)
# ==========================================
# We initialize the system at Resting Potential (-65mV).
# The gates are initialized to their steady-state probabilities at this voltage.

V[0] = -65.0
n[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))
m[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
h[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))

print("System Initialized. Starting Simulation...")

# ==========================================
# 5. Main Simulation Loop (Euler Integration)
# ==========================================

for i in range(1, len(t)): # Start at 1 to look back at [i-1]

    # --- A. Define Stimulus (Input Current) ---
    # Experiment: Double Pulse Protocol
    # Pulse 1: 10ms (Standard Spike)
    # Pulse 2: 15ms (Demonstrating Absolute Refractory Period failure)

    if 10.0 <= t[i-1] <= 11.0:
        I_inj = 10.0   # uA/cm^2
    elif 15.0 <= t[i-1] <= 16.0:
        I_inj = 10.0   # uA/cm^2 (Will fail to spike due to refractory period)
    else:
        I_inj = 0.0

    # --- B. Update Gating Variables (The Kinetics) ---
    # We use the differential equation: dn/dt = alpha*(1-n) - beta*n

    # Potassium (n)
    alpha_n_val = alpha_n(V[i-1])
    beta_n_val  = beta_n(V[i-1])
    dn = alpha_n_val * (1 - n[i-1]) - beta_n_val * n[i-1]
    n[i] = n[i-1] + dn * dt  # Euler's method for numerical integration

    # Sodium Activation (m)
    alpha_m_val = alpha_m(V[i-1])
    beta_m_val  = beta_m(V[i-1])
    dm = alpha_m_val * (1 - m[i-1]) - beta_m_val * m[i-1]
    m[i] = m[i-1] + dm * dt

    # Sodium Inactivation (h)
    alpha_h_val = alpha_h(V[i-1])
    beta_h_val  = beta_h(V[i-1])
    dh = alpha_h_val * (1 - h[i-1]) - beta_h_val * h[i-1]
    h[i] = h[i-1] + dh * dt

    # --- C. Update Membrane Voltage (The Output) ---

    # 1. Calculate Conductances
    # Note: Na channel has 3 'm' subunits and 1 'h' subunit (m^3 * h)
    # Note: K channel has 4 'n' subunits (n^4)
    g_Na = g_Na_max * (m[i-1]**3) * h[i-1]
    g_K  = g_K_max  * (n[i-1]**4)
    g_L  = 0.3      # Constant leak

    # 2. Calculate Ionic Currents (Ohm's Law: I = g * (V - E))
    I_Na = g_Na * (V[i-1] - E_Na)
    I_K  = g_K  * (V[i-1] - E_K)
    I_L  = g_L  * (V[i-1] - E_L)

    # 3. Membrane Equation (Kirchhoff's Current Law)
    # I_inj = I_cap + I_ionic  ->  dV/dt = (I_inj - I_ionic) / C_m
    dV = (I_inj - I_Na - I_K - I_L) / C_m

    # 4. Step Voltage Forward
    V[i] = V[i-1] + dV * dt

print("Simulation Complete.")

# ==========================================
# 6. Visualization
# ==========================================

plt.figure(figsize=(12, 8))

# Plot 1: Membrane Voltage
plt.subplot(2, 1, 1)
plt.plot(t, V, color='black', linewidth=2)
plt.title('Hodgkin-Huxley Action Potential Simulation')
plt.ylabel('Membrane Voltage (mV)')
plt.xlabel('Time (ms)')
plt.grid(True, alpha=0.5)

# Visualizing the Stimulus Windows
plt.axvspan(10, 11, color='red', alpha=0.2, label='Stimulus 1 (Success)')
plt.axvspan(15, 16, color='gray', alpha=0.2, label='Stimulus 2 (Refractory Failure)')
plt.legend(loc='upper right')

# Plot 2: Gating Variables
plt.subplot(2, 1, 2)

plt.plot(t, m, 'r', label='m (Na Activation)', linewidth=2)
plt.plot(t, h, 'g', label='h (Na Inactivation)', linewidth=2)
plt.plot(t, n, 'b', label='n (K Activation)', linewidth=2)
plt.title('Gating Variable Dynamics')
plt.ylabel('Probability (0.0 - 1.0)')
plt.xlabel('Time (ms)')
plt.legend(loc='center right')
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()