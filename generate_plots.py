import numpy as np
import matplotlib.pyplot as plt
import os

# Create the plots folder if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# --- Shared Constants & Helper Functions ---
C_m = 1.0
g_Na_max = 120.0
g_K_max  = 36.0
g_L      = 0.3
E_Na = 50.0
E_K  = -77.0
E_L  = -54.387
dt = 0.01
t = np.arange(0, 50 + dt, dt)

def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
def beta_n(V):  return 0.125 * np.exp(-(V + 65) / 80)
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
def beta_m(V):  return 4.0 * np.exp(-(V + 65) / 18)
def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V):  return 1 / (1 + np.exp(-(V + 35) / 10))

# Function to run the simulation with specific stimulus timing
def run_simulation(shock_times):
    V = np.zeros(len(t))
    n = np.zeros(len(t))
    m = np.zeros(len(t))
    h = np.zeros(len(t))
    
    V[0] = -65.0
    n[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))
    m[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
    h[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))
    
    for i in range(1, len(t)):
        I_inj = 0.0
        # Check if current time matches any shock window
        for (start, end, current) in shock_times:
            if start <= t[i-1] <= end:
                I_inj = current
        
        # Update Gates
        n[i] = n[i-1] + (alpha_n(V[i-1]) * (1 - n[i-1]) - beta_n(V[i-1]) * n[i-1]) * dt
        m[i] = m[i-1] + (alpha_m(V[i-1]) * (1 - m[i-1]) - beta_m(V[i-1]) * m[i-1]) * dt
        h[i] = h[i-1] + (alpha_h(V[i-1]) * (1 - h[i-1]) - beta_h(V[i-1]) * h[i-1]) * dt
        
        # Update Voltage
        g_Na = g_Na_max * (m[i-1]**3) * h[i-1]
        g_K  = g_K_max  * (n[i-1]**4)
        I_Na = g_Na * (V[i-1] - E_Na)
        I_K  = g_K  * (V[i-1] - E_K)
        I_L  = g_L  * (V[i-1] - E_L)
        dV = (I_inj - I_Na - I_K - I_L) / C_m
        V[i] = V[i-1] + dV * dt
        
    return V, n, m, h

# --- PLOT 1: The Standard Action Potential ---
# Stimulus at 10ms only
V, n, m, h = run_simulation([(10, 11, 10)])

plt.figure(figsize=(10, 6))
plt.plot(t, V, 'k', linewidth=2)
plt.title('Hodgkin-Huxley Action Potential')
plt.ylabel('Voltage (mV)')
plt.xlabel('Time (ms)')
plt.grid(True)
plt.savefig('plots/action_potential.png') # Save it
print("Saved plots/action_potential.png")

# --- PLOT 2: The Phase Plane (Voltage vs n) ---
# Shows the limit cycle trajectory
plt.figure(figsize=(6, 6))
plt.plot(V, n, 'purple', linewidth=2)
plt.title('Phase Plane Trajectory')
plt.ylabel('Potassium Gate Probability (n)')
plt.xlabel('Membrane Voltage (mV)')
plt.grid(True)
plt.savefig('plots/phase_plane.png') # Save it
print("Saved plots/phase_plane.png")

# --- PLOT 3: The Refractory Failure ---
# Stimulus at 10ms AND 15ms (The failure case)
V_fail, _, _, _ = run_simulation([(10, 11, 10), (15, 16, 10)])

plt.figure(figsize=(10, 6))
plt.plot(t, V_fail, 'k', linewidth=2)
plt.axvspan(10, 11, color='green', alpha=0.2, label='Stimulus 1 (Success)')
plt.axvspan(15, 16, color='red', alpha=0.2, label='Stimulus 2 (Failure)')
plt.title('Demonstration of Absolute Refractory Period')
plt.ylabel('Voltage (mV)')
plt.xlabel('Time (ms)')
plt.legend()
plt.grid(True)
plt.savefig('plots/refractory_failure.png') # Save it
print("Saved plots/refractory_failure.png")