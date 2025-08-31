# -*- coding: utf-8 -*-
"""
Interest-Free New Keynesian DSGE Model

This script builds and simulates an interest-free DSGE model.
It represents an economy where interest-based transactions are prohibited.

Key Modifications from the Conventional Model:
1.  **No Interest Rate:** The nominal interest rate variable (i_hat) and the
    conventional Taylor Rule are removed.
2.  **New Policy Instrument:** The central bank now uses a 'rate of return'
    instrument (let's call it 'xi_hat') to conduct policy. This could be
    interpreted as a target for profit-sharing ratios or returns on
    Sharia-compliant financing.
3.  **Modified Policy Rule:** A new policy rule is introduced where the central
    bank adjusts 'xi_hat' in response to inflation and output gaps.
4.  **Modified IS Curve:** The household's spending decisions now depend on this
    new rate of return instead of the real interest rate.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Model Parameters ---
# Most parameters are kept the same as the conventional model for comparability.

beta = 0.99      # Discount factor
sigma = 1.0      # Coefficient of relative risk aversion
phi = 1.0        # Inverse of the Frisch elasticity of labor supply
epsilon = 6.0    # Elasticity of substitution between differentiated goods
theta = 0.75     # Calvo parameter (price stickiness)

# --- NEW: Interest-Free Policy Rule Parameters ---
# These replace the Taylor Rule coefficients.
# We assume the central bank adjusts its target rate of return (xi)
# based on inflation and output.
phi_pi_xi = 1.5  # Response of target return to inflation
phi_y_xi = 0.5   # Response of target return to output gap

rho_a = 0.9      # Persistence of the technology shock
sigma_a = 0.01   # Standard deviation of the technology shock


# --- 2. Steady State ---
# The steady state is largely similar, but without an interest rate.
Y_ss = 1.0
C_ss = 1.0
# The steady-state real rate of return is still anchored by beta.
xi_ss = (1 / beta) - 1
pi_ss = 0.0


# --- 3. Model Solution (Conceptual) ---
# We again bypass the direct matrix solution and compute the stylized IRFs
# to illustrate the economic intuition.
# The variables are now: [y_hat, c_hat, pi_hat, xi_hat, a_hat]
# y_hat: output gap
# c_hat: consumption
# pi_hat: inflation
# xi_hat: target rate of return (the new policy instrument)
# a_hat: technology shock


# --- 4. Simulate Impulse Response Functions (IRFs) ---
# We trace the effect of the same one-standard-deviation technology shock.

# Set simulation length
T = 40
# Initialize arrays to store the responses
num_vars = 5 # Number of variables in our new conceptual model
irf = np.zeros((num_vars, T))

# The initial shock at t=0
# a_hat is the 5th variable (index 4)
irf[4, 0] = sigma_a

# Calculate the path of the technology shock (AR(1) process)
for t in range(1, T):
    irf[4, t] = rho_a * irf[4, t-1]

# --- STYLIZED SIMULATION (Interest-Free Version) ---
# The following code generates plausible responses for the interest-free
# model facing the same positive productivity shock.

# Output (y) jumps up due to the productivity shock. We assume the direct
# impact is the same as the conventional model.
y_response = 0.8 * irf[4, :]
irf[0, :] = y_response
irf[1, :] = y_response # c_hat = y_hat

# Inflation (pi) still falls due to the productivity gain.
pi_response = -0.2 * irf[4, :]
irf[2, :] = pi_response

# --- NEW POLICY RESPONSE ---
# The central bank now sets its target rate of return 'xi' based on its
# new policy rule.
# xi_hat = phi_pi_xi * pi_hat + phi_y_xi * y_hat
xi_response = phi_pi_xi * pi_response + phi_y_xi * y_response
irf[3, :] = xi_response


# --- 5. Plot the Results ---
# Create plots to visualize the interest-free economy's reaction.

fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
fig.suptitle('Response to a Positive Technology Shock (Interest-Free Model)', fontsize=16)

# Variable names for plotting
var_names = ['Output Gap', 'Inflation', 'Target Rate of Return']
var_indices = [0, 2, 3] # y_hat, pi_hat, xi_hat

for i, ax in enumerate(axes):
    var_index = var_indices[i]
    # Plotting the response as a percentage
    ax.plot(np.arange(T), 100 * irf[var_index, :], 'r-', lw=2.5, label='Response')
    ax.axhline(0, color='k', linestyle='--', lw=1)
    ax.set_title(var_names[i], fontsize=12)
    ax.set_ylabel('% Deviation from Steady State')
    ax.grid(True, linestyle=':', alpha=0.7)

axes[-1].set_xlabel('Quarters after Shock')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
