import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norminvgauss

# Define parameters
n_simulations = 1000
T = 30
dt = 1
n_timesteps = int(T/dt)
hurst_parameter = 0.45  # Example value
nig_params = (1, 0, 1, 1)  # Example parameters for the NIG distribution

# Define initial_price and mean_price
initial_price = 10000  # Example value
mean_price = 8000  # Example value

# Initialize the simulation
prices = np.zeros((n_simulations, n_timesteps + 1))
prices[:, 0] = initial_price

# Fractional Ornstein–Uhlenbeck driven by the NIG process
for i in range(1, n_timesteps + 1):
    random_increments = norminvgauss.rvs(*nig_params, size=n_simulations)
    prices[:, i] = prices[:, i - 1] + (mean_price - prices[:, i - 1]) * dt * hurst_parameter + random_increments

# Plot the simulated trajectories
for i in range(n_simulations):
    plt.plot(prices[i, :])

plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Trajectories of Simulated Bitcoin Prices")
plt.show()
