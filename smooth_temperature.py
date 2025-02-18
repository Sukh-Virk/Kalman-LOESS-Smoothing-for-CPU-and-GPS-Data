import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter  # Ensure pykalman is installed

# Generate noisy sine and cosine waves
samples = 25
r = np.linspace(0, 2*np.pi, samples)

obs = pd.DataFrame()
obs['sin'] = np.sin(r) + np.random.normal(0, 0.5, samples)
obs['cos'] = np.cos(r) + np.random.normal(0, 0.5, samples)

# Apply LOWESS smoothing
fil = lowess(obs['sin'], r, frac=0.5, it=0)
filt = lowess(obs['cos'], r, frac=0.5, it=0)

# Define Kalman Filter for a single variable (sin)
matrix = np.array([[1]])  # 1D transition matrix
tc = np.diag([0.03])**2  # Process noise covariance
oc = np.diag([0.02])**2  # Observation noise covariance
initial1 = np.array([obs['sin'].iloc[0]])  # Ensure it's a NumPy array

# Initialize Kalman Filter
kf = KalmanFilter(
    initial_state_mean=initial1,
    observation_covariance=oc,
    transition_covariance=tc,
    transition_matrices=matrix
)

# Apply Kalman smoothing (Ensure correct shape)
ksmooth, _ = kf.smooth(obs[['sin']].values)

# Create figure before plotting
plt.figure(figsize=(8, 5))

# Plot Kalman smoothed results
plt.plot(r, ksmooth[:, 0], label="Kalman Smoothed sin", color="blue", linewidth=2)

# Plot LOWESS smoothed results
plt.plot(fil[:, 0], fil[:, 1], label="LOWESS Smoothed sin", linestyle="dashed", color="green")
plt.plot(filt[:, 0], filt[:, 1], label="LOWESS Smoothed cos", linestyle="dashed", color="purple")

# Scatter noisy observations
plt.scatter(r, obs['sin'], label="Noisy sin", color="red", alpha=0.6)
plt.scatter(r, obs['cos'], label="Noisy cos", color="magenta", alpha=0.6)

# Labels and title
plt.title("Kalman Filter vs LOWESS Smoothing")
plt.xlabel("Time (radians)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Save figure
plt.savefig("no.png")  # ✅ Add valid file extension
plt.show()
