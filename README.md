# Kalman-LOESS-Smoothing-for-CPU-and-GPS-Data

CPU Temperature Noise Reduction - Smoothing CPU temperature data using LOESS and Kalman filtering.

GPS Data Smoothing - Using Kalman filters to correct GPS tracking errors and calculate accurate walking distances.

🔥 Task 1: CPU Temperature Noise Reduction

Objective

The goal is to separate sensor noise from true CPU temperature variations by applying smoothing techniques.

📂 Input File

sysinfo.csv: Contains CPU temperature (°C), CPU usage (%), and system load data.

🛠 Process

Visualize Raw Data

Plot CPU temperature over time.

LOESS Smoothing

Use the lowess function from statsmodels to smooth temperature fluctuations.

Tune the frac parameter to balance noise reduction and signal preservation.

Kalman Filtering

Use pykalman to factor in CPU usage, system load, and fan speed for better smoothing.

Tune transition_covariance and observation_covariance to optimize predictions.

🚀 Running the Script
