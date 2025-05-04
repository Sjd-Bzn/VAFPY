import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data: Number of walkers and correlation energies
x = np.array([50, 100, 200, 400, 800])

# Local Rebalancing dataset
y1 = np.array([-7.584, -7.587, -7.595, -7.595, -7.596])
y1_errors = np.array([0.002] * len(x))

# Global Rebalancing dataset
y2 = np.array([-7.592, -7.593, -7.599, -7.597, -7.596])
y2_errors = np.array([0.002] * len(x))

# Convergence model: Exponential decay
def conv_model(x, a, b, c):
    return a - b * np.exp(-c * x)

# Fit the model to both datasets
params1, _ = curve_fit(conv_model, x, y1, p0=[-7.6, 1.0, 0.01])
params2, _ = curve_fit(conv_model, x, y2, p0=[-7.6, 1.0, 0.01])
y2_conv_mean = np.mean(y2[:])  # Global convergence level
band_width = 0.002  # Error band ±0.002


# Generate smooth x and fitted y values
x_fit = np.linspace(x.min(), x.max(), 300)
y1_fit = conv_model(x_fit, *params1)
y2_fit = conv_model(x_fit, *params2)

# Professional color palette (edit if desired)
color1 = "#1f77b4"  # Blue (Local)
color2 = "#2ca02c"  # Green (Global)
color3 = '#F25C05' # Orange
color4 = '#6C2BA1' # Purple 

# Plotting
plt.figure(figsize=(6.5, 4.5))

# Convergence zone
plt.fill_between(x_fit, y2_conv_mean - band_width, y2_conv_mean + band_width,
                 color=color3, alpha=0.15, label='Global Convergence Zone')

# Fitted curves
plt.plot(x_fit, y1_fit, color=color4, linewidth=2.2, label='Local Rebalancing')
plt.plot(x_fit, y2_fit, color=color3, linewidth=2.2, label='Global Rebalancing')

# Data points with error bars
plt.errorbar(x, y1, yerr=y1_errors, fmt='o', color=color4, alpha=0.8,
             markersize=6, capsize=3)
plt.errorbar(x, y2, yerr=y2_errors, fmt='s', color=color3, alpha=0.8,
             markersize=6, capsize=3)

# Labels and layout
plt.xlabel("Number of Walkers per Core N$_w$/Core", fontsize=13)
plt.ylabel("Correlation Energy (eV)", fontsize=13)
plt.title("Convergence of Rebalancing", fontsize=14)
plt.xticks(ticks=x, fontsize=11)
y_ticks = np.unique(np.concatenate((y1, y2)))
plt.yticks(ticks=y_ticks, fontsize=11)
#plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=11)
plt.tight_layout()

# Save as high-resolution PDF for publication
plt.savefig("Converged_region_of_the_Rebalancing.pdf", dpi=600, bbox_inches="tight")
plt.show()

