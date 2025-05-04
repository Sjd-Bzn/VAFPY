"""
Extrapolation of MP2 (VASP) Correlation Energy from AFQMC Data
Cleaned and annotated for scientific clarity.
Generates a plot and saves extrapolated values to a PDF.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Known X and Y values (AFQMC and MP2 Correlation Energies)
x_known = np.array([-11.14648, -23.03148, -28.053 ,-30.360])
y_all = np.array([-10.144, -23.485, -28.338, -31.197, -32.835, -33.713, -36.588])

# Fit a linear model: x = a * y + b using only the known values
a, b = np.polyfit(y_all[:len(x_known)], x_known, 1)

# Extrapolate X values for the remaining Y values
y_extra = y_all[len(x_known):]
x_extra = a * y_extra + b

# Combine known and extrapolated values for visualization
x_full = np.concatenate([x_known, x_extra])
y_full = y_all

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x_known, y_all[:len(x_known)], 'bo-', label='Known Data')
plt.plot(x_extra, y_extra, 'ro--', label='Extrapolated X')
plt.xlabel('AFQMC Correlation Energy')
plt.ylabel('MP2 (VASP) Correlation Energy')
plt.title('Extrapolation of AFQMC from MP2 Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("extrapolation_plot.png")  # Save the figure for later use
plt.show()

# Save extrapolated results to a PDF
with PdfPages('output.pdf') as pdf:
    # Page 1: Add the figure
    fig = plt.figure(figsize=(8, 5))
    img = plt.imread('extrapolation_plot.png')
    plt.imshow(img)
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

    # Page 2: Add text results
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    text = "Extrapolated AFQMC X values:\n\n"
    for i, val in enumerate(x_extra, start=1):
        text += f"{i}. {val:.5f}\n"
    ax.text(0.05, 0.95, text, va='top', fontsize=12, family='monospace')
    pdf.savefig(fig)
    plt.close()

print("Extrapolated values saved to output.pdf")

