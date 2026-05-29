import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import exp1
import os
import re

# ==========================================
# User Defined Inputs (Matching ParmParse)
# ==========================================
n_cell = 512
dom_lo = -5.0
dom_hi = 5.0
gauss_cen_x = 0.0
gauss_cen_y = 0.25
variance = 1.0   # From SourceField.H
plotfile = "plt00" + str(n_cell)

# ==========================================
# Directory Setup
# ==========================================
# Extract just the folder name (e.g., 'plt00000' from '../Results/plt00000')
basename = os.path.basename(plotfile.rstrip('/'))

# Extract the continuous string of numbers from the basename
match = re.search(r'\d+', basename)
num_str = match.group(0) if match else "XXXXX"

# Create the output directory
out_dir = f"post{num_str}"
os.makedirs(out_dir, exist_ok=True)
print(f"Output directory established: {os.path.abspath(out_dir)}")

# ==========================================
# 1. Load AMReX Data
# ==========================================
print(f"Loading {plotfile}...")
ds = yt.load(plotfile)

grid = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
phi_num = grid['phi'].squeeze().v 

# 2. Generate Physical Coordinate Grid
dx = (dom_hi - dom_lo) / n_cell
x_coords = np.linspace(dom_lo + dx/2.0, dom_hi - dx/2.0, n_cell)
y_coords = np.linspace(dom_lo + dx/2.0, dom_hi - dx/2.0, n_cell)

X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

# 3. Compute Exact Analytical Solution
R2 = (X - gauss_cen_x)**2 + (Y - gauss_cen_y)**2
R2 = np.where(R2 == 0, 1e-15, R2)

phi_exact = (variance / 4.0) * (np.log(R2 / variance) + exp1(R2 / variance))

# 4. Compute Absolute Error
abs_error = np.abs(phi_num - phi_exact)
max_error = np.max(abs_error)
print(f"Maximum Absolute Error: {max_error:.4e}")

# ==========================================
# 5. Plotting Results (2D)
# ==========================================
extent = [dom_lo, dom_hi, dom_lo, dom_hi]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

im0 = axs[0].imshow(phi_num.T, extent=extent, origin='lower', cmap='viridis')
axs[0].set_title(r"AMReX Numerical $\phi$")
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(phi_exact.T, extent=extent, origin='lower', cmap='viridis')
axs[1].set_title(r"Exact Analytical $\phi$")
fig.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(abs_error.T, extent=extent, origin='lower', cmap='magma')
axs[2].set_title(f"Absolute Error\n(Max: {max_error:.2e})")
fig.colorbar(im2, ax=axs[2])

plt.tight_layout()

# Save into the new directory
out_path_2d = os.path.join(out_dir, "error_maps_2D.png")
plt.savefig(out_path_2d, dpi=300)
print(f"Saved 2D visual error map to: {os.path.abspath(out_path_2d)}")

# ==========================================
# 6. Plotting Results (1D)
# ==========================================
mid_idx = n_cell // 2  
plt.figure(figsize=(8, 5))

plt.plot(x_coords, phi_num[:, mid_idx], 'ro', label='AMReX Numerical', markersize=5)
plt.plot(x_coords, phi_exact[:, mid_idx], 'k-', label='Exact Analytical', linewidth=2)

plt.title("1D Cross-Section at $y = 0$")
plt.xlabel("$x$ coordinate")
plt.ylabel(r"Potential ($\phi$)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save into the new directory
out_path_1d = os.path.join(out_dir, "slice_1D.png")
plt.savefig(out_path_1d, dpi=300)
print(f"Saved 1D slice comparison to: {os.path.abspath(out_path_1d)}")