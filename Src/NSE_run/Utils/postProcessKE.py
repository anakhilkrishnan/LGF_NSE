import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

def analyze_ke_data(filepath):
    # Read the tab-separated data
    try:
        df = pd.read_csv(filepath, sep='\t')
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
        sys.exit(1)

    # Clean up column names (strip any accidental whitespace)
    df.columns = df.columns.str.strip()

    # Detect the number of spatial dimensions tracked in the file
    comp_cols = [col for col in df.columns if 'totalKE_dir_comp' in col]
    n_dim = len(comp_cols)
    print(f"Successfully loaded data. Detected {n_dim} spatial dimensions.")

    # Create a multi-panel plot
    # Fig 1: Total KE Error (The drift)
    # Fig 2: Direct vs Evolved KE
    # Fig 3: Component-wise KE
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- Plot 1: The Algebraic Energy Defect ---
    ax = axes[0]
    ax.plot(df['Time'], df['totalKE_err'], color='red', linewidth=1.5, label='Cumulative Drift')
    ax.set_ylabel('Total KE Error')
    ax.set_title('Kinetic Energy Preserving (KEP) Diagnostic')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # --- Plot 2: Total Kinetic Energy (Direct vs Evolved) ---
    ax = axes[1]
    ax.plot(df['Time'], df['totalKE_direct'], label='Direct (u²/2)', linestyle='-', color='black')
    ax.plot(df['Time'], df['totalKE_evolved'], label='Evolved (Scalar Advection)', linestyle='--', color='dodgerblue')
    ax.set_ylabel('Total Kinetic Energy')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # --- Plot 3: Component-wise Energy Breakdown ---
    ax = axes[2]
    colors = ['orange', 'green', 'purple']
    for idim in range(n_dim):
        dir_col = f'totalKE_dir_comp{idim}'
        evol_col = f'totalKE_evol_comp{idim}'
        
        if dir_col in df.columns:
            ax.plot(df['Time'], df[dir_col], color=colors[idim % 3], 
                    label=f'Comp {idim} Direct')
        if evol_col in df.columns:
            ax.plot(df['Time'], df[evol_col], color=colors[idim % 3], linestyle='--', alpha=0.7, 
                    label=f'Comp {idim} Evolved')

    ax.set_xlabel('Simulation Time')
    ax.set_ylabel('Component KE')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Place legend outside the plot for the component breakdown
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig('kinetic_energy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Kinetic Energy Diagnostics")
    parser.add_argument("filename", nargs='?', default="kinetic_energy.dat", 
                        help="Path to the kinetic energy data file (default: kinetic_energy.dat)")
    
    args = parser.parse_args()
    analyze_ke_data(args.filename)