#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 15:51:09 2026

@author: Paul Junk
"""

#!/usr/bin/env python3
"""
XRD Plotting Template - Multiple datasets with stacking and color gradient
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
})

# =============================================================================
# REFERENCE POSITIONS (ICDD Database) - Comment/uncomment as needed
# =============================================================================

# TiN (cubic)
TIN_111 = 36.59
TIN_200 = 42.50
TIN_220 = 61.68

# TiAl (tetragonal)
TIAL_100 = 31.7
TI7AL3_002 = 38.8
TIAL_002 = 44.6
TIAL_110 = 45.4
TIAL_111 = 50.9

# Ti2AlN(hexagonal, weird one)
TI2ALN_10_13 = 39.9

# Ti3AlN (cubic)
TI2AL1N3_111 = 37.0
TI2AL1N3_200 = 43.0
TI2AL1N3_220 = 62.4

# Si substrate peaks
SI_220 = 32.9
SI_400 = 69.1

# Sapphire (Al2O3) substrate peaks
SAPPHIRE_0006 = 41.7
SAPPHIRE_00012 = 90.7


def load_data(filepath):
    """Load XRD data from CSV file"""
    data = np.genfromtxt(filepath, delimiter=',', skip_header=0)
    data = data[~np.isnan(data).any(axis=1)]
    return data[:, 0], data[:, 1]


def plot_xrd_stacked(
    filepaths,
    labels,
    title,
    output_name,
    substrate='Si',
    log_scale=False,
    stack_offset=5,
    cmap='inferno',
    smooth_sigma=1
):
    """
    Plot multiple XRD datasets stacked with color gradient.
    
    Parameters:
    -----------
    filepaths : list of str
        List of file paths to XRD data files (in order from bottom to top)
    labels : list of str
        List of labels for each dataset
    title : str
        Plot title
    output_name : str
        Output filename for saving
    substrate : str
        'Si' or 'Sapphire' for substrate reference lines
    log_scale : bool
        Use log scale for y-axis (True for 2theta-omega, False for GID)
    stack_offset : float or None
        Manual offset between stacked curves. If None, auto-calculated
    cmap : str
        Matplotlib colormap name for gradient (default: 'inferno')
    smooth_sigma : float
        Gaussian smoothing parameter
    """
    
    if len(filepaths) != len(labels):
        raise ValueError("Number of filepaths must match number of labels")
    
    n_datasets = len(filepaths)
    
    # Load all datasets
    datasets = []
    for filepath in filepaths:
        two_theta, intensity = load_data(filepath)
        smoothed = gaussian_filter1d(intensity, sigma=smooth_sigma)
        datasets.append((two_theta, intensity, smoothed))
    
    # Get colormap
    colors = plt.colormaps.get_cmap(cmap)(np.linspace(0.2, 0.9, n_datasets))
    
    # Calculate y-axis limits and offset
    mask_range = (30, 65)
    all_smoothed = [d[2] for d in datasets]
    
    if stack_offset is None:
        # Auto-calculate offset based on maximum intensity
        max_intensities = []
        for two_theta, _, smoothed in datasets:
            mask = (two_theta >= mask_range[0]) & (two_theta <= mask_range[1])
            max_intensities.append(smoothed[mask].max())
        stack_offset = np.mean(max_intensities) * 0.1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot datasets from bottom to top with stacking
    for i, (two_theta, intensity, smoothed) in enumerate(datasets):
        offset = i * stack_offset
        color = colors[i]
        
        # Plot raw data (faint)
        ax.plot(two_theta, intensity + offset, '-', 
                color=color, alpha=0.25, linewidth=0.5)
        
        # Plot smoothed data
        ax.plot(two_theta, smoothed + offset, '-', 
                color=color, linewidth=1.5, label=labels[i])
    
    # Calculate y-limits
    mask = (datasets[0][0] >= mask_range[0]) & (datasets[0][0] <= mask_range[1])
    
    if log_scale:
        ax.set_yscale('log')
        y_min = all_smoothed[0][mask].min() * 0.5
        y_max = (all_smoothed[-1][mask].max() + (n_datasets - 1) * stack_offset) * 1.5
        ax.set_ylim(max(y_min, 1), y_max)
    else:
        y_max = all_smoothed[-1][mask].max()*0.1 + (n_datasets - 1) * stack_offset*1.5
        ax.set_ylim(0, y_max * 1.1)
    
    # =========================================================================
    # REFERENCE LINES - Modify as needed
    # =========================================================================
    
    # Get y_max for reference line heights
    y_lim = ax.get_ylim()[1]
    
    # --- TiAl reference lines ---
    ax.axvline(TI7AL3_002, color='darkgreen', ls='--', lw=1.5, alpha=0.7)
    ax.text(TI7AL3_002, y_lim*0.1, 'Ti$_{0.7}$Al$_{0.3}$ (002)', ha='center', fontsize=11, 
            color='darkgreen', weight='bold', 
            bbox=dict(boxstyle='round', facecolor='lightgoldenrodyellow', edgecolor='green', alpha=0.9))
    
    # --- Ti2Al1N3 reference lines ---
    ax.axvline(TI2AL1N3_111, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.text(TI2AL1N3_111, y_lim*0.91, '(Ti$_{0.65}$Al$_{0.35}$)N (111)', 
            ha='right', fontsize=11, color='darkred', weight='bold', 
            bbox=dict(boxstyle='round', facecolor='lightgoldenrodyellow', edgecolor='red', alpha=0.9))
    
    ax.axvline(TI2AL1N3_200, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.text(TI2AL1N3_200, y_lim*0.91, '(Ti$_{0.65}$Al$_{0.35}$)N (200)', 
            ha='left', fontsize=11, color='darkred', weight='bold', 
            bbox=dict(boxstyle='round', facecolor='lightgoldenrodyellow', edgecolor='red', alpha=0.9))
    
    ax.axvline(TI2AL1N3_220, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.text(TI2AL1N3_220, y_lim*0.77, '(Ti$_{0.65}$Al$_{0.35}$)N (220)', 
            ha='right', fontsize=11, color='darkred', weight='bold', 
            bbox=dict(boxstyle='round', facecolor='lightgoldenrodyellow', edgecolor='red', alpha=0.9))
    
    # =========================================================================
    # SUBSTRATE MARKERS
    # =========================================================================
    
    if substrate == 'Si':
        ax.axvspan(50, 57.5, alpha=0.15, color='grey', zorder=0)
        ax.text(54, y_lim*0.9, 'Si', fontsize=14, ha='center', 
                color='dimgrey', fontweight='bold')
    
    # =========================================================================
    # LABELS AND FORMATTING
    # =========================================================================
    
    ax.set_xlabel(r'2$\theta$ (degrees)', fontsize=13)
    ax.set_ylabel('Intensity (a.u.) + offset', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(30, 65)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    plt.tick_params(labelleft=False)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved: {output_name}")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    
    # Define your base directory
    base_dir = '/media/sf_Junk.Paul/Data/PJ-XRD-XRR/PJ/GID/'
    
    # Example: Multiple samples stacked
    filepaths = [
        base_dir + 'PJ_TiAl_010q_GID.txt',
        base_dir + 'PJ_TiAl_013q_GID.txt',
        base_dir + 'PJ_TiAl_014q_GID.txt',
        base_dir + 'PJ_TiAl_011q_GID.txt',
        # Add more files here in order from bottom to top
    ]
    
    labels = [
        '0 Pa N$_2$',
        '0.1 Pa N$_2$', 
        '0.2 Pa N$_2$',
        '0.3 Pa N$_2$',
        # Add corresponding labels
    ]
    
    plot_xrd_stacked(
        filepaths=filepaths,
        labels=labels,
        title='GID-XRD: TiAl on Si substrate',
        output_name='/home/bruh/Documents/XRD_stacked_qcm.png',
        substrate='Si',
        log_scale=False,
        stack_offset=None,  # Auto-calculate, or set manually e.g. 500
        cmap='plasma',  # Options: 'inferno', 'viridis', 'plasma', 'magma', 'cividis'
        smooth_sigma=1
    )
    
    print("\n" + "="*60)
    print("Modify filepaths and labels lists to add/remove datasets")
    print("Order matters: first in list = bottom of stack")
    print("="*60)