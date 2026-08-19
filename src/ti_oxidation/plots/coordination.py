"""
Coordination number integration from g(r) data.

Reads radial distribution functions g(r) from .dat (xmgrace) or .csv files.
Computes cumulative coordination number using numerical integration:
  CN(r) = ∫₀^r 4π ρ r'² g(r') dr'

where ρ is the normalization density.

Output: PNG plot, NPZ data file, and log file with metadata.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


# Track which style was applied
_STYLE_APPLIED = None

# Get the path to the ACS style file (in same directory as this module)
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_ACS_STYLE_PATH = os.path.join(_MODULE_DIR, 'acs_style.mplstyle')

# Load custom ACS style
if os.path.exists(_ACS_STYLE_PATH):
    plt.style.use(_ACS_STYLE_PATH)
    _STYLE_APPLIED = "Custom ACS (American Chemical Society)"
else:
    _STYLE_APPLIED = "Default matplotlib (ACS style file not found)"


def report_style():
    """Report which matplotlib style is being used."""
    print(f"Plotting style: {_STYLE_APPLIED}")


def read_gr_data(filepath):
    """
    Read g(r) data from .dat (xmgrace) or .csv file.
    
    Expected formats:
    - .dat (xmgrace): space/tab-separated columns: r g(r)
    - .csv: columns: index,r,g(r)
    
    Returns:
        tuple: (r, gr) as 1D numpy arrays
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.csv':
        # CSV format: index,r,g(r)
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=float)
        if data.ndim == 1:
            # Single row
            data = data.reshape(1, -1)
        r = data[:, 1]
        gr = data[:, 2]
    else:
        # Assume xmgrace .dat format: r g(r)
        data = np.genfromtxt(filepath, dtype=float)
        if data.ndim == 1:
            # Single row
            data = data.reshape(1, -1)
        r = data[:, 0]
        gr = data[:, 1]
    
    return r, gr


def compute_coordination_number(r, gr, density):
    """
    Compute cumulative coordination number via trapezoid integration.
    
    CN(r) = ∫₀^r 4π ρ r'² g(r') dr'
    
    Args:
        r: radial distance array (Angstrom or consistent unit)
        gr: radial distribution function g(r)
        density: normalization density (atoms/Angstrom³ or consistent unit)
    
    Returns:
        numpy array: coordination number as function of r
    """
    # Integrand: 4π ρ r² g(r)
    integrand = 4.0 * np.pi * density * r**2 * gr
    
    # Cumulative trapezoid integration
    cn = np.cumsum(np.gradient(integrand, r)) * np.gradient(r, r)
    
    # Better: use cumulative trapz (scipy not available, use manual)
    cn = np.zeros_like(r)
    for i in range(1, len(r)):
        cn[i] = cn[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * (r[i] - r[i-1])
    
    return cn


def plot_coordination(r, cn, src_file, density, out_path):
    """
    Plot coordination number vs r with ACS styling.
    
    Args:
        r: radial distance array
        cn: coordination number array
        src_file: source filename (for label)
        density: normalization density (for label)
        out_path: output PNG file path
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    ax.plot(r, cn, linewidth=1.5, color='#1f77b4', label='CN(r)')
    ax.set_xlabel('r (Å)')
    ax.set_ylabel('Coordination Number')
    ax.set_title(f'Coordination Number: {Path(src_file).stem}')
    
    # Add metadata text box
    textstr = f'Source: {Path(src_file).name}\nDensity: {density:.6f} atoms/Å³'
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {out_path}")
    plt.close(fig)


def save_data(r, gr, cn, out_npz_path):
    """
    Save r, g(r), and CN(r) to NPZ file.
    
    Args:
        r: radial distance array
        gr: radial distribution function
        cn: coordination number
        out_npz_path: output NPZ file path
    """
    np.savez(out_npz_path, r=r, g_r=gr, coordination_number=cn)
    print(f"  Data saved: {out_npz_path}")


def write_log(log_path, src_files, density, out_dir):
    """
    Write processing log with metadata.
    
    Args:
        log_path: output log file path
        src_files: list of source file paths
        density: normalization density used
        out_dir: output directory
    """
    with open(log_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Coordination Number Integration Log\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Plotting style: {_STYLE_APPLIED}\n\n")
        f.write(f"Normalization Density: {density:.10e} atoms/Å³\n")
        f.write(f"Output Directory: {out_dir}\n\n")
        f.write("Source Files:\n")
        for src in src_files:
            f.write(f"  - {src}\n")
        f.write("\n")
        f.write("Computation:\n")
        f.write("  CN(r) = ∫₀^r 4π ρ r'² g(r') dr'\n")
        f.write("  (integrated via trapezoid rule)\n")
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"  Log saved: {log_path}")


def run(args):
    """
    Main entry point for coordination number analysis.
    
    Args:
        args: argparse Namespace with:
              - files: list of input .dat/.csv files
              - density: normalization density (required)
              - out_dir: output directory (default: adjacent to first input)
    """
    report_style()
    
    if not args.files:
        print("Error: No input files provided", file=sys.stderr)
        sys.exit(1)
    
    density = args.density
    
    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # Default: adjacent to first input file
        out_dir = Path(args.files[0]).parent / "coordination_output"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print(f"Coordination Number Integration")
    print(f"{'=' * 70}")
    print(f"Density: {density:.10e} atoms/Å³")
    print(f"Output: {out_dir}\n")
    
    # Process each file
    for src_file in args.files:
        src_path = Path(src_file)
        if not src_path.exists():
            print(f"Warning: {src_file} not found, skipping", file=sys.stderr)
            continue
        
        print(f"Processing: {src_path.name}")
        
        # Read data
        r, gr = read_gr_data(src_path)
        
        # Compute coordination number
        cn = compute_coordination_number(r, gr, density)
        
        # Generate output filenames (stem from source)
        stem = src_path.stem
        png_out = out_dir / f"{stem}_coordination.png"
        npz_out = out_dir / f"{stem}_coordination.npz"
        
        # Plot
        plot_coordination(r, cn, src_file, density, png_out)
        
        # Save data
        save_data(r, gr, cn, npz_out)
    
    # Write single log file for all processed files
    log_out = out_dir / "coordination_log.txt"
    write_log(log_out, args.files, density, str(out_dir))
    
    print(f"\n{'=' * 70}")
    print(f"Complete. Output directory: {out_dir}")
    print(f"{'=' * 70}\n")
