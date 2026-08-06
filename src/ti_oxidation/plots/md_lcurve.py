"""
Deep learning learning curve plotter.
Parses lcurve.out (MLIP training logs) and generates publication-quality plots.
Supports batch processing of nested directory structures.

Usage in dispatcher:
  python plot.py lcurve *.lcurve.out
  python plot.py lcurve --dir ./training_runs --recursive
  python plot.py lcurve lcurve.out --logx
"""
import argparse
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


# --- ACS Style Loading ---
def _load_acs_style():
    """Load custom ACS matplotlib style from module directory."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    acs_style_path = os.path.join(module_dir, 'acs_style.mplstyle')
    
    if os.path.exists(acs_style_path):
        plt.style.use(acs_style_path)
        return "Custom ACS (American Chemical Society)"
    else:
        return "Default matplotlib"

_STYLE_APPLIED = _load_acs_style()


# --- Parsing ---
def parse_lcurve(file_path):
    """
    Parse lcurve.out file.
    Returns: dict with 'step', 'rmse_val', 'rmse_trn', 'rmse_e_val', 'rmse_e_trn', 'rmse_f_val', 'rmse_f_trn', 'lr'
    """
    data = {
        'step': [],
        'rmse_val': [],
        'rmse_trn': [],
        'rmse_e_val': [],
        'rmse_e_trn': [],
        'rmse_f_val': [],
        'rmse_f_trn': [],
        'lr': [],
    }
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) < 8:
            continue
        
        try:
            step = int(parts[0])
            rmse_val = float(parts[1])
            rmse_trn = float(parts[2])
            rmse_e_val = float(parts[3])
            rmse_e_trn = float(parts[4])
            rmse_f_val = float(parts[5])
            rmse_f_trn = float(parts[6])
            lr = float(parts[7])
            
            data['step'].append(step)
            data['rmse_val'].append(rmse_val)
            data['rmse_trn'].append(rmse_trn)
            data['rmse_e_val'].append(rmse_e_val)
            data['rmse_e_trn'].append(rmse_e_trn)
            data['rmse_f_val'].append(rmse_f_val)
            data['rmse_f_trn'].append(rmse_f_trn)
            data['lr'].append(lr)
        except (ValueError, IndexError):
            continue
    
    if not data['step']:
        raise ValueError(f"No valid data found in {file_path}")
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def filter_nan_pairs(x, *arrays):
    """
    Filter out indices where any of the arrays have NaN.
    x: x-axis array
    *arrays: multiple y-axis arrays
    Returns: (x_filtered, *arrays_filtered)
    """
    valid_mask = ~np.isnan(x)
    for arr in arrays:
        valid_mask &= ~np.isnan(arr)
    
    x_filtered = x[valid_mask]
    arrays_filtered = [arr[valid_mask] for arr in arrays]
    
    return (x_filtered, *arrays_filtered)


# --- Plotting ---
def plot_individual_and_combined(data, output_dir, label, logx=False):
    """
    Create and save 3 individual plots + 1 combined plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    step = data['step']
    rmse_val = data['rmse_val']
    rmse_trn = data['rmse_trn']
    rmse_e_val = data['rmse_e_val']
    rmse_e_trn = data['rmse_e_trn']
    rmse_f_val = data['rmse_f_val']
    rmse_f_trn = data['rmse_f_trn']
    
    # Filter NaN values
    step, rmse_val, rmse_trn = filter_nan_pairs(step, rmse_val, rmse_trn)
    step, rmse_e_val, rmse_e_trn = filter_nan_pairs(step, rmse_e_val, rmse_e_trn)
    step, rmse_f_val, rmse_f_trn = filter_nan_pairs(step, rmse_f_val, rmse_f_trn)
    
    # --- Individual Plot 1: Training Loss ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(step, rmse_trn, 'b-', linewidth=1.5, label='Train', marker='o', markersize=4, alpha=0.7)
    ax.semilogy(step, rmse_val, 'r-', linewidth=1.5, label='Validation', marker='s', markersize=4, alpha=0.7)
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel("Step")
    ax.set_ylabel("RMSE")
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{label}_training_loss.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {out_file}")
    
    # --- Individual Plot 2: Energy RMSE ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(step, rmse_e_trn, 'b-', linewidth=1.5, label='Train', marker='o', markersize=4, alpha=0.7)
    ax.semilogy(step, rmse_e_val, 'r-', linewidth=1.5, label='Validation', marker='s', markersize=4, alpha=0.7)
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy RMSE (eV)")
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{label}_energy_rmse.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {out_file}")
    
    # --- Individual Plot 3: Force RMSE ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(step, rmse_f_trn, 'b-', linewidth=1.5, label='Train', marker='o', markersize=4, alpha=0.7)
    ax.semilogy(step, rmse_f_val, 'r-', linewidth=1.5, label='Validation', marker='s', markersize=4, alpha=0.7)
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel("Step")
    ax.set_ylabel("Force RMSE (eV/Angstrom)")
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{label}_force_rmse.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {out_file}")
    
    # --- Combined: 3 subplots in 1 row ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training Loss
    axs[0].semilogy(step, rmse_trn, 'b-', linewidth=1.5, label='Train', marker='o', markersize=4, alpha=0.7)
    axs[0].semilogy(step, rmse_val, 'r-', linewidth=1.5, label='Validation', marker='s', markersize=4, alpha=0.7)
    if logx:
        axs[0].set_xscale('log')
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("RMSE")
    axs[0].legend(loc='best')
    axs[0].grid(True, which='both', alpha=0.3)
    
    # Energy RMSE
    axs[1].semilogy(step, rmse_e_trn, 'b-', linewidth=1.5, label='Train', marker='o', markersize=4, alpha=0.7)
    axs[1].semilogy(step, rmse_e_val, 'r-', linewidth=1.5, label='Validation', marker='s', markersize=4, alpha=0.7)
    if logx:
        axs[1].set_xscale('log')
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Energy RMSE (eV)")
    axs[1].legend(loc='best')
    axs[1].grid(True, which='both', alpha=0.3)
    
    # Force RMSE
    axs[2].semilogy(step, rmse_f_trn, 'b-', linewidth=1.5, label='Train', marker='o', markersize=4, alpha=0.7)
    axs[2].semilogy(step, rmse_f_val, 'r-', linewidth=1.5, label='Validation', marker='s', markersize=4, alpha=0.7)
    if logx:
        axs[2].set_xscale('log')
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("Force RMSE (eV/Angstrom)")
    axs[2].legend(loc='best')
    axs[2].grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{label}_combined.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {out_file}")


# --- File Discovery ---
def find_lcurve_files(directory, recursive=True):
    """
    Find all lcurve files in directory (with or without .out extension).
    Matches: lcurve.out, lcurve, lcurve_*, *lcurve* etc.
    Uses rglob for unlimited depth recursion.
    Returns: list of (filepath, relative_parent) tuples
    """
    found = []
    base_path = Path(directory)
    
    if not base_path.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    if recursive:
        # Search for all files with rglob at unlimited depth
        for fpath in base_path.rglob("*"):
            if not fpath.is_file():
                continue
            
            # Match if 'lcurve' is in the filename (case-insensitive)
            if 'lcurve' not in fpath.name.lower():
                continue
            
            # Skip common non-data files
            if fpath.suffix in ['.tmp', '.bak', '.swp']:
                continue
            if fpath.name.startswith('.'):
                continue
            
            # Preserve full parent directory structure
            relative_parent = fpath.parent.relative_to(base_path)
            found.append((fpath, relative_parent))
    else:
        # Only search top level (current directory only)
        for fpath in base_path.glob("*"):
            if not fpath.is_file():
                continue
            
            if 'lcurve' not in fpath.name.lower():
                continue
            
            if fpath.suffix in ['.tmp', '.bak', '.swp']:
                continue
            if fpath.name.startswith('.'):
                continue
            
            relative_parent = fpath.parent.relative_to(base_path)
            found.append((fpath, relative_parent))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_found = []
    for fpath, rel_parent in found:
        key = str(fpath)
        if key not in seen:
            seen.add(key)
            unique_found.append((fpath, rel_parent))
    
    return unique_found


# --- Main Entry Point ---
def run(args):
    """
    Main function called by CLI dispatcher.
    args.files = list of lcurve.out files (if provided)
    args.dir = directory to search (if provided)
    args.recursive = bool (for directory search)
    args.out_dir = output directory
    args.logx = bool (use log scale for x-axis)
    """
    
    out_dir = getattr(args, 'out_dir', None)
    logx = getattr(args, 'logx', False)
    
    print(f"\n{'='*60}")
    print(f"Deep Learning Learning Curve Plotter")
    print(f"{'='*60}")
    print(f"Plotting style: {_STYLE_APPLIED}\n")
    
    # Determine files to process
    lcurve_files = []
    
    if hasattr(args, 'files') and args.files:
        # Direct file arguments
        lcurve_files = [(Path(f), Path(f).parent) for f in args.files]
    elif hasattr(args, 'dir') and args.dir:
        # Directory search (recursive by default)
        directory = getattr(args, 'dir', '.')
        recursive = getattr(args, 'recursive', True)  # Default to True
        print(f"Searching directory: {directory}")
        print(f"Recursive: {recursive}\n")
        try:
            lcurve_files = find_lcurve_files(directory, recursive=recursive)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Error: No files or directory provided")
        sys.exit(1)
    
    if not lcurve_files:
        print("Error: No lcurve files found in directory (or no files matching 'lcurve' pattern)")
        print("  Expected patterns: lcurve, lcurve.out, *lcurve*, lcurve_*")
        sys.exit(1)
    
    print(f"Found {len(lcurve_files)} lcurve file(s)\n")
    
    # Set default output directory
    if not out_dir:
        out_dir = "learning_curves"
    
    # Process each file
    for fpath, parent_dir in lcurve_files:
        print(f"Processing: {fpath}")
        
        try:
            data = parse_lcurve(fpath)
        except ValueError as e:
            print(f"  Error: {e}")
            continue
        
        # Determine output subdirectory (preserve parent structure)
        if parent_dir != Path('.'):
            current_out_dir = os.path.join(out_dir, str(parent_dir))
        else:
            current_out_dir = out_dir
        
        os.makedirs(current_out_dir, exist_ok=True)
        
        # Generate label from filename (remove .out)
        label = fpath.stem.replace('.lcurve', '')
        
        # Plot
        plot_individual_and_combined(data, current_out_dir, label, logx=logx)
        print(f"  Done\n")
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: All plots saved to: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser(description="Deep learning learning curve plotter")
    parser.add_argument("files", nargs='*', help="lcurve.out files to plot")
    parser.add_argument("--dir", default=None, help="Directory to search for lcurve.out files (recursive by default)")
    parser.add_argument("--no-recursive", action='store_true', help="Disable recursive search (only top level)")
    parser.add_argument("--out-dir", default="learning_curves", help="Output directory")
    parser.add_argument("--logx", action='store_true', help="Use log scale for x-axis (step)")
    args = parser.parse_args()
    run(args)