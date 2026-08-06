"""
LAMMPS MD log fluctuation analysis.
Performs Gaussian fitting on temperature and potential energy distributions.
Computes statistical moments and thermostat efficiency metrics.

Usage in dispatcher:
  python plot.py fluct logfile --step-min 1000 --step-max 50000
"""
import argparse
import numpy as np
from math import sqrt
import os
import sys
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt


# --- ACS Style Loading ---
def _load_acs_style():
    """Load custom ACS matplotlib style from module directory."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    acs_style_path = os.path.join(module_dir, 'acs_style.mplstyle')
    
    if os.path.exists(acs_style_path):
        plt.style.use(acs_style_path)
        return "Custom ACS (American Chemical Society)"
    else:
        return "Default matplotlib (ACS style file not found)"

_STYLE_APPLIED = _load_acs_style()


# --- Parsing ---
def parse_structured_log(file_path):
    """
    Parse merged LAMMPS log file.
    Returns: (atoms, steps, temps, pes)
    """
    with open(file_path) as f:
        lines = f.readlines()

    atoms = None
    data = []

    # atoms
    for i, line in enumerate(lines):
        if line.strip().lower() == "atoms":
            if i + 1 < len(lines):
                try:
                    atoms = int(lines[i + 1].strip())
                except ValueError:
                    atoms = None
            break

    # md_log
    md_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "md_log":
            md_idx = i
            break

    if md_idx is None:
        raise ValueError("md_log section not found")

    # header
    header_idx = None
    for i in range(md_idx + 1, len(lines)):
        if lines[i].strip():
            header_idx = i
            break

    headers = lines[header_idx].strip().split()
    header_lower = [h.lower() for h in headers]

    step_idx = header_lower.index("step")

    if "poteng" not in header_lower:
        raise ValueError("No PotEng column found")

    pe_idx = header_lower.index("poteng")

    if "c_mytemp" in header_lower:
        temp_idx = header_lower.index("c_mytemp")
    elif "temp" in header_lower:
        temp_idx = header_lower.index("temp")
    else:
        raise ValueError("No temperature column found")

    for line in lines[header_idx + 1:]:
        parts = line.strip().split()

        if len(parts) != len(headers):
            continue

        try:
            data.append({
                "Step": int(float(parts[step_idx])),
                "Temp": float(parts[temp_idx]),
                "PotEng": float(parts[pe_idx]),
            })
        except ValueError:
            continue

    if not data:
        raise ValueError("No valid data in log file")

    return atoms, data


def filter_data(data, step_min, step_max):
    """Filter data by timestep range."""
    filtered = []
    for d in data:
        step = d["Step"]
        
        if step_min is not None and step < step_min:
            continue
        if step_max is not None and step > step_max:
            continue
        
        filtered.append(d)
    
    if not filtered:
        raise ValueError("No data in timestep range")
    
    return filtered


# --- Statistics ---
def compute_stats(values, atoms=None):
    """
    Compute statistics for a data array.
    Returns: dict with mean, std, var, fluctuation, etc.
    """
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    var = float(np.var(values))
    fluct = sigma / mu if mu != 0 else 0.0
    
    stats = {
        'mean': mu,
        'std': sigma,
        'var': var,
        'fluct': fluct,
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'count': len(values),
    }
    
    if atoms is not None:
        stats['sqrt_n_inv'] = 1.0 / sqrt(atoms)
    
    return stats


# --- Plotting ---
def plot_individual_histograms(temps, pes, atoms, output_dir, label):
    """
    Save two individual histogram plots (temperature, potential energy).
    Includes Gaussian fit with mu/sigma in box.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Temperature histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mu_t, sigma_t = norm.fit(temps)
    x_t = np.linspace(np.min(temps), np.max(temps), 200)
    pdf_t = norm.pdf(x_t, mu_t, sigma_t)
    
    ax.hist(temps, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.plot(x_t, pdf_t, 'r-', linewidth=2, label='Gaussian fit')
    
    # Add stats box
    textstr = f'$\\mu$ = {mu_t:.3f} K\n$\\sigma$ = {sigma_t:.3f} K'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{label}_temperature_hist.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {out_file}")
    
    # Potential Energy histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mu_p, sigma_p = norm.fit(pes)
    x_p = np.linspace(np.min(pes), np.max(pes), 200)
    pdf_p = norm.pdf(x_p, mu_p, sigma_p)
    
    ax.hist(pes, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.plot(x_p, pdf_p, 'r-', linewidth=2, label='Gaussian fit')
    
    # Add stats box
    textstr = f'$\\mu$ = {mu_p:.3f} eV\n$\\sigma$ = {sigma_p:.3f} eV'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_xlabel("Potential Energy (eV)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{label}_poteng_hist.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {out_file}")


def plot_combined_histograms(temps, pes, atoms, output_dir, label):
    """
    Save combined 2-subplot histogram figure.
    Includes Gaussian fits with mu/sigma in boxes.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Temperature
    mu_t, sigma_t = norm.fit(temps)
    x_t = np.linspace(np.min(temps), np.max(temps), 200)
    pdf_t = norm.pdf(x_t, mu_t, sigma_t)
    
    axs[0].hist(temps, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
    axs[0].plot(x_t, pdf_t, 'r-', linewidth=2, label='Gaussian fit')
    
    textstr = f'$\\mu$ = {mu_t:.3f} K\n$\\sigma$ = {sigma_t:.3f} K'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axs[0].text(0.98, 0.97, textstr, transform=axs[0].transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    
    axs[0].set_xlabel("Temperature (K)")
    axs[0].set_ylabel("Probability Density")
    axs[0].legend(loc='upper left')
    
    # Potential Energy
    mu_p, sigma_p = norm.fit(pes)
    x_p = np.linspace(np.min(pes), np.max(pes), 200)
    pdf_p = norm.pdf(x_p, mu_p, sigma_p)
    
    axs[1].hist(pes, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
    axs[1].plot(x_p, pdf_p, 'r-', linewidth=2, label='Gaussian fit')
    
    textstr = f'$\\mu$ = {mu_p:.3f} eV\n$\\sigma$ = {sigma_p:.3f} eV'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axs[1].text(0.98, 0.97, textstr, transform=axs[1].transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    
    axs[1].set_xlabel("Potential Energy (eV)")
    axs[1].set_ylabel("Probability Density")
    axs[1].legend(loc='upper left')
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{label}_combined_histograms.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {out_file}")


# --- Logging ---
def write_analysis_log(output_dir, logfile, step_min, step_max, temps, pes, atoms, temps_stats, pes_stats):
    """
    Write detailed analysis report to log file.
    Includes Gaussian fit parameters, statistics, and thermostat metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mu_t, sigma_t = norm.fit(temps)
    mu_p, sigma_p = norm.fit(pes)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_content = f"""================================================================================
LAMMPS MD FLUCTUATION ANALYSIS
================================================================================
Generated: {timestamp}
Source Log: {logfile}

FILTERING
---------
Timestep Range: {step_min if step_min else 'None'} to {step_max if step_max else 'None'}
Data Points: {len(temps)}

TEMPERATURE (K)
---------------
Mean:                    {temps_stats['mean']:.6f}
Std Dev:                 {temps_stats['std']:.6f}
Variance:                {temps_stats['var']:.6f}
Fluctuation (std/mean):  {temps_stats['fluct']:.6e}
Min / Max:               {temps_stats['min']:.6f} / {temps_stats['max']:.6f}

Gaussian Fit:
  Mean (mu) = {mu_t:.6f} K
  Std (sigma) = {sigma_t:.6f} K

POTENTIAL ENERGY (eV)
---------------------
Mean:                    {pes_stats['mean']:.6f}
Std Dev:                 {pes_stats['std']:.6f}
Variance:                {pes_stats['var']:.6f}
Fluctuation (std/mean):  {pes_stats['fluct']:.6e}
Min / Max:               {pes_stats['min']:.6f} / {pes_stats['max']:.6f}

Gaussian Fit:
  Mean (mu) = {mu_p:.6f} eV
  Std (sigma) = {sigma_p:.6f} eV

SYSTEM
------
Number of Atoms:         {atoms if atoms else 'N/A'}
"""
    
    if atoms is not None:
        sqrt_n_inv = temps_stats['sqrt_n_inv']
        log_content += f"1/sqrt(N):               {sqrt_n_inv:.6f}\n"
        log_content += f"\nThermostat Efficiency:\n"
        log_content += f"  std_T / (1/sqrt(N)):    {temps_stats['fluct'] / sqrt_n_inv:.6f}\n"
        log_content += f"  (Expected ~1.0 for ideal NVT ensemble)\n"
    
    log_content += f"\n{'='*80}\n"
    
    log_file = os.path.join(output_dir, "analysis.log")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(log_content)
    
    print(f"  Saved: {log_file}")


# --- Interactive Mode ---
def interactive_step_range():
    """Ask user for optional timestep range filtering."""
    print("\n" + "="*60)
    print("OPTIONAL TIMESTEP FILTERING")
    print("="*60)
    
    step_min_input = input("Minimum timestep (leave blank for none): ").strip()
    step_min = int(step_min_input) if step_min_input else None
    
    step_max_input = input("Maximum timestep (leave blank for none): ").strip()
    step_max = int(step_max_input) if step_max_input else None
    
    if step_min is not None and step_max is not None and step_min > step_max:
        print("Warning: step_min > step_max. Swapping...")
        step_min, step_max = step_max, step_min
    
    if step_min or step_max:
        print(f"\nTimestep range: {step_min if step_min else '∞'} to {step_max if step_max else '∞'}")
    else:
        print("\nUsing all data.")
    
    return step_min, step_max


def interactive_custom_label():
    """Ask user for custom output label."""
    label_input = input("\nCustom output label (leave blank for log filename): ").strip()
    return label_input if label_input else None


# --- Main Entry Point ---
def run(args):
    """
    Main function called by CLI dispatcher.
    args.file = single LAMMPS log file
    args.out_dir = output directory
    args.step_min, args.step_max = optional timestep filtering
    args.interactive = bool
    """
    
    logfile = args.file
    out_dir = args.out_dir
    step_min = getattr(args, 'step_min', None)
    step_max = getattr(args, 'step_max', None)
    interactive_mode = getattr(args, 'interactive', False)
    
    print(f"\n{'='*60}")
    print(f"LAMMPS MD Fluctuation Analysis")
    print(f"{'='*60}")
    print(f"Plotting style: {_STYLE_APPLIED}\n")
    
    # Interactive mode
    if interactive_mode:
        step_min, step_max = interactive_step_range()
        custom_label = interactive_custom_label()
    else:
        custom_label = None
    
    # Create output directory with timestep range in name
    if not out_dir:
        file_base = os.path.basename(logfile).split('.')[0]
        if step_min is not None or step_max is not None:
            step_min_str = str(step_min) if step_min else "start"
            step_max_str = str(step_max) if step_max else "end"
            out_dir = f"fluct_{file_base}_{step_min_str}to{step_max_str}"
        else:
            out_dir = f"fluct_{file_base}_all"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Parse log
    print(f"Parsing: {logfile}")
    try:
        atoms, data = parse_structured_log(logfile)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Filter by timestep
    try:
        filtered = filter_data(data, step_min, step_max)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    temps = np.array([d["Temp"] for d in filtered])
    pes = np.array([d["PotEng"] for d in filtered])
    
    print(f"  Atoms: {atoms if atoms else 'N/A'}")
    print(f"  Data points: {len(temps)}")
    if step_min or step_max:
        print(f"  Timestep range: {step_min if step_min else 'None'} to {step_max if step_max else 'None'}")
    
    # Compute statistics
    temps_stats = compute_stats(temps, atoms)
    pes_stats = compute_stats(pes, atoms)
    
    # Label
    if custom_label:
        label = custom_label
    else:
        label = os.path.basename(logfile).split('.')[0]
    
    print(f"\nGenerating plots and report...")
    
    # Individual plots
    plot_individual_histograms(temps, pes, atoms, out_dir, label)
    
    # Combined plot
    plot_combined_histograms(temps, pes, atoms, out_dir, label)
    
    # Analysis log
    write_analysis_log(out_dir, logfile, step_min, step_max, temps, pes, atoms, temps_stats, pes_stats)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: Analysis saved to: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser(description="LAMMPS MD fluctuation analysis")
    parser.add_argument("file", help="LAMMPS log file")
    parser.add_argument("--out-dir", default="", help="Output directory (auto-generated if empty)")
    parser.add_argument("--step-min", type=int, default=None, help="Minimum timestep")
    parser.add_argument("--step-max", type=int, default=None, help="Maximum timestep")
    parser.add_argument("-i", "--interactive", action='store_true', help="Interactive mode")
    args = parser.parse_args()
    run(args)