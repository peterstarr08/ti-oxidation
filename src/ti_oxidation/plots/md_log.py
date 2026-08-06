"""
LAMMPS MD log plotter.
Parses merged LAMMPS logs and generates publication-quality plots with ACS styling.

Usage in dispatcher:
  python plot.py md logfile1 logfile2 ... --out-dir plots_md --mark-temp 1000
"""
import argparse
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
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
        return "Default matplotlib (ACS style file not found)"

_STYLE_APPLIED = _load_acs_style()


# --- Parsing ---
def parse_merged_log(file_path):
    """
    Parse merged LAMMPS log file.
    Returns: (steps, poteng, temperature, atoms)
    """
    steps = []
    poteng = []
    temperature = []
    atoms = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        raise ValueError(f"Empty file: {file_path}")

    # --- find atoms ---
    for i, line in enumerate(lines):
        if line.strip().lower() == "atoms":
            try:
                atoms = int(lines[i + 1].strip())
            except:
                raise ValueError(f"Invalid atoms value in {file_path}")
            break

    if atoms is None:
        raise ValueError(f"'atoms' section not found in {file_path}")

    # --- find md_log ---
    md_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "md_log":
            md_idx = i
            break

    if md_idx is None:
        raise ValueError(f"'md_log' section not found in {file_path}")

    # --- header ---
    header_idx = None
    for i in range(md_idx + 1, len(lines)):
        if lines[i].strip():
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"No header found after md_log in {file_path}")

    header = lines[header_idx].strip().split()
    header_lower = [h.lower() for h in header]

    try:
        step_idx = header_lower.index("step")
        poteng_idx = header_lower.index("poteng")
        mytemp_idx = header_lower.index("c_mytemp")
    except ValueError:
        raise ValueError(f"Required columns (Step, PotEng, c_MyTemp) not found in {file_path}")

    # --- data ---
    for line in lines[header_idx + 1:]:
        parts = line.strip().split()

        if len(parts) != len(header):
            continue

        try:
            steps.append(int(float(parts[step_idx])))
            poteng.append(float(parts[poteng_idx]))
            temperature.append(float(parts[mytemp_idx]))
        except ValueError:
            continue

    if not steps:
        raise ValueError(f"No valid data in {file_path}")

    return steps, poteng, temperature, atoms


def trim_data(steps, poteng, poteng_pa, temperature, xmin, xmax):
    """Trim all arrays to [xmin, xmax] range."""
    t_s, t_p, t_pa, t_t = [], [], [], []

    for s, p, pa, t in zip(steps, poteng, poteng_pa, temperature):
        if (xmin is None or s >= xmin) and (xmax is None or s <= xmax):
            t_s.append(s)
            t_p.append(p)
            t_pa.append(pa)
            t_t.append(t)

    return t_s, t_p, t_pa, t_t


# --- Plotting ---
def plot_individual(steps, poteng, poteng_pa, temperature, output_dir, label, ps_per_frame, mark_temp):
    """
    Save three individual subplots (poteng, poteng_pa, temperature) as separate PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    time_ps = [s * ps_per_frame for s in steps]
    
    # Potential Energy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(time_ps, poteng, linewidth=1.5)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Potential Energy (eV)")
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{label}_poteng.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {out_file}")
    
    # Potential Energy per Atom
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(time_ps, poteng_pa, linewidth=1.5)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Potential Energy per Atom (eV/atom)")
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{label}_poteng_pa.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {out_file}")
    
    # Temperature
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(time_ps, temperature, linewidth=1.5)
    ax.axhline(y=mark_temp, color='r', linestyle='--', linewidth=1.5, label=f'{mark_temp} K')
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Temperature (K)")
    ax.legend()
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{label}_temperature.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {out_file}")


def plot_combined_per_file(steps, poteng, poteng_pa, temperature, output_dir, label, ps_per_frame, mark_temp):
    """
    Save combined 3-subplot figure for a single file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    time_ps = [s * ps_per_frame for s in steps]
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    
    axs[0].plot(time_ps, poteng, linewidth=1.5)
    axs[0].set_ylabel("Potential Energy (eV)")
    
    axs[1].plot(time_ps, poteng_pa, linewidth=1.5)
    axs[1].set_ylabel("Potential Energy per Atom (eV/atom)")
    
    axs[2].plot(time_ps, temperature, linewidth=1.5)
    axs[2].axhline(y=mark_temp, color='r', linestyle='--', linewidth=1.5, label=f'{mark_temp} K')
    axs[2].set_xlabel("Time (ps)")
    axs[2].set_ylabel("Temperature (K)")
    axs[2].legend()
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{label}_combined.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {out_file}")


def plot_all_combined(all_data, output_dir, ps_per_frame, mark_temp, subplot_titles=None):
    """
    Save comparison plots for all files (3 subplots, one line per file).
    subplot_titles: optional dict with keys 'poteng', 'poteng_pa', 'temperature'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    
    # Potential Energy
    for label, steps, poteng, _, _ in all_data:
        time_ps = [s * ps_per_frame for s in steps]
        axs[0].plot(time_ps, poteng, label=label, linewidth=1.5)
    
    axs[0].set_ylabel("Potential Energy (eV)")
    if subplot_titles and subplot_titles.get('poteng'):
        axs[0].set_title(subplot_titles['poteng'])
    axs[0].legend()
    
    # Per Atom
    for label, steps, _, poteng_pa, _ in all_data:
        time_ps = [s * ps_per_frame for s in steps]
        axs[1].plot(time_ps, poteng_pa, label=label, linewidth=1.5)
    
    axs[1].set_ylabel("Potential Energy per Atom (eV/atom)")
    if subplot_titles and subplot_titles.get('poteng_pa'):
        axs[1].set_title(subplot_titles['poteng_pa'])
    axs[1].legend()
    
    # Temperature
    for label, steps, _, _, temperature in all_data:
        time_ps = [s * ps_per_frame for s in steps]
        axs[2].plot(time_ps, temperature, label=label, linewidth=1.5)
    
    axs[2].axhline(y=mark_temp, color='r', linestyle='--', linewidth=1.5, label=f'{mark_temp} K')
    axs[2].set_xlabel("Time (ps)")
    axs[2].set_ylabel("Temperature (K)")
    if subplot_titles and subplot_titles.get('temperature'):
        axs[2].set_title(subplot_titles['temperature'])
    axs[2].legend()
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, "all_combined_all_subplots.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"  Saved: {out_file}")


# --- Interactive Mode ---
def interactive_rename(logfiles):
    """
    Ask user to optionally rename files for legend display.
    Returns: dict mapping original basename -> custom label
    """
    labels = {}
    
    print("\n" + "="*60)
    print("INTERACTIVE RENAME MODE")
    print("="*60)
    print(f"Found {len(logfiles)} log file(s):")
    for i, lf in enumerate(logfiles, 1):
        print(f"  {i}. {os.path.basename(lf)}")
    
    user_input = input("\nRename files for legend? (y/n, default=n): ").strip().lower()
    
    if user_input == 'y':
        for lf in logfiles:
            basename = os.path.basename(lf)
            default_name = basename.split('.')[0]  # Remove extension
            prompt = f"  Label for '{basename}' [{default_name}]: "
            custom_name = input(prompt).strip()
            labels[basename] = custom_name if custom_name else default_name
        print("\nLabels set.")
    else:
        # Use default (filename without extension)
        for lf in logfiles:
            basename = os.path.basename(lf)
            labels[basename] = basename.split('.')[0]
        print("\nUsing default labels (filename without extension).")
    
    return labels


def interactive_ps_per_frame():
    """Ask user for picoseconds per frame (timestep)."""
    while True:
        try:
            ps_input = input("\nPicoseconds per timestep (default=1.0 ps/frame): ").strip()
            if not ps_input:
                return 1.0
            ps_per_frame = float(ps_input)
            if ps_per_frame <= 0:
                print("  Error: Must be positive. Try again.")
                continue
            return ps_per_frame
        except ValueError:
            print("  Error: Invalid number. Try again.")


def interactive_subplot_titles():
    """Ask user for optional subplot titles."""
    titles = {}
    
    print("\n" + "="*60)
    print("OPTIONAL SUBPLOT TITLES (leave blank to skip)")
    print("="*60)
    
    poteng_title = input("  Potential Energy subplot title: ").strip()
    if poteng_title:
        titles['poteng'] = poteng_title
    
    poteng_pa_title = input("  Potential Energy per Atom subplot title: ").strip()
    if poteng_pa_title:
        titles['poteng_pa'] = poteng_pa_title
    
    temperature_title = input("  Temperature subplot title: ").strip()
    if temperature_title:
        titles['temperature'] = temperature_title
    
    if titles:
        print(f"\nSubplot titles set: {list(titles.keys())}")
    else:
        print("\nNo subplot titles provided (will use defaults).")
    
    return titles if titles else None


# --- Main Entry Point ---
def run(args):
    """
    Main function called by CLI dispatcher.
    args.files = list of LAMMPS log files
    args.out_dir = output directory
    args.mark_temp = temperature reference line (default 973 K)
    args.xmin, args.xmax = optional trimming range
    args.interactive = bool
    """
    
    logfiles = args.files
    out_dir = args.out_dir
    mark_temp = args.mark_temp
    xmin = getattr(args, 'xmin', None)
    xmax = getattr(args, 'xmax', None)
    interactive_mode = getattr(args, 'interactive', False)
    
    print(f"\n{'='*60}")
    print(f"LAMMPS MD Log Plotter")
    print(f"{'='*60}")
    print(f"Plotting style: {_STYLE_APPLIED}\n")
    
    # Interactive mode: ask for custom labels, ps_per_frame, and subplot titles
    if interactive_mode:
        labels = interactive_rename(logfiles)
        ps_per_frame = interactive_ps_per_frame()
        subplot_titles = interactive_subplot_titles()
    else:
        # Use defaults
        labels = {os.path.basename(lf): os.path.basename(lf).split('.')[0] for lf in logfiles}
        ps_per_frame = 1.0
        subplot_titles = None
    
    print(f"\nPicoseconds per timestep: {ps_per_frame} ps/frame")
    print(f"Temperature reference: {mark_temp} K")
    print(f"Output directory: {out_dir}\n")
    
    # Create output directory with timestamp fallback
    if not out_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        first_name = os.path.basename(logfiles[0]).split('.')[0]
        out_dir = f"plots_{first_name}_{timestamp}"
    
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")
    
    all_data = []
    
    for logfile in logfiles:
        basename = os.path.basename(logfile)
        print(f"Processing: {basename}")
        
        try:
            steps, poteng, temperature, atoms = parse_merged_log(logfile)
        except ValueError as e:
            print(f"  Error: {e}")
            continue
        
        # Compute per-atom before trimming
        poteng_pa = [p / atoms for p in poteng]
        
        # Trim
        steps, poteng, poteng_pa, temperature = trim_data(
            steps, poteng, poteng_pa, temperature,
            xmin, xmax
        )
        
        label = labels[basename]
        all_data.append((label, steps, poteng, poteng_pa, temperature))
        
        # Individual plots (3 separate PNGs)
        plot_individual(steps, poteng, poteng_pa, temperature, out_dir, label, ps_per_frame, mark_temp)
        
        # Combined plot (3 subplots in 1 PNG)
        plot_combined_per_file(steps, poteng, poteng_pa, temperature, out_dir, label, ps_per_frame, mark_temp)
    
    # All files combined (3 subplots, all lines on each)
    if all_data:
        print(f"\nGenerating comparison plots...")
        plot_all_combined(all_data, out_dir, ps_per_frame, mark_temp, subplot_titles)
        print(f"\n{'='*60}")
        print(f"SUCCESS: All plots saved to: {out_dir}")
        print(f"{'='*60}\n")
    else:
        print("ERROR: No valid data found in any files.")
        sys.exit(1)


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser(description="LAMMPS MD log plotter")
    parser.add_argument("files", nargs='+', help="LAMMPS log files")
    parser.add_argument("--out-dir", default="plots_md", help="Output directory")
    parser.add_argument("--mark-temp", type=float, default=973, help="Temperature reference line (default: 973 K)")
    parser.add_argument("--xmin", type=int, default=None, help="Trim: min timestep")
    parser.add_argument("--xmax", type=int, default=None, help="Trim: max timestep")
    parser.add_argument("-i", "--interactive", action='store_true', help="Interactive mode")
    args = parser.parse_args()
    run(args)