"""
Standalone module to plot g(r) vs r from CSV files.
Uses custom ACS (American Chemical Society) matplotlib stylesheet.
Assumes r is in Angstrom.
"""

import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


def read_csv_file(filename):
    """
    Read CSV file and extract r and g(r) columns.
    Returns tuple (r_values, g_values, filename_base) or (None, None, None) on error.
    """
    r_values = []
    g_values = []

    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader, None)

            r_idx = None
            g_idx = None

            if header:
                for i, col in enumerate(header):
                    col_clean = col.strip().lower()
                    if col_clean == 'r':
                        r_idx = i
                    elif col_clean == 'g(r)' or col_clean == 'g(r':
                        g_idx = i
                if r_idx is None or g_idx is None:
                    r_idx = 1
                    g_idx = 2
            else:
                r_idx = 1
                g_idx = 2

            file.seek(0)
            reader = csv.reader(file)
            if header:
                next(reader, None)

            for row in reader:
                try:
                    if len(row) > max(r_idx, g_idx):
                        r_val = float(row[r_idx].strip())
                        g_val = float(row[g_idx].strip())
                        r_values.append(r_val)
                        g_values.append(g_val)
                except (ValueError, IndexError):
                    continue

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return None, None, None

    base_name = os.path.splitext(os.path.basename(filename))[0]
    return r_values, g_values, base_name


def prompt_for_renames(file_data_list):
    """
    Ask user to rename output/plot labels for each file.
    Returns list of (filename, r_values, g_values, label).
    """
    answer = input("\nRename files for output/plot labels? (y/n): ").strip().lower()
    if answer != 'y':
        return file_data_list

    renamed_list = []
    for filename, r_values, g_values, base_name in file_data_list:
        new_name = input(f"  New name for '{filename}' (blank = keep as is): ").strip()
        if new_name:
            renamed_list.append((filename, r_values, g_values, new_name))
        else:
            renamed_list.append((filename, r_values, g_values, base_name))
    return renamed_list


def prompt_for_comparison_label():
    """
    Ask user for a custom label for the comparison plot.
    Returns the label or default string.
    """
    label = input("\nLabel for comparison plot (blank = 'Comparison of All Files'): ").strip()
    return label if label else "Comparison of All Files"


def setup_detailed_axes(ax, r_values, precision=2):
    """
    Set up detailed axis formatting with major/minor ticks and grid.
    """
    major_ticks = []
    if r_values:
        r_min = min(r_values)
        r_max = max(r_values)
        r_range = r_max - r_min

        if r_range > 0:
            rough_step = r_range / 12
            nice_steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
            step = min([s for s in nice_steps if s >= rough_step], 
                      key=lambda x: abs(x - rough_step))

            major_ticks = []
            current = r_min
            while current <= r_max + step / 2:
                major_ticks.append(round(current, 6))
                current += step

            if len(major_ticks) > 20:
                step *= 2
                major_ticks = []
                current = r_min
                while current <= r_max + step / 2:
                    major_ticks.append(round(current, 6))
                    current += step

        ax.set_xticks(major_ticks)

        if len(major_ticks) > 1:
            minor_step = step / 5 if step > 0.01 else step / 2
            minor_ticks = []
            current = r_min
            while current <= r_max + minor_step / 2:
                if round(current, 6) not in major_ticks:
                    minor_ticks.append(round(current, 6))
                current += minor_step
            ax.set_xticks(minor_ticks, minor=True)

    format_str = f'%.{precision}f'
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(format_str))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(format_str))

    if len(major_ticks) > 10:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)

    ax.set_xlabel(r'$r$ ($\mathrm{\AA}$)')
    ax.set_ylabel(r'$g(r)$')


def plot_individual_files(file_data_list, precision=2, r_cutoff=None):
    """
    Plot each file individually as a continuous line and save as HQ PNG.
    Returns dict {label: fig}.
    
    Args:
        file_data_list: List of (filename, r_values, g_values, label) tuples
        precision: Decimal precision for axis labels
        r_cutoff: Optional cutoff in Angstrom; data beyond this is excluded
    """
    figures = {}
    for filename, r_values, g_values, label in file_data_list:
        if not r_values or not g_values:
            continue

        # Apply r_cutoff if specified
        if r_cutoff is not None:
            r_vals_plot = [r for r in r_values if r <= r_cutoff]
            indices = [i for i, r in enumerate(r_values) if r <= r_cutoff]
            g_vals_plot = [g for i, g in enumerate(g_values) if i in indices]
        else:
            r_vals_plot = r_values
            g_vals_plot = g_values

        if not r_vals_plot or not g_vals_plot:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(r_vals_plot, g_vals_plot, linestyle='-', linewidth=1.5)

        setup_detailed_axes(ax, r_vals_plot, precision)
        ax.set_title(f'$g(r)$ vs $r$ - {label}')

        fig.tight_layout()

        # Sanitize filename
        safe_label = label.replace(' ', '_').replace('/', '_')
        output_filename = f"gr_{safe_label}.png"
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_filename}")

        figures[label] = fig

    return figures


def plot_all_together(file_data_list, precision=2, title_label="Comparison of All Files", r_cutoff=None, gr_cutoff=None):
    """
    Plot all files together in a single figure for comparison.
    Returns the combined figure.
    
    Args:
        file_data_list: List of (filename, r_values, g_values, label) tuples
        precision: Decimal precision for axis labels
        title_label: Title for the comparison plot
        r_cutoff: Optional cutoff in Angstrom; data beyond this is excluded (x-axis)
        gr_cutoff: Optional g(r) y-axis cutoff; y-axis view is limited to this value
    """
    if not file_data_list:
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    all_r = []
    for filename, r_values, g_values, label in file_data_list:
        if r_values:
            if r_cutoff is not None:
                all_r.extend([r for r in r_values if r <= r_cutoff])
            else:
                all_r.extend(r_values)

    for filename, r_values, g_values, label in file_data_list:
        if not r_values or not g_values:
            continue

        # Apply r_cutoff if specified
        if r_cutoff is not None:
            indices = [i for i, r in enumerate(r_values) if r <= r_cutoff]
            r_vals_plot = [r for i, r in enumerate(r_values) if i in indices]
            g_vals_plot = [g for i, g in enumerate(g_values) if i in indices]
        else:
            r_vals_plot = r_values
            g_vals_plot = g_values

        if not r_vals_plot or not g_vals_plot:
            continue

        ax.plot(r_vals_plot, g_vals_plot,
                linestyle='-',
                linewidth=1.2,
                label=label,
                alpha=0.9)

    setup_detailed_axes(ax, all_r, precision)
    ax.set_title(f'$g(r)$ vs $r$ - {title_label}')

    ax.legend(loc='best', fontsize=9)

    # Apply g(r) y-axis cutoff if specified (COMPARISON PLOT ONLY)
    if gr_cutoff is not None:
        ax.set_ylim(bottom=0, top=gr_cutoff)

    fig.tight_layout()

    # Sanitize filename
    safe_title = title_label.replace(' ', '_').replace('/', '_')
    output_filename = f"gr_comparison_{safe_title}.png"
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_filename}")

    return fig


def interactive_menu(file_data_list, individual_figures, combined_figure, precision=2, r_cutoff=None, gr_cutoff=None, comparison_label="Comparison of All Files"):
    """
    Interactive menu for inspecting multiple plots at once.
    List all plots (1 to n for individual, n+1 for combined).
    User enters space-separated integers to open selected plots.
    Plots are smaller in interactive mode for easy comparison.
    
    Args:
        gr_cutoff: Optional g(r) y-axis cutoff for comparison plot only
    """
    # Close all existing figures
    plt.close('all')
    
    labels = list(individual_figures.keys())
    n_individual = len(labels)
    
    print("\n--- Available plots ---")
    for idx, label in enumerate(labels, start=1):
        print(f"  [{idx}] {label}")
    if combined_figure is not None:
        print(f"  [{n_individual + 1}] {comparison_label}")
    
    print("\nEnter space-separated plot numbers to display (e.g., '1 2 3' or '1 4')")
    selection = input("Selection (blank = exit): ").strip()
    
    if not selection:
        print("Exiting.")
        plt.close('all')
        return
    
    try:
        indices = [int(x.strip()) for x in selection.split()]
    except ValueError:
        print("Invalid input. Exiting.")
        plt.close('all')
        return
    
    # Validate and create figures for selected plots
    created_count = 0
    errors = []
    
    for idx in indices:
        try:
            if 1 <= idx <= n_individual:
                # Individual plot (smaller for interactive mode)
                label = labels[idx - 1]
                
                # Find the data for this label
                for filename, r_values, g_values, lbl in file_data_list:
                    if lbl == label:
                        # Apply r_cutoff
                        if r_cutoff is not None:
                            indices_cut = [i for i, r in enumerate(r_values) if r <= r_cutoff]
                            r_vals_plot = [r for i, r in enumerate(r_values) if i in indices_cut]
                            g_vals_plot = [g for i, g in enumerate(g_values) if i in indices_cut]
                        else:
                            r_vals_plot = r_values
                            g_vals_plot = g_values
                        
                        # Create figure (smaller size for interactive mode)
                        fig, ax = plt.subplots(figsize=(6, 4.5))
                        ax.plot(r_vals_plot, g_vals_plot, linestyle='-', linewidth=1.5)
                        setup_detailed_axes(ax, r_vals_plot, precision)
                        ax.set_title(f'$g(r)$ vs $r$ - {lbl}')
                        fig.tight_layout()
                        created_count += 1
                        break
            
            elif idx == n_individual + 1 and combined_figure is not None:
                # Combined plot (smaller for interactive mode)
                fig, ax = plt.subplots(figsize=(8, 5.5))
                
                all_r = []
                for filename, r_values, g_values, label in file_data_list:
                    if r_values:
                        if r_cutoff is not None:
                            all_r.extend([r for r in r_values if r <= r_cutoff])
                        else:
                            all_r.extend(r_values)
                
                for filename, r_values, g_values, label in file_data_list:
                    if not r_values or not g_values:
                        continue
                    
                    # Apply r_cutoff
                    if r_cutoff is not None:
                        indices_cut = [i for i, r in enumerate(r_values) if r <= r_cutoff]
                        r_vals_plot = [r for i, r in enumerate(r_values) if i in indices_cut]
                        g_vals_plot = [g for i, g in enumerate(g_values) if i in indices_cut]
                    else:
                        r_vals_plot = r_values
                        g_vals_plot = g_values
                    
                    if not r_vals_plot or not g_vals_plot:
                        continue
                    
                    ax.plot(r_vals_plot, g_vals_plot,
                            linestyle='-',
                            linewidth=1.2,
                            label=label,
                            alpha=0.9)
                
                setup_detailed_axes(ax, all_r, precision)
                ax.set_title(f'$g(r)$ vs $r$ - {comparison_label}')
                ax.legend(loc='best', fontsize=8)
                
                # Apply g(r) y-axis cutoff to interactive comparison plot
                if gr_cutoff is not None:
                    ax.set_ylim(bottom=0, top=gr_cutoff)
                
                fig.tight_layout()
                created_count += 1
            else:
                errors.append(f"Plot {idx} out of range")
        
        except (IndexError, KeyError) as e:
            errors.append(f"Plot {idx}: {str(e)}")
    
    if created_count > 0:
        print(f"\nOpening {created_count} plot window(s)...")
        plt.show()
    else:
        print("No valid plots selected.")
    
    if errors:
        for err in errors:
            print(f"Warning: {err}")
    
    plt.close('all')


def prompt_for_r_cutoff():
    """
    Ask user for optional r cutoff value in Angstrom.
    Returns float or None.
    """
    cutoff_str = input("\nOptional r cutoff in Angstrom (blank = no cutoff): ").strip()
    if not cutoff_str:
        return None
    try:
        cutoff = float(cutoff_str)
        if cutoff <= 0:
            print("Warning: cutoff must be positive, ignoring.")
            return None
        return cutoff
    except ValueError:
        print("Warning: invalid cutoff value, ignoring.")
        return None


def prompt_for_gr_cutoff():
    """
    Ask user for optional g(r) y-axis cutoff for comparison plot.
    Returns float or None.
    """
    cutoff_str = input("\nOptional g(r) y-axis cutoff for comparison plot (blank = no cutoff): ").strip()
    if not cutoff_str:
        return None
    try:
        cutoff = float(cutoff_str)
        return cutoff
    except ValueError:
        print("Warning: invalid g(r) cutoff value, ignoring.")
        return None


def run(args):
    """
    Main entry point for g(r) plotting. Called by the CLI wrapper.
    
    Expected args attributes:
      - files: list of CSV filenames
      - precision: int, decimal precision for axis labels
      - no_individual: bool, skip individual plots
      - no_comparison: bool, skip comparison plot
      - interactive: bool, show interactive menu
      - r_cutoff: float or None, cutoff in Angstrom (x-axis)
      - gr_cutoff: float or None, g(r) y-axis cutoff for comparison plot only
    """
    report_style()
    
    file_data_list = []
    for filename in args.files:
        print(f"Reading: {filename}")
        r_values, g_values, base_name = read_csv_file(filename)
        if r_values is not None:
            file_data_list.append((filename, r_values, g_values, base_name))
            print(f"  - Loaded {len(r_values)} data points")
        else:
            print(f"  - Failed to read {filename}")

    if not file_data_list:
        print("No valid data files found.")
        return

    file_data_list = prompt_for_renames(file_data_list)
    r_cutoff = prompt_for_r_cutoff()
    gr_cutoff = prompt_for_gr_cutoff()

    print(f"\nProcessing {len(file_data_list)} file(s) with precision: {args.precision} decimal places...")
    if r_cutoff is not None:
        print(f"r cutoff (x-axis): {r_cutoff} Å")
    if gr_cutoff is not None:
        print(f"g(r) cutoff (y-axis, comparison plot only): {gr_cutoff}")

    individual_figures = {}
    combined_figure = None
    comparison_label = None

    if not args.no_individual:
        print("\nCreating individual plots...")
        individual_figures = plot_individual_files(file_data_list, args.precision, r_cutoff)

    if not args.no_comparison:
        print("\nCreating comparison plot...")
        comparison_label = prompt_for_comparison_label()
        combined_figure = plot_all_together(file_data_list, args.precision, comparison_label, r_cutoff, gr_cutoff)

    print("\nDone! All plots saved as HQ PNG files.")

    if args.interactive:
        interactive_menu(file_data_list, individual_figures, combined_figure, 
                        args.precision, r_cutoff, gr_cutoff, comparison_label or "Comparison of All Files")