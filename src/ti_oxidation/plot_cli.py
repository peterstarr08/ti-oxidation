#!/usr/bin/env python3
"""
Unified CLI for scientific plotting.
Dispatch to different plot modules via subparsers.

Usage:
  python plot.py gr [options] file1.csv file2.csv ...
  python plot.py md [options] log1 log2 ...
  python plot.py coordination --density DENSITY [options] file1.dat ...
  python plot.py stoic [options] layer_data1.npz layer_data2.npz ...
  python plot.py --help
"""
import sys
import argparse
import ti_oxidation.plots.gr as gr_plot
import ti_oxidation.plots.md_log as md_plot
import ti_oxidation.plots.md_fluct as md_fluct_plot
import ti_oxidation.plots.md_lcurve as md_lcurve_plot
import ti_oxidation.plots.coordination as coord_plot
import ti_oxidation.plots.stoichiometry as stoic_plot


def create_gr_parser(subparsers):
    """
    Create and configure the 'gr' subparser for g(r) plotting.
    """
    gr_parser = subparsers.add_parser(
        'gr',
        help='Plot g(r) vs r from CSV files (radial distribution function)',
        description='Plot radial distribution functions from CSV data with SciencePlots styling.'
    )
    gr_parser.add_argument(
        'files',
        nargs='+',
        help='CSV files to plot (must contain r and g(r) columns)'
    )
    gr_parser.add_argument(
        '-p', '--precision',
        type=int,
        default=2,
        metavar='N',
        help='Decimal precision for axis labels (default: 2, range: 0-10)'
    )
    gr_parser.add_argument(
        '--no-individual',
        action='store_true',
        help='Skip individual plots (only create comparison)'
    )
    gr_parser.add_argument(
        '--no-comparison',
        action='store_true',
        help='Skip comparison plot (only create individual plots)'
    )
    gr_parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Show interactive menu to inspect plots (default: off)'
    )
    gr_parser.set_defaults(func=gr_plot.run)


def create_md_parser(subparsers):
    """
    Create and configure the 'md' subparser for LAMMPS MD log plotting.
    """
    md_parser = subparsers.add_parser(
        'md',
        help='Plot LAMMPS MD logs (potential energy, temperature)',
        description='Parse and plot merged LAMMPS MD logs with ACS styling for publication.'
    )
    md_parser.add_argument(
        'files',
        nargs='+',
        help='LAMMPS merged log files to plot'
    )
    md_parser.add_argument(
        '--out-dir',
        default='plots_md',
        metavar='DIR',
        help='Output directory for plots (default: plots_md)'
    )
    md_parser.add_argument(
        '--mark-temp',
        type=float,
        default=973,
        metavar='K',
        help='Temperature reference line in K (default: 973 K)'
    )
    md_parser.add_argument(
        '--timestep',
        type=float,
        default=1.0,
        metavar='PS',
        help='Picoseconds per LAMMPS timestep (default: 1.0 ps)'
    )
    md_parser.add_argument(
        '--xmin',
        type=int,
        default=None,
        metavar='STEP',
        help='Trim: minimum timestep'
    )
    md_parser.add_argument(
        '--xmax',
        type=int,
        default=None,
        metavar='STEP',
        help='Trim: maximum timestep'
    )
    md_parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Interactive mode: rename files and set ps-per-frame (default: off)'
    )
    md_parser.set_defaults(func=md_plot.run)


def create_fluct_parser(subparsers):
    """
    Create and configure the 'fluct' subparser for LAMMPS MD fluctuation analysis.
    """
    fluct_parser = subparsers.add_parser(
        'fluct',
        help='Fluctuation analysis of LAMMPS MD log (Gaussian fitting)',
        description='Analyze temperature and energy fluctuations with Gaussian fitting and thermostat metrics.'
    )
    fluct_parser.add_argument(
        'file',
        help='LAMMPS log file to analyze'
    )
    fluct_parser.add_argument(
        '--out-dir',
        default='',
        metavar='DIR',
        help='Output directory (auto-generated if empty)'
    )
    fluct_parser.add_argument(
        '--step-min',
        type=int,
        default=None,
        metavar='STEP',
        help='Minimum timestep (default: none)'
    )
    fluct_parser.add_argument(
        '--step-max',
        type=int,
        default=None,
        metavar='STEP',
        help='Maximum timestep (default: none)'
    )
    fluct_parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Interactive mode: set timestep range and custom label'
    )
    fluct_parser.set_defaults(func=md_fluct_plot.run)


def create_lcurve_parser(subparsers):
    """
    Create and configure the 'lcurve' subparser for learning curve plotting.
    """
    lcurve_parser = subparsers.add_parser(
        'lcurve',
        help='Plot learning curves from deep learning training logs',
        description='Parse and plot lcurve.out (MLIP training logs) with support for batch processing.'
    )
    lcurve_parser.add_argument(
        'files',
        nargs='*',
        help='lcurve.out files to plot'
    )
    lcurve_parser.add_argument(
        '--dir',
        default=None,
        metavar='DIR',
        help='Directory to search for lcurve.out files'
    )
    lcurve_parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='Enable recursive search (default: True, use --no-recursive to disable)'
    )
    lcurve_parser.add_argument(
        '--no-recursive',
        action='store_false',
        dest='recursive',
        help='Disable recursive search (only search top level)'
    )
    lcurve_parser.add_argument(
        '--out-dir',
        default='learning_curves',
        metavar='DIR',
        help='Output directory (default: learning_curves)'
    )
    lcurve_parser.add_argument(
        '--logx',
        action='store_true',
        help='Use log scale for x-axis (step)'
    )
    lcurve_parser.set_defaults(func=md_lcurve_plot.run)


def create_coordination_parser(subparsers):
    """
    Create and configure the 'coordination' subparser for coordination number integration.
    """
    coord_parser = subparsers.add_parser(
        'coordination',
        help='Compute coordination number from g(r) via integration',
        description='Integrate g(r) to compute cumulative coordination number: CN(r) = ∫ 4π ρ r² g(r) dr'
    )
    coord_parser.add_argument(
        'files',
        nargs='+',
        help='Input files (.dat xmgrace or .csv) containing r and g(r) data'
    )
    coord_parser.add_argument(
        '--density',
        type=float,
        required=True,
        metavar='DENSITY',
        help='Normalization density (atoms/Å³) [REQUIRED]'
    )
    coord_parser.add_argument(
        '-o', '--out-dir',
        default='',
        metavar='DIR',
        help='Output directory (default: adjacent to source files)'
    )
    coord_parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Interactive mode: plt.show() and query coordination numbers (default: off)'
    )
    coord_parser.set_defaults(func=coord_plot.run)


def create_stoic_parser(subparsers):
    """
    Create and configure the 'stoic' subparser for Ti-O sandwich stoichiometry plots.
    """
    stoic_parser = subparsers.add_parser(
        'stoic',
        help='Plot O/Ti sandwich ratios per frame from layer_data.npz',
        description='For each frame, find O layers sandwiched between two Ti layers and plot '
                     '(O atoms in that layer) / (Ti atoms in the bottom bounding layer) vs frame, '
                     'one line per Ti-Ti pair, plus a time-averaged summary plot.'
    )
    stoic_parser.add_argument(
        'files',
        nargs='+',
        help='layer_data.npz files (from trajectory_layer_clustering) to plot'
    )
    stoic_parser.add_argument(
        '--out-dir',
        default='plots_stoic',
        metavar='DIR',
        help='Output directory for plots (default: plots_stoic)'
    )
    stoic_parser.add_argument(
        '--expected-n-ti',
        type=int,
        default=None,
        metavar='N',
        dest='expected_n_ti',
        help='Expected number of Ti layers per frame; frames that detect a different count '
             'are dropped as outliers. Overrides auto-detection for this count if given '
             '(default: auto-detect from the majority shape across frames)'
    )
    stoic_parser.add_argument(
        '--expected-n-o-groups',
        type=int,
        default=None,
        metavar='N',
        dest='expected_n_o_groups',
        help='Expected number of O-sandwich groups per frame (consecutive O layers between '
             'two Ti layers, or above the topmost Ti layer, count as one group); frames that '
             'detect a different count are dropped as outliers. Overrides auto-detection for '
             'this count if given (default: auto-detect from the majority shape across frames)'
    )
    stoic_parser.add_argument(
        '--auto-threshold',
        type=float,
        default=0.9,
        metavar='FRACTION',
        dest='auto_threshold',
        help='Minimum fraction of frames that must share the same (n_ti, n_o_groups) shape '
             'for auto-detection to accept it as the expected shape; otherwise the run aborts '
             'with an error (default: 0.9). Only used for whichever of --expected-n-ti / '
             '--expected-n-o-groups is not given manually'
    )
    stoic_parser.add_argument(
        '--no-filter',
        action='store_true',
        dest='no_filter',
        help='Disable outlier filtering entirely (no auto-detection, no manual expected '
             'counts): plot every frame as-is, same as before auto-detection existed'
    )
    stoic_parser.set_defaults(func=stoic_plot.run)


def main():
    """
    Main CLI entry point with subparsers.
    """
    parser = argparse.ArgumentParser(
        prog='plot',
        description='Unified CLI for scientific plotting (g(r), MD logs, coordination numbers, etc.)',
        epilog='Examples:\n'
               '  python plot.py gr data1.csv data2.csv\n'
               '  python plot.py md log1.txt log2.txt -i --out-dir plots_sim\n'
               '  python plot.py md --mark-temp 1000 --timestep 0.001 log.txt\n'
               '  python plot.py fluct log.txt --step-min 1000 --step-max 50000\n'
               '  python plot.py fluct log.txt -i --out-dir fluct_analysis\n'
               '  python plot.py coordination --density 0.085 grdata.dat\n'
               '  python plot.py coordination --density 0.085 -i file1.dat file2.dat\n'
               '  python plot.py coordination --density 0.085 -o ./cn_output file1.dat file2.dat\n'
               '  python plot.py lcurve lcurve.out\n'
               '  python plot.py lcurve --dir ./training --recursive --logx\n'
               '  python plot.py stoic layer_data.npz --out-dir plots_stoic\n'
               '  python plot.py stoic layer_data.npz  # auto-detects expected shape\n'
               '  python plot.py stoic layer_data.npz --expected-n-ti 4  # o-groups still auto\n'
               '  python plot.py stoic layer_data.npz --expected-n-ti 4 --expected-n-o-groups 8\n'
               '  python plot.py stoic layer_data.npz --auto-threshold 0.8\n'
               '  python plot.py stoic layer_data.npz --no-filter\n'
               '  python plot.py --help',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(
        title='subcommands',
        description='Available plot types',
        dest='command',
        required=True
    )

    # Register subcommands
    create_gr_parser(subparsers)
    create_md_parser(subparsers)
    create_fluct_parser(subparsers)
    create_lcurve_parser(subparsers)
    create_coordination_parser(subparsers)
    create_stoic_parser(subparsers)
    # Future: create_rdf_parser(subparsers), create_msd_parser(subparsers), etc.

    args = parser.parse_args()

    # Validate args (gr-specific)
    if hasattr(args, 'precision'):
        if args.precision < 0 or args.precision > 10:
            parser.error("Precision must be between 0 and 10")
        if args.no_individual and args.no_comparison:
            parser.error("Cannot skip both individual and comparison plots")

    # Dispatch to the appropriate module's run function
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()