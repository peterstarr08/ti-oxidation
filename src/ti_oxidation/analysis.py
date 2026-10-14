import argparse

from ti_oxidation.tio_analysis.stoichimetry_study import cluster_trajectory


def add_stoic_subparser(subparsers):
    parser = subparsers.add_parser('stoic', help='cluster Ti/O layers across a LAMMPS dump trajectory')
    parser.add_argument('traj_path', help='path to LAMMPS dump trajectory file')
    parser.add_argument('out_dir', help='output directory for debug plots and layer_data.npz')
    parser.add_argument('--method', choices=['hdbscan', 'dbscan', 'svc'], default='hdbscan')

    # hdbscan params
    parser.add_argument('--ti-min', type=int, default=10)
    parser.add_argument('--ti-max', type=int, default=200)
    parser.add_argument('--o-min', type=int, default=4)
    parser.add_argument('--o-max', type=int, default=800)

    # dbscan params
    parser.add_argument('--ti-eps', type=float, default=0.8)
    parser.add_argument('--ti-min-samples', type=int, default=2)
    parser.add_argument('--o-eps', type=float, default=0.8)
    parser.add_argument('--o-min-samples', type=int, default=2)

    # svc params
    parser.add_argument('--n-ti-layers', type=int)
    parser.add_argument('--n-o-layers', type=int)
    parser.add_argument('--ti-kernel', default='rbf')
    parser.add_argument('--ti-C', type=float, default=10.0)
    parser.add_argument('--ti-gamma', default='scale')
    parser.add_argument('--o-kernel', default='rbf')
    parser.add_argument('--o-C', type=float, default=10.0)
    parser.add_argument('--o-gamma', default='scale')
    parser.add_argument('--ti-prob-threshold', type=float, default=0.9)
    parser.add_argument('--o-prob-threshold', type=float, default=0.9)
    parser.add_argument('--ti-min-layer-size', type=int, default=None)
    parser.add_argument('--ti-max-layer-size', type=int, default=None)
    parser.add_argument('--o-min-layer-size', type=int, default=None)
    parser.add_argument('--o-max-layer-size', type=int, default=None)

    parser.add_argument('--n-debug-plots', type=int, default=40)
    parser.add_argument('--n-workers', type=int, default=9)
    parser.set_defaults(func=run_stoic)


def run_stoic(args):
    if args.method == 'hdbscan':
        method_kwargs = dict(
            ti_min=args.ti_min,
            ti_max=args.ti_max,
            o_min=args.o_min,
            o_max=args.o_max,
        )
    elif args.method == 'dbscan':
        method_kwargs = dict(
            ti_eps=args.ti_eps,
            ti_min_samples=args.ti_min_samples,
            o_eps=args.o_eps,
            o_min_samples=args.o_min_samples,
        )
    else:
        method_kwargs = dict(
            n_ti_layers=args.n_ti_layers,
            n_o_layers=args.n_o_layers,
            ti_kernel=args.ti_kernel,
            ti_C=args.ti_C,
            ti_gamma=args.ti_gamma,
            o_kernel=args.o_kernel,
            o_C=args.o_C,
            o_gamma=args.o_gamma,
            ti_prob_threshold=args.ti_prob_threshold,
            o_prob_threshold=args.o_prob_threshold,
            ti_min_layer_size=args.ti_min_layer_size,
            ti_max_layer_size=args.ti_max_layer_size,
            o_min_layer_size=args.o_min_layer_size,
            o_max_layer_size=args.o_max_layer_size,
        )

    out_path = cluster_trajectory(
        args.traj_path,
        args.out_dir,
        method=args.method,
        method_kwargs=method_kwargs,
        n_debug_plots=args.n_debug_plots,
        n_workers=args.n_workers,
    )
    print(out_path)


def main():
    parser = argparse.ArgumentParser(prog='ti_analysis', description='Ti-oxidation analysis CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    add_stoic_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)