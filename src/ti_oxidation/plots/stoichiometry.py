import os
import colorsys
import zlib
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm

_STYLE_APPLIED = None

# Output resolution for saved figures. Figure size and font sizes come from the style file.
_SAVE_DPI = 300

# Get the path to the ACS style file (in same directory as this module)
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_ACS_STYLE_PATH = os.path.join(_MODULE_DIR, 'acs_style.mplstyle')

# Load custom ACS style
if os.path.exists(_ACS_STYLE_PATH):
    plt.style.use(_ACS_STYLE_PATH)
    _STYLE_APPLIED = "Custom ACS (American Chemical Society)"
else:
    _STYLE_APPLIED = "Default matplotlib (ACS style file not found)"

# Fixed palette keyed by base Ti index (or label hash), not by iteration order
# -- so a given physical group (e.g. Ti13-Ti14) is always the same color
# across every plot and every file, regardless of how many groups happen to
# be present.
#
# Hues come from a fixed set of _N_HUE_BINS evenly spaced points around the
# color wheel (15 deg apart at 24 bins), not from hashing straight into a
# continuous hue -- a handful of independent hashes landing in a continuous
# space can easily clump together by chance. Picking from a small set of
# guaranteed-far-apart hues rules that out. Saturation/value are then cycled
# using the rest of the hash so that the rarer case of two labels sharing a
# hue bin still look different.
_N_HUE_BINS = 24
_SAT_LEVELS = [0.95, 0.55, 1.0, 0.7]
_VAL_LEVELS = [0.85, 0.6]


def _binned_color(key):
    """Deterministic color for an integer key, spread across _N_HUE_BINS
    evenly-spaced hues plus cycled saturation/value for same-bin collisions."""
    bin_idx = key % _N_HUE_BINS
    variant = key // _N_HUE_BINS
    hue = bin_idx / _N_HUE_BINS
    saturation = _SAT_LEVELS[variant % len(_SAT_LEVELS)]
    value = _VAL_LEVELS[(variant // len(_SAT_LEVELS)) % len(_VAL_LEVELS)]
    return colorsys.hsv_to_rgb(hue, saturation, value)


def _index_color(index):
    return _binned_color(index)


def _label_color(label):
    """Deterministic color for a full group label (e.g. 'Ti6-Ti7', 'Ti6-top').
    Hashes the whole label, not just the base Ti index, so two groups sharing
    a base layer but capped differently (Ti6-Ti7 vs Ti6-top) still get
    different colors."""
    return _binned_color(zlib.crc32(label.encode('utf-8')))

# Axis/tick sizing shared by every plot in this module.
_LABEL_FONTSIZE = 13
_TICK_LABELSIZE = 11
_MAJOR_TICK_LENGTH = 6
_MINOR_TICK_LENGTH = 3


def report_style():
    """Report which matplotlib style is being used."""
    print(f'matplotlib style: {_STYLE_APPLIED}')


def _style_axes(ax=None, minor_x=True, minor_y=True):
    """Bigger axis labels, plus major (labelled) and minor (unlabelled) ticks."""
    ax = ax or plt.gca()
    ax.xaxis.label.set_fontsize(_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(_LABEL_FONTSIZE)
    if minor_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if minor_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', length=_MAJOR_TICK_LENGTH, labelsize=_TICK_LABELSIZE)
    ax.tick_params(which='minor', length=_MINOR_TICK_LENGTH)


def _outside_legend(ax=None, ncol=1, fontsize=8):
    """Legend placed outside the axes (right side), so it never sits on top of the data."""
    ax = ax or plt.gca()
    ax.legend(ncol=ncol, fontsize=fontsize, framealpha=0.9,
              loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


def load_frame_layers(data, f_pos):
    mask = data['layer_frame'] == f_pos
    idxs = np.where(mask)[0]

    entries = []
    for i in idxs:
        tag = str(data['layer_tag'][i])
        count = int(data['layer_count'][i])
        n_atoms = int(data['layer_end'][i] - data['layer_start'][i])
        zmean = float(data['layer_zmean'][i])
        entries.append((tag, count, n_atoms, zmean))

    entries.sort(key=lambda e: e[3])
    return entries


def _label_sort_key(label):
    """Sort 'Ti3-Ti4', 'Ti12-top', ... numerically by the base Ti layer (bottom to top)."""
    return int(label.split('-')[0][2:])


def _frame_sandwich_groups(entries):
    """
    For one frame's entries (as returned by load_frame_layers, sorted by z),
    collect all consecutive O layers directly above each Ti layer, up to the
    next Ti layer or the top of the slab.

    - Ti layers with no O above them are skipped (no group).
    - O layers below the lowest Ti layer are ignored.
    - Labels: 'Ti{a}-Ti{b}' when capped by a Ti layer, 'Ti{a}-top' for the topmost group.

    Returns: [(label, ti_atoms, o_atoms), ...] for this one frame.
    """
    n = len(entries)
    groups = []

    i = 0
    while i < n:
        if entries[i][0] != 'Ti':
            i += 1
            continue

        ti_tag, ti_count, ti_atoms, _ = entries[i]

        j = i + 1
        o_atoms = 0
        while j < n and entries[j][0] == 'O':
            o_atoms += entries[j][2]
            j += 1

        if j > i + 1:  # at least one O layer above
            upper = f'Ti{entries[j][1] + 1}' if j < n else 'top'
            label = f'Ti{ti_count + 1}-{upper}'
            groups.append((label, ti_atoms, o_atoms))

        i = j

    return groups


def _scan_frames(data):
    """
    Single pass over the trajectory that computes each frame's Ti/O sandwich
    groups once. The result is cached and reused by auto-detection and by
    the raw/filtered grouping step, so the npz arrays only get walked one
    time no matter how many of those downstream steps run.

    Returns: [(frame_num, n_ti, frame_groups), ...] in frame order, where
    frame_groups = [(label, ti_atoms, o_atoms), ...] for that frame (see
    _frame_sandwich_groups) and n_ti is that frame's detected Ti layer count.
    """
    frame_numbers = data['frame_numbers']
    n_frames = len(frame_numbers)

    cache = []
    for f_pos in tqdm(range(n_frames), desc='scanning frames'):
        entries = load_frame_layers(data, f_pos)
        frame_groups = _frame_sandwich_groups(entries)
        n_ti = sum(1 for e in entries if e[0] == 'Ti')
        cache.append((int(frame_numbers[f_pos]), n_ti, frame_groups))
    return cache


def auto_detect_expected_counts(frame_cache, threshold=0.9):
    """
    Look at every frame's (n_ti, n_o_groups) pair and find the most common
    one. If it covers at least `threshold` of all frames, return
    (expected_n_ti, expected_n_o_groups, frequency, counts). Otherwise raise
    ValueError with a diagnostic breakdown -- the data is too inconsistent
    to auto-detect a single expected shape, and guessing wrong here would
    silently pick the wrong frames to keep.
    """
    n_frames = len(frame_cache)
    if n_frames == 0:
        raise ValueError('auto-detection failed: no frames in this file.')

    counts = Counter((n_ti, len(frame_groups)) for _, n_ti, frame_groups in frame_cache)
    (mode_n_ti, mode_n_o), mode_hits = counts.most_common(1)[0]
    frequency = mode_hits / n_frames

    if frequency < threshold:
        breakdown = ', '.join(f'(n_ti={p[0]}, n_o_groups={p[1]}): {hits} frames ({hits / n_frames:.1%})'
                               for p, hits in counts.most_common(5))
        raise ValueError(
            f'auto-detection failed: the most common shape (n_ti={mode_n_ti}, n_o_groups={mode_n_o}) '
            f'covers only {frequency:.1%} of {n_frames} frames (need >= {threshold:.1%}). '
            f'Top shapes seen: {breakdown}. Pass --expected-n-ti/--expected-n-o-groups manually instead.'
        )

    return mode_n_ti, mode_n_o, frequency, counts


def group_frames(frame_cache, expected_n_ti=None, expected_n_o_groups=None):
    """
    Build the raw and filtered Ti/O sandwich groupings from a pre-scanned
    frame_cache (see _scan_frames). Always returns the unfiltered ("raw")
    grouping. When expected_n_ti and/or expected_n_o_groups are given, also
    builds a "filtered" grouping: a frame is left out of it (contributes no
    data point to any group) if its detected Ti layer count, or its detected
    number of O-sandwich groups, doesn't match the expected value. Cluster
    analysis occasionally produces a spurious extra layer or misses a real
    one; since there's no reliable way to tell which detected layer/group is
    the bogus one, the whole frame is treated as an outlier and dropped
    rather than guessing which part of it to keep. Dropped frames are
    recorded in log_entries. When neither expected_* is given, filtered ==
    raw and log_entries is empty.

    Returns: (raw_groups, filtered_groups, log_entries), each raw/filtered
    groups dict shaped {label: [(frame, ti_atoms, o_atoms), ...]}.
    """
    filtering_on = expected_n_ti is not None or expected_n_o_groups is not None

    raw_groups = {}
    filtered_groups = {}
    log_entries = []

    for frame_num, n_ti, frame_groups in frame_cache:
        for label, ti_atoms, o_atoms in frame_groups:
            raw_groups.setdefault(label, []).append((frame_num, ti_atoms, o_atoms))

        if not filtering_on:
            continue

        n_o_groups = len(frame_groups)
        reasons = []
        if expected_n_ti is not None and n_ti != expected_n_ti:
            reasons.append('ti_count')
        if expected_n_o_groups is not None and n_o_groups != expected_n_o_groups:
            reasons.append('o_group_count')

        if reasons:
            log_entries.append({
                'frame': frame_num,
                'n_ti_detected': n_ti,
                'expected_n_ti': expected_n_ti,
                'n_o_groups_detected': n_o_groups,
                'expected_n_o_groups': expected_n_o_groups,
                'reason': '+'.join(reasons),
            })
            continue

        for label, ti_atoms, o_atoms in frame_groups:
            filtered_groups.setdefault(label, []).append((frame_num, ti_atoms, o_atoms))

    if not filtering_on:
        filtered_groups = raw_groups

    return raw_groups, filtered_groups, log_entries


def compute_sandwich_groups(data, expected_n_ti=None, expected_n_o_groups=None):
    """Convenience wrapper: scan a file and group it in one call. Prefer
    _scan_frames + group_frames directly when auto-detection or repeated
    grouping on the same file is also needed, to avoid rescanning."""
    frame_cache = _scan_frames(data)
    return group_frames(frame_cache, expected_n_ti, expected_n_o_groups)


def write_dropped_frames_log(log_entries, log_path, header_lines=None):
    """Write the frames group_frames dropped as outliers to a CSV log,
    with optional descriptive comment lines (e.g. how expected_n_ti/
    expected_n_o_groups were determined) before the header row."""
    with open(log_path, 'w') as fh:
        for line in header_lines or []:
            fh.write(f'# {line}\n')
        fh.write('frame,n_ti_detected,expected_n_ti,n_o_groups_detected,expected_n_o_groups,reason\n')
        for e in log_entries:
            fh.write(f"{e['frame']},{e['n_ti_detected']},{e['expected_n_ti']},"
                     f"{e['n_o_groups_detected']},{e['expected_n_o_groups']},{e['reason']}\n")
    print(f'  dropped {len(log_entries)} frame(s) as outliers, logged -> {log_path}')


def compute_sandwich_ratios(groups):
    """Ratio = total O atoms / Ti atoms in the base layer, from compute_sandwich_groups output."""
    results = {}
    for label, points in groups.items():
        results[label] = [(frame, o_atoms / ti_atoms) for frame, ti_atoms, o_atoms in points]
    return results


def plot_timeseries(results, out_path):
    plt.figure(figsize=(7, 5))
    for label in sorted(results.keys(), key=_label_sort_key):
        points = sorted(results[label])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        step = max(1, len(x) // 100)
        plt.plot(x, y, marker='o', markersize=2, linewidth=0.6, markevery=step,
                  color=_label_color(label), label=label)

    plt.xlabel('frame')
    plt.ylabel('O atoms / Ti atoms')
    _outside_legend(ncol=1, fontsize=8)
    _style_axes()
    plt.savefig(out_path, dpi=_SAVE_DPI, bbox_inches='tight')
    plt.close()


def plot_time_average(results, out_path):
    labels = sorted(results.keys(), key=_label_sort_key)
    means = [np.mean([p[1] for p in results[l]]) for l in labels]
    stds = [np.std([p[1] for p in results[l]]) for l in labels]
    colors = [_label_color(l) for l in labels]

    plt.figure()
    plt.bar(labels, means, yerr=stds, capsize=4, color=colors)
    plt.xlabel('Ti layer group')
    plt.ylabel('time-averaged O/Ti ratio')
    plt.xticks(rotation=45, ha='right')
    _style_axes(minor_x=False)
    plt.savefig(out_path, dpi=_SAVE_DPI, bbox_inches='tight')
    plt.close()


def plot_sandwich_counts(groups, out_path):
    """Debug plot: raw Ti (base layer) and O (sandwiched) atom counts per group,
    two lines per ratio-group, same color as that group's ratio line -- marker
    (o vs s) and linestyle (solid vs dashed) tell Ti and O apart."""
    plt.figure(figsize=(7, 5))
    for label in sorted(groups.keys(), key=_label_sort_key):
        points = sorted(groups[label])
        x = [p[0] for p in points]
        ti_y = [p[1] for p in points]
        o_y = [p[2] for p in points]
        step = max(1, len(x) // 100)
        color = _label_color(label)

        plt.plot(x, ti_y, marker='o', markersize=2, linewidth=0.6,
                  linestyle='-', markevery=step, color=color, label=f'{label} Ti')
        plt.plot(x, o_y, marker='s', markersize=2, linewidth=0.6,
                  linestyle='--', markevery=step, color=color, label=f'{label} O')

    plt.xlabel('frame')
    plt.ylabel('atom count')
    _outside_legend(ncol=1, fontsize=7)
    _style_axes()
    plt.savefig(out_path, dpi=_SAVE_DPI, bbox_inches='tight')
    plt.close()


def compute_layer_counts_timeseries(data):
    frame_numbers = data['frame_numbers']
    n_frames = len(frame_numbers)

    series = {}
    for f_pos in range(n_frames):
        entries = load_frame_layers(data, f_pos)
        for tag, count, n_atoms, zmean in entries:
            series.setdefault((tag, count), []).append((int(frame_numbers[f_pos]), n_atoms))

    return series


def plot_layer_counts_timeseries(series, out_path):
    keys = sorted(series.keys(), key=lambda k: (0 if k[0] == 'Ti' else 1, k[1]))

    plt.figure()
    for tag, count in keys:
        points = sorted(series[(tag, count)])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        color = _index_color(count)
        linestyle = '-' if tag == 'Ti' else '--'
        plt.plot(x, y, color=color, linestyle=linestyle, linewidth=0.8, label=f'{tag}{count + 1}')

    plt.xlabel('frame')
    plt.ylabel('atom count')
    _outside_legend(ncol=1, fontsize=7)
    _style_axes()
    plt.savefig(out_path, dpi=_SAVE_DPI, bbox_inches='tight')
    plt.close()


def print_sandwich_stats(results, label_prefix=''):
    print(f'O/Ti ratio stats{label_prefix} (O layers above each Ti layer):')
    for label in sorted(results.keys(), key=_label_sort_key):
        y = [p[1] for p in results[label]]
        print(f'  {label:>10s}  n={len(y):5d}  mean={np.mean(y):.4f}  std={np.std(y):.4f}  '
              f'min={np.min(y):.4f}  max={np.max(y):.4f}')


def _plot_all(results, groups, ts_path, avg_path, counts_path):
    plot_timeseries(results, ts_path)
    plot_time_average(results, avg_path)
    plot_sandwich_counts(groups, counts_path)


def run(args):
    report_style()
    os.makedirs(args.out_dir, exist_ok=True)

    manual_n_ti = getattr(args, 'expected_n_ti', None)
    manual_n_o_groups = getattr(args, 'expected_n_o_groups', None)
    no_filter = getattr(args, 'no_filter', False)
    auto_threshold = getattr(args, 'auto_threshold', 0.9)

    for f in args.files:
        print(f'processing: {f}')
        try:
            data = np.load(f)
            stem = os.path.splitext(os.path.basename(f))[0]

            frame_cache = _scan_frames(data)

            expected_n_ti = manual_n_ti
            expected_n_o_groups = manual_n_o_groups
            auto_note = None

            if not no_filter:
                need_auto_ti = expected_n_ti is None
                need_auto_o = expected_n_o_groups is None

                if need_auto_ti or need_auto_o:
                    # Auto-detect is the default whenever a count wasn't
                    # given manually; manual values (checked above) always
                    # take precedence over whatever auto-detection finds.
                    auto_n_ti, auto_n_o, frequency, _ = auto_detect_expected_counts(
                        frame_cache, threshold=auto_threshold)
                    print(f'  auto-detected shape: n_ti={auto_n_ti}, n_o_groups={auto_n_o} '
                          f'({frequency:.1%} of {len(frame_cache)} frames agree)')
                    auto_note = (f'auto-detected majority shape: n_ti={auto_n_ti}, '
                                 f'n_o_groups={auto_n_o} ({frequency:.1%} of frames, '
                                 f'threshold={auto_threshold:.1%})')
                    if need_auto_ti:
                        expected_n_ti = auto_n_ti
                    if need_auto_o:
                        expected_n_o_groups = auto_n_o

            groups_raw, groups_filt, log_entries = group_frames(
                frame_cache, expected_n_ti=expected_n_ti, expected_n_o_groups=expected_n_o_groups)
            results_raw = compute_sandwich_ratios(groups_raw)

            filtering_on = expected_n_ti is not None or expected_n_o_groups is not None

            if not filtering_on:
                print_sandwich_stats(results_raw)

                ts_path = os.path.join(args.out_dir, f'{stem}_sandwich_timeseries.png')
                avg_path = os.path.join(args.out_dir, f'{stem}_sandwich_time_average.png')
                counts_path = os.path.join(args.out_dir, f'{stem}_sandwich_counts_timeseries.png')
                _plot_all(results_raw, groups_raw, ts_path, avg_path, counts_path)

                print(f'  -> {ts_path}')
                print(f'  -> {avg_path}')
                print(f'  -> {counts_path}')
            else:
                results_filt = compute_sandwich_ratios(groups_filt)

                print_sandwich_stats(results_raw, label_prefix=' (raw)')
                print_sandwich_stats(results_filt, label_prefix=' (outliers removed)')

                log_path = os.path.join(args.out_dir, f'{stem}_dropped_frames.log')
                header_lines = [f'expected_n_ti={expected_n_ti}', f'expected_n_o_groups={expected_n_o_groups}']
                if manual_n_ti is not None:
                    header_lines[0] += ' (manual)'
                if manual_n_o_groups is not None:
                    header_lines[1] += ' (manual)'
                if auto_note:
                    header_lines.append(auto_note)
                write_dropped_frames_log(log_entries, log_path, header_lines=header_lines)

                ts_raw = os.path.join(args.out_dir, f'{stem}_sandwich_timeseries_raw.png')
                avg_raw = os.path.join(args.out_dir, f'{stem}_sandwich_time_average_raw.png')
                counts_raw = os.path.join(args.out_dir, f'{stem}_sandwich_counts_timeseries_raw.png')
                _plot_all(results_raw, groups_raw, ts_raw, avg_raw, counts_raw)

                ts_filt = os.path.join(args.out_dir, f'{stem}_sandwich_timeseries_filtered.png')
                avg_filt = os.path.join(args.out_dir, f'{stem}_sandwich_time_average_filtered.png')
                counts_filt = os.path.join(args.out_dir, f'{stem}_sandwich_counts_timeseries_filtered.png')
                _plot_all(results_filt, groups_filt, ts_filt, avg_filt, counts_filt)

                print(f'  -> {ts_raw}')
                print(f'  -> {avg_raw}')
                print(f'  -> {counts_raw}')
                print(f'  -> {ts_filt}')
                print(f'  -> {avg_filt}')
                print(f'  -> {counts_filt}')
        except OSError as e:
            print(f'  skipped ({e}) -- is a png open in another program?')