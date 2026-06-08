#!/usr/bin/env python3

import argparse
import numpy as np
from math import sqrt
import os

import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_combined_with_stats(temps, pes, atoms, logfile, output_file):
    # --- layout: 2 rows, 2 cols (right column = stats) ---
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1])

    ax_temp = fig.add_subplot(gs[0, 0])
    ax_pe = fig.add_subplot(gs[1, 0])
    text_ax = fig.add_subplot(gs[:, 1])
    text_ax.axis("off")

    # --- Temperature ---
    mu_t, std_t = norm.fit(temps)
    x_t = np.linspace(min(temps), max(temps), 200)
    pdf_t = norm.pdf(x_t, mu_t, std_t)

    ax_temp.hist(temps, bins=50, density=True)
    ax_temp.plot(x_t, pdf_t)
    ax_temp.set_title(f"Temperature (mu={mu_t:.3f}, sigma={std_t:.3f})")
    ax_temp.set_xlabel("Temperature")
    ax_temp.set_ylabel("Density")

    # --- Potential Energy ---
    mu_p, std_p = norm.fit(pes)
    x_p = np.linspace(min(pes), max(pes), 200)
    pdf_p = norm.pdf(x_p, mu_p, std_p)

    ax_pe.hist(pes, bins=50, density=True)
    ax_pe.plot(x_p, pdf_p)
    ax_pe.set_title(f"Potential Energy (mu={mu_p:.3f}, sigma={std_p:.3f})")
    ax_pe.set_xlabel("Potential Energy")
    ax_pe.set_ylabel("Density")

    # --- Stats text ---
    stats_lines = [
        "Temperature:",
        f"Mean = {temps.mean():.5f}",
        f"Std  = {temps.std():.5f}",
        f"Var  = {temps.var():.5f}",
        f"Fluct = {temps.std()/temps.mean():.5f}",
        "",
        "Potential Energy:",
        f"Mean = {pes.mean():.5f}",
        f"Std  = {pes.std():.5f}",
        f"Var  = {pes.var():.5f}",
        f"Fluct = {pes.std()/pes.mean():.5f}",
        ""
    ]

    if atoms is not None:
        stats_lines.append(f"Atoms = {atoms}")
        stats_lines.append(f"1/sqrt(N) = {1/sqrt(atoms):.5f}")
    else:
        stats_lines.append("Atoms = N/A")

    full_text = "\n".join(stats_lines)

    # place text in right panel
    text_ax.text(0, 1, full_text, va='top', fontsize=9, family="monospace")

    # --- Title ---
    fig.suptitle(os.path.basename(logfile))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Gaussian analysis of merged LAMMPS log")

    p.add_argument("logfile")

    p.add_argument("--step_min", type=int, default=None)
    p.add_argument("--step_max", type=int, default=None)

    p.add_argument("--temp_min", type=float, default=None)
    p.add_argument("--temp_max", type=float, default=None)

    p.add_argument("--pe_min", type=float, default=None)
    p.add_argument("--pe_max", type=float, default=None)

    p.add_argument("--suffix", default="output.png")

    return p.parse_args()


def parse_structured_log(file_path):
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

    return atoms, data


def main():
    args = parse_args()

    atoms, data = parse_structured_log(args.logfile)

    filtered = []
    for d in data:
        step = d["Step"]
        temp = d["Temp"]
        pe = d["PotEng"]

        if args.step_min is not None and step < args.step_min:
            continue
        if args.step_max is not None and step > args.step_max:
            continue

        if args.temp_min is not None and temp < args.temp_min:
            continue
        if args.temp_max is not None and temp > args.temp_max:
            continue

        if args.pe_min is not None and pe < args.pe_min:
            continue
        if args.pe_max is not None and pe > args.pe_max:
            continue

        filtered.append(d)

    if not filtered:
        print("No data in range")
        return

    temps = np.array([d["Temp"] for d in filtered])
    pes = np.array([d["PotEng"] for d in filtered])

    output_file = args.suffix
    plot_combined_with_stats(temps, pes, atoms, args.logfile, output_file)

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()