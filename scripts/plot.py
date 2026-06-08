import argparse
import matplotlib.pyplot as plt
import os


def parse_merged_log(file_path):
    steps = []
    poteng = []
    mytemp = []
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
        raise ValueError(f"Required columns not found in {file_path}")

    # --- data ---
    for line in lines[header_idx + 1:]:
        parts = line.strip().split()

        if len(parts) != len(header):
            continue

        try:
            steps.append(int(float(parts[step_idx])))
            poteng.append(float(parts[poteng_idx]))
            mytemp.append(float(parts[mytemp_idx]))
        except ValueError:
            continue

    if not steps:
        raise ValueError(f"No valid data in {file_path}")

    return steps, poteng, mytemp, atoms


# --- FIXED: unified trimming ---
def trim_all(steps, poteng, poteng_pa, mytemp, xmin, xmax):
    t_s, t_p, t_pa, t_t = [], [], [], []

    for s, p, pa, t in zip(steps, poteng, poteng_pa, mytemp):
        if (xmin is None or s >= xmin) and (xmax is None or s <= xmax):
            t_s.append(s)
            t_p.append(p)
            t_pa.append(pa)
            t_t.append(t)

    return t_s, t_p, t_pa, t_t


def plot_single(steps, poteng, poteng_pa, mytemp, output):
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    axs[0].plot(steps, poteng)
    axs[0].set_ylabel("Potential Energy")
    axs[0].set_title("Potential Energy vs Timestep")

    axs[1].plot(steps, poteng_pa)
    axs[1].set_ylabel("PotEng / Atom")
    axs[1].set_title("Potential Energy per Atom")

    axs[2].plot(steps, mytemp)
    axs[2].set_xlabel("Timestep")
    axs[2].set_ylabel("c_mytemp")
    axs[2].set_title("Temperature")

    fig.suptitle(output.split('.')[0])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output)
    plt.close()


def plot_combined(all_data, output_prefix):
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    # Potential Energy
    for label, steps, poteng, _, _ in all_data:
        axs[0].plot(steps, poteng, label=label)

    axs[0].set_ylabel("Potential Energy")
    axs[0].set_title("Potential Energy Comparison")
    axs[0].legend()

    # Per Atom
    for label, steps, _, _, poteng_pa in all_data:
        axs[1].plot(steps, poteng_pa, label=label)

    axs[1].set_ylabel("PotEng / Atom")
    axs[1].set_title("Per-Atom Energy Comparison")
    axs[1].legend()

    # Temperature
    for label, steps, _, mytemp, _ in all_data:
        axs[2].plot(steps, mytemp, label=label)

    axs[2].set_xlabel("Timestep")
    axs[2].set_ylabel("c_mytemp")
    axs[2].set_title("Temperature Comparison")
    axs[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{output_prefix}_combined.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot merged LAMMPS log data")
    parser.add_argument("logfiles", nargs='+')
    parser.add_argument("--out_prefix", default="output")
    parser.add_argument("--xmin", type=int, default=None)
    parser.add_argument("--xmax", type=int, default=None)

    args = parser.parse_args()

    all_data = []

    for logfile in args.logfiles:
        try:
            steps, poteng, mytemp, atoms = parse_merged_log(logfile)
        except ValueError as e:
            print(e)
            continue

        # compute per-atom BEFORE trimming
        poteng_pa = [p / atoms for p in poteng]

        # unified trimming
        steps, poteng, poteng_pa, mytemp = trim_all(
            steps, poteng, poteng_pa, mytemp,
            args.xmin, args.xmax
        )

        label = os.path.basename(logfile)
        all_data.append((label, steps, poteng, mytemp, poteng_pa))

        out_file = f"{args.out_prefix}_{label}.png"
        plot_single(steps, poteng, poteng_pa, mytemp, out_file)

        print(f"Saved: {out_file}")

    if all_data:
        plot_combined(all_data, args.out_prefix)
        print("Saved combined plots")
    else:
        print("No valid data found")


if __name__ == "__main__":
    main()