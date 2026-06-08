import argparse
import os
import re


def detect_thermo_block(lines, start_idx):
    header = lines[start_idx].strip().split()
    data = []
    i = start_idx + 1

    while i < len(lines):
        parts = lines[i].strip().split()

        if len(parts) == 0:
            i += 1
            continue

        try:
            row = [float(x) for x in parts]
        except ValueError:
            break

        if len(parts) != len(header):
            break

        data.append(row)
        i += 1

    return header, data, i


def extract_atoms(lines):
    """
    Extract number of atoms from log
    """
    for line in lines:
        match = re.search(r"(\d+)\s+atoms", line)
        if match:
            return int(match.group(1))
    return None


def parse_log(file_path):
    """
    Extract thermo blocks ONLY after 'run' command.
    Also extract atom count.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    atoms = extract_atoms(lines)

    blocks = []
    i = 0
    in_run_section = False
    run_count = 0

    while i < len(lines):
        line = lines[i].strip().lower()

        # detect run command
        if line.startswith("run"):
            in_run_section = True
            run_count += 1
            print(f"[INFO] Found run command at line {i}")
            i += 1
            continue

        parts = lines[i].strip().split()

        # detect thermo block ONLY if inside run section
        if in_run_section and len(parts) > 0 and parts[0].lower() == "step":
            header, data, i = detect_thermo_block(lines, i)

            if data:
                blocks.append((header, data))
            continue

        i += 1

    if run_count == 0:
        print(f"[WARNING] No 'run' command found in {file_path}")

    return atoms, blocks


def merge_blocks(blocks):
    """
    Merge thermo blocks using Step column:
    - Detect continuation via step reset
    - Remove duplicate restart step
    """
    if not blocks:
        return None, [], False

    base_header = blocks[0][0]
    step_idx = base_header.index("Step")

    merged_data = []
    continuation_found = False

    last_step = None
    seen_steps = set()

    for header, data in blocks:
        if header != base_header:
            print("[WARNING] Header mismatch detected. Skipping block.")
            continue

        for row in data:
            step = int(row[step_idx])

            if last_step is not None and step < last_step:
                continuation_found = True
                print(f"[INFO] Continuation detected at step {step}")

            if step in seen_steps:
                print(f"[INFO] Removing duplicate step {step} (restart overlap)")
                continue

            merged_data.append(row)
            seen_steps.add(step)
            last_step = step

    return base_header, merged_data, continuation_found


def write_structured_log(header, data, atoms, output_file):
    """
    Write structured output:
    
    atoms
    <value>

    md_log
    <header>
    <data>
    """
    with open(output_file, "w") as f:
        # atoms section
        f.write("atoms\n")
        if atoms is not None:
            f.write(f"{atoms}\n\n")
        else:
            f.write("unknown\n\n")

        # md log section
        f.write("md_log\n")
        f.write(" ".join(header) + "\n")

        for row in data:
            f.write(" ".join(f"{x:.6g}" for x in row) + "\n")


def process_files(logfiles, output):
    all_blocks = []
    atoms = None

    for logfile in logfiles:
        file_atoms, blocks = parse_log(logfile)

        if file_atoms is not None and atoms is None:
            atoms = file_atoms

        if not blocks:
            print(f"[WARNING] No MD thermo data found in {logfile}")
            continue

        print(f"[INFO] Found {len(blocks)} MD thermo block(s) in {logfile}")
        all_blocks.extend(blocks)

    if not all_blocks:
        print("No valid data found.")
        return

    header, merged_data, continuation_found = merge_blocks(all_blocks)

    if not continuation_found:
        print("[WARNING] No continuation detected")

    write_structured_log(header, merged_data, atoms, output)
    print(f"[INFO] Structured log written to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LAMMPS MD log data into structured format"
    )
    parser.add_argument("logfiles", nargs="+")
    parser.add_argument("-o", "--output", default="merged.log")

    args = parser.parse_args()

    process_files(args.logfiles, args.output)


if __name__ == "__main__":
    main()