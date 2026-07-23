import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_clusters(snapshot: Atoms, labels, filepath: str):
    ti_atoms = Atoms([a for a in snapshot if a.symbol == "Ti"])
    pos = ti_atoms.get_positions()

    unique_labels = np.unique(labels)
    cmap = plt.cm.tab20

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    legend_elements = []

    for i, label in enumerate([l for l in unique_labels if l != -1]):
        mask = labels == label
        color = cmap(i % 20)
        ax1.scatter(pos[mask, 0], pos[mask, 2], s=5, color=color)
        ax2.scatter(pos[mask, 1], pos[mask, 2], s=5, color=color)
        legend_elements.append(
            Line2D([0], [0], marker='o', linestyle='', color=color,
                   label=f'Cluster {label} ({mask.sum()})')
        )

    noise_mask = labels == -1
    if np.any(noise_mask):
        ax1.scatter(pos[noise_mask, 0], pos[noise_mask, 2], s=5, color='black')
        ax2.scatter(pos[noise_mask, 1], pos[noise_mask, 2], s=5, color='black')
        legend_elements.append(
            Line2D([0], [0], marker='o', linestyle='', color='black',
                   label=f'Noise ({noise_mask.sum()})')
        )

    ax1.set_xlabel("x (Å)");  ax1.set_ylabel("z (Å)");  ax1.set_title("x–z projection")
    ax2.set_xlabel("y (Å)");  ax2.set_ylabel("z (Å)");  ax2.set_title("y–z projection")
    fig.legend(handles=legend_elements, bbox_to_anchor=(1.18, 0.5), loc='center right')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()



def plot_oxygen_regions(snapshot, assignments, filepath: str):
    oxy = Atoms([a for a in snapshot if a.symbol == "O"])
    pos = oxy.get_positions()

    regions = sorted(set(r for _, _, r in assignments))
    cmap    = plt.cm.tab20
    region_colors = {region: cmap(i % 20) for i, region in enumerate(regions)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    legend_elements = []

    for region in regions:
        idxs = [idx for idx, _, r in assignments if r == region]
        xyz  = pos[idxs]
        color = region_colors[region]
        ax1.scatter(xyz[:, 0], xyz[:, 2], color=color, s=10)
        ax2.scatter(xyz[:, 1], xyz[:, 2], color=color, s=10)
        legend_elements.append(
            Line2D([0], [0], marker='o', linestyle='', color=color,
                   label=f"{region} ({len(idxs)})")
        )

    ax1.set_xlabel("x (Å)");  ax1.set_ylabel("z (Å)");  ax1.set_title("x–z projection")
    ax2.set_xlabel("y (Å)");  ax2.set_ylabel("z (Å)");  ax2.set_title("y–z projection")
    fig.legend(handles=legend_elements, bbox_to_anchor=(1.22, 0.5), loc='center right')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()



