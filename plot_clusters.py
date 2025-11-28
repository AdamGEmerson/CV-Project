#!/usr/bin/env python3
"""
Plot clustered frames using the PCA coordinates stored in
data/landmarks/all_segments_clustered_with_xy.json.

Usage:
    python plot_clusters.py [path/to/all_segments_clustered_with_xy.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd

DEFAULT_JSON = Path("data/landmarks/all_segments_clustered_with_xy.json")


def load_cluster_dataframe(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r") as fh:
        payload = json.load(fh)

    frames = payload.get("clustered_frames", [])
    records = []
    for frame in frames:
        if all(coord in frame for coord in ("x", "y", "z")):
            records.append(
                {
                    "segment": frame.get("segment"),
                    "frame": frame["frame"],
                    "cluster": frame["cluster"],
                    "x": frame.get("x"),
                    "y": frame.get("y"),
                    "z": frame.get("z"),
                }
            )

    return pd.DataFrame(records)


def build_color_map(unique_clusters: list[int]) -> dict[int, str]:
    cmap = plt.get_cmap("tab10", len(unique_clusters))
    colors = {}
    for idx, cluster_id in enumerate(unique_clusters):
        if cluster_id == -1:
            colors[cluster_id] = "#d1d5db"  # light gray for noise
        else:
            colors[cluster_id] = cmap(idx % cmap.N)
    return colors


def plot_clusters(df: pd.DataFrame) -> None:
    unique_clusters = sorted(df["cluster"].unique())
    cluster_colors = build_color_map(unique_clusters)
    df["color"] = df["cluster"].map(cluster_colors)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for cluster_id, group in df.groupby("cluster"):
        label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        ax.scatter(
            group["x"],
            group["y"],
            group["z"],
            s=25 if cluster_id == -1 else 40,
            c=group["color"],
            label=label,
            alpha=0.65 if cluster_id == -1 else 0.85,
            edgecolor="none",
        )

    ax.set_title("Cluster embedding scatter plot (3D PCA space)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.show()


def main(raw_args: list[str]) -> None:
    json_path = Path(raw_args[0]) if raw_args else DEFAULT_JSON
    if not json_path.exists():
        raise SystemExit(f"JSON file not found: {json_path}")

    df = load_cluster_dataframe(json_path)
    if df.empty:
        raise SystemExit("No frames with PCA coordinates found in the JSON file.")

    plot_clusters(df)


if __name__ == "__main__":
    main(sys.argv[1:])

