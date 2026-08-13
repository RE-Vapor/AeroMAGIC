#!/usr/bin/env python3
"""Render the requested start3 spatial trajectory from a MAGICIAN LMDB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lmdb
import matplotlib.pyplot as plt
import numpy as np

from macarons.utility.magician_utils import load_from_lmdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    environment = lmdb.open(str(args.lmdb), readonly=True, lock=False)
    try:
        trajectory = load_from_lmdb(environment, "huge_1_office/3")
    finally:
        environment.close()
    if trajectory is None:
        raise KeyError("LMDB has no huge_1_office/3 trajectory")
    xyz = np.asarray(trajectory["X_cam_history"])
    coverage = np.asarray(trajectory["coverage"], dtype=np.float64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(xyz[:, 0], xyz[:, 2], xyz[:, 1], "-o", markersize=2, linewidth=1)
    axis.scatter(*xyz[0, [0, 2, 1]], color="green", s=55, label="start3")
    axis.scatter(*xyz[-1, [0, 2, 1]], color="red", s=55, label="end")
    axis.set_xlabel("MAGIC X (m)")
    axis.set_ylabel("MAGIC Z (m)")
    axis.set_zlabel("MAGIC Y (m)")
    axis.set_title(f"{args.label}: huge_1_office start3")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    summary = {
        "label": args.label,
        "lmdb_key": "huge_1_office/3",
        "observations": int(len(xyz)),
        "start_xyz_m": xyz[0].tolist(),
        "end_xyz_m": xyz[-1].tolist(),
        "final_coverage": float(coverage[-1]),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
