"""
End-to-end pipeline: terrain-independent RPC model construction.

Steps
-----
1. Load NAD raw data from config.json.
2. Parse orbit, attitude, camera pointing, and scan-time data.

Usage
-----
    # python main.py [--no-plot] [--nad-config path/to/config.json]
    python main.py [--nad-config path/to/config.json]
"""

import os
import sys
import argparse
import json
import numpy as np

# Ensure the package is importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rpc_model import (
    load_nad_bundle,
    OrbitData,
    AttitudeData,
    ImagingTimeData,
    CBRData,
)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_nad_raw_dataset(data_dir, nad_config_path=None):
    bundle = load_nad_bundle(data_dir=data_dir, config_path=nad_config_path)
    gps: OrbitData = bundle["gps"]
    att: AttitudeData = bundle["attitude"]
    imaging_time: ImagingTimeData = bundle["imaging_time"]
    nad_cbr: CBRData = bundle["nad_cbr"]

    return gps, att, imaging_time, nad_cbr


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(no_plot=False, nad_config_path=None):
    print("=" * 65)
    print("  Terrain-independent RPC model construction pipeline")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 0: resolve config path
    # ------------------------------------------------------------------
    default_nad_config_path = os.path.join(DATA_DIR, "config.json")
    resolved_nad_config_path = nad_config_path or default_nad_config_path
    if not os.path.exists(resolved_nad_config_path):
        raise FileNotFoundError(
            f"NAD config file not found: {resolved_nad_config_path}. "
            "Please provide --nad-config or create data/config.json."
        )

    # ------------------------------------------------------------------
    # Step 1: load data
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading input data …")
    print(f"  Source: NAD raw text files via config ({resolved_nad_config_path})")
    (
        gps,
        att,
        imaging_time,
        nad_cbr
    ) = load_nad_raw_dataset(DATA_DIR, nad_config_path=resolved_nad_config_path)

    print("\n[Load Summary]")
    gps_times, gps_positions, gps_velocities = gps.to_arrays()
    att_times, att_quaternions = att.to_arrays()

    print(f"  GPS samples        : {len(gps_times)}")
    if len(gps_times) > 0:
        print(f"    time range       : [{gps_times[0]:.6f}, {gps_times[-1]:.6f}]")
        print(f"    positions shape  : {gps_positions.shape}")
        print(f"    velocities shape : {gps_velocities.shape}")

    print(f"  Attitude samples   : {len(att_times)}")
    if len(att_times) > 0:
        print(f"    time range       : [{att_times[0]:.6f}, {att_times[-1]:.6f}]")
        print(f"    quaternions shape: {att_quaternions.shape}")

    print(f"  Imaging lines      : {len(imaging_time.rel_lines)}")
    if len(imaging_time.times) > 0:
        print(f"    time range       : [{imaging_time.times[0]:.6f}, {imaging_time.times[-1]:.6f}]")
        print(f"    delta_t range    : [{np.min(imaging_time.delta_times):.6f}, {np.max(imaging_time.delta_times):.6f}]")

    print(f"  CBR rows           : {len(nad_cbr.column_indices)}")
    if len(nad_cbr.column_indices) > 0:
        print(f"    declared count   : {nad_cbr.declared_count}")
        print(f"    angle_1 range    : [{np.min(nad_cbr.angle_1):.6f}, {np.max(nad_cbr.angle_1):.6f}]")
        print(f"    angle_2 range    : [{np.min(nad_cbr.angle_2):.6f}, {np.max(nad_cbr.angle_2):.6f}]")

    
    # test coordinate transformations(J2000 ECI ↔ ECEF)
    print("\nTest Coordinate transformations …")
    print(att.samples[0].date_time)  # sanity check: print first attitude sample's timestamp
    from rpc_model import j2000_to_ecef, ecef_to_j2000
    print(j2000_to_ecef(np.array([0, 0, 0]), 0))

    # ------------------------------------------------------------------
    # Step 2: build interpolators (orbit + attitude)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 3: instantiate strict imaging model
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 4: build 3-D ground control grid
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 5: fit RPC model
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 6: accuracy assessment
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Save model coefficients
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Optional: residual scatter plot
    # ------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPC model construction pipeline")
    # parser.add_argument("--no-plot", action="store_true",
    #                     help="Skip the residual scatter plot")
    parser.add_argument("--nad-config", default=None,
                        help="JSON config file for NAD raw file paths")
    args = parser.parse_args()
    # main(no_plot=args.no_plot, nad_config_path=args.nad_config)
    main(nad_config_path=args.nad_config)
