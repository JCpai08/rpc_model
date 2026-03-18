"""
End-to-end pipeline: terrain-independent RPC model construction.

Steps
-----
1. Generate (or load) synthetic input data.
2. Load orbit, attitude, camera pointing, and scan-time data.
3. Instantiate the strict pushbroom imaging model.
4. Build a 3-D ground control grid via backward projection.
5. Fit RPC model coefficients (least squares).
6. Assess accuracy against both the training grid and a separate check set.
7. Print a summary and (optionally) save the model coefficients.

Usage
-----
    python main.py [--no-plot]
"""

import os
import sys
import argparse
import json
import numpy as np

# Ensure the package is importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rpc_model import (
    OrbitInterpolator,
    AttitudeInterpolator,
    PushbroomImagingModel,
    build_control_grid,
    RPCSolver,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_orbit(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    times     = d[:, 0]
    positions = d[:, 1:4]
    velocities = d[:, 4:7]
    return times, positions, velocities


def load_attitude(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    times = d[:, 0]
    quats = d[:, 1:5]
    return times, quats


def load_camera_angles(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 1]   # angle_rad for each column


def load_scan_times(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 1]   # time_s for each row


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(no_plot=False):
    print("=" * 65)
    print("  Terrain-independent RPC model construction pipeline")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 0: ensure synthetic data exists
    # ------------------------------------------------------------------
    orbit_path  = os.path.join(DATA_DIR, "orbit.csv")
    att_path    = os.path.join(DATA_DIR, "attitude.csv")
    cam_path    = os.path.join(DATA_DIR, "camera_angles.csv")
    scan_path   = os.path.join(DATA_DIR, "scan_times.csv")

    if not all(os.path.exists(p) for p in [orbit_path, att_path, cam_path, scan_path]):
        print("\n[Step 0] Generating synthetic input data …")
        from data.simulate_data import generate
        meta = generate(out_dir=DATA_DIR)
        T_J2000_EPOCH = meta["T_J2000_EPOCH"]
    else:
        print("\n[Step 0] Data files already exist – skipping generation.")
        # Read epoch from orbit file header; we use a fixed known value
        T_J2000_EPOCH = 788_918_400.0

    # ------------------------------------------------------------------
    # Step 1: load data
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading input data …")
    orb_times, orb_pos, orb_vel = load_orbit(orbit_path)
    att_times, att_quats        = load_attitude(att_path)
    pointing_angles             = load_camera_angles(cam_path)
    scan_times                  = load_scan_times(scan_path)

    print(f"  Orbit    : {len(orb_times)} samples,  "
          f"t ∈ [{orb_times[0]:.1f}, {orb_times[-1]:.1f}] s")
    print(f"  Attitude : {len(att_times)} samples,  "
          f"t ∈ [{att_times[0]:.1f}, {att_times[-1]:.1f}] s")
    print(f"  Camera   : {len(pointing_angles)} columns, "
          f"FOV [{np.rad2deg(pointing_angles[0]):.2f}°, "
          f"{np.rad2deg(pointing_angles[-1]):.2f}°]")
    print(f"  Scan lines: {len(scan_times)}, "
          f"t ∈ [{scan_times[0]:.5f}, {scan_times[-1]:.5f}] s")

    # ------------------------------------------------------------------
    # Step 2: build interpolators (orbit + attitude)
    # ------------------------------------------------------------------
    print("\n[Step 2] Building orbit and attitude interpolators …")
    orbit_interp    = OrbitInterpolator(orb_times, orb_pos, orb_vel, order=8)
    attitude_interp = AttitudeInterpolator(att_times, att_quats)

    # ------------------------------------------------------------------
    # Step 3: instantiate strict imaging model
    # ------------------------------------------------------------------
    print("\n[Step 3] Instantiating strict pushbroom imaging model …")
    model = PushbroomImagingModel(
        scan_times=scan_times,
        orbit_interp=orbit_interp,
        attitude_interp=attitude_interp,
        pointing_angles=pointing_angles,
        j2000_offset=T_J2000_EPOCH,   # scan_times are relative; add epoch for GMST
    )

    # Quick sanity check: backward-project corner pixels
    corners = [(0, 0), (0, 499), (999, 0), (999, 499), (499, 249)]
    print("  Corner pixel ↔ ground check:")
    for r, c in corners:
        lon, lat, h = model.backward_project(r, c, h=0.0)
        print(f"    pixel ({r:4d},{c:3d}) → lon={lon:.4f}°  lat={lat:.4f}°  h={h:.1f} m")

    # ------------------------------------------------------------------
    # Step 4: build 3-D ground control grid
    # ------------------------------------------------------------------
    print("\n[Step 4] Building terrain-independent ground control grid …")
    heights = [-500.0, 0.0, 500.0, 1000.0, 2000.0]
    lons, lats, hs, rows_ref, cols_ref = build_control_grid(
        model,
        heights=heights,
        n_row_levels=10,
        n_col_levels=10,
    )
    print(f"  Control grid: {len(lons)} points  "
          f"(height levels: {heights})")
    print(f"  Ground extent:  lon [{lons.min():.4f}°, {lons.max():.4f}°]"
          f"  lat [{lats.min():.4f}°, {lats.max():.4f}°]")

    # ------------------------------------------------------------------
    # Step 5: fit RPC model
    # ------------------------------------------------------------------
    print("\n[Step 5] Solving RPC model coefficients …")
    solver = RPCSolver(lambda_reg=1e-4)
    rpc = solver.fit(lons, lats, hs, rows_ref, cols_ref)
    print("  RPC model fitted successfully.")
    print(f"  Row model  – Num_L[0] = {rpc.a_L[0]:.6f}  Den_L[0] = 1.0 (fixed)")
    print(f"  Col model  – Num_S[0] = {rpc.a_S[0]:.6f}  Den_S[0] = 1.0 (fixed)")

    # ------------------------------------------------------------------
    # Step 6: accuracy assessment
    # ------------------------------------------------------------------
    print("\n[Step 6] Accuracy assessment …")

    # (a) Training-set residuals
    acc_train = rpc.assess_accuracy(lons, lats, hs, rows_ref, cols_ref)
    print(f"\n  Training set ({len(lons)} points):")
    print(f"    Row RMSE = {acc_train['row_rmse']:.4f} px   Max = {acc_train['row_max']:.4f} px")
    print(f"    Col RMSE = {acc_train['col_rmse']:.4f} px   Max = {acc_train['col_max']:.4f} px")

    # (b) Independent check grid (denser, different heights)
    check_heights = [-250.0, 250.0, 750.0, 1500.0]
    lons_c, lats_c, hs_c, rows_c, cols_c = build_control_grid(
        model,
        heights=check_heights,
        n_row_levels=15,
        n_col_levels=15,
    )
    acc_check = rpc.assess_accuracy(lons_c, lats_c, hs_c, rows_c, cols_c)
    print(f"\n  Check set ({len(lons_c)} points, independent heights):")
    print(f"    Row RMSE = {acc_check['row_rmse']:.4f} px   Max = {acc_check['row_max']:.4f} px")
    print(f"    Col RMSE = {acc_check['col_rmse']:.4f} px   Max = {acc_check['col_max']:.4f} px")

    # ------------------------------------------------------------------
    # Save model coefficients
    # ------------------------------------------------------------------
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rpc_coefficients.json")
    with open(out_path, "w") as f:
        json.dump(rpc.to_dict(), f, indent=2)
    print(f"\n[Result] RPC coefficients saved to: {out_path}")

    # ------------------------------------------------------------------
    # Optional: residual scatter plot
    # ------------------------------------------------------------------
    if not no_plot:
        _plot_residuals(acc_check, lons_c, lats_c)

    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    print("=" * 65)
    return rpc, acc_check


def _plot_residuals(acc, lons, lats):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available – skipping plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("RPC Model Residuals (check set)", fontsize=13)

    sc0 = axes[0].scatter(lons, lats, c=acc["row_res"], cmap="RdBu", s=20)
    plt.colorbar(sc0, ax=axes[0], label="Row residual [px]")
    axes[0].set_xlabel("Longitude [°]")
    axes[0].set_ylabel("Latitude [°]")
    axes[0].set_title("Row residuals")

    sc1 = axes[1].scatter(lons, lats, c=acc["col_res"], cmap="RdBu", s=20)
    plt.colorbar(sc1, ax=axes[1], label="Col residual [px]")
    axes[1].set_xlabel("Longitude [°]")
    axes[1].set_ylabel("Latitude [°]")
    axes[1].set_title("Column residuals")

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "residuals.png")
    plt.savefig(plot_path, dpi=120)
    print(f"  Residual plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPC model construction pipeline")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the residual scatter plot")
    args = parser.parse_args()
    main(no_plot=args.no_plot)
