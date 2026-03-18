"""
Generate synthetic input data for the RPC model pipeline.

Data produced
-------------
data/orbit.csv       – discrete ECEF orbit samples (t, X, Y, Z, Vx, Vy, Vz)
data/attitude.csv    – discrete J2000 attitude quaternions (t, qw, qx, qy, qz)
data/camera_angles.csv – cross-track pointing angles per column (col, angle_rad)
data/scan_times.csv  – imaging time per scan line (row, time_sec)

Scenario
--------
* Sun-synchronous orbit, ~500 km altitude, inclination 97.4°.
* Image strip: 1 000 scan lines × 500 columns.
* Ground-sample distance: ~5 m (along-track); FOV ±3° cross-track.
* Orbit epoch: t = 0 s aligned with the start of the imaging window.
  The J2000 time reference is set to T_J2000 = 788_918_400 s after J2000
  (approximately 2025-01-01 00:00 UTC) for a realistic GMST calculation.
"""

import os
import sys
import numpy as np

# Allow running as a script from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rpc_model.constants import GM, OMEGA_E, WGS84_A
from rpc_model.coord_transform import (
    j2000_to_ecef_matrix,
    rotation_matrix_to_quaternion,
)

# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------

# J2000 seconds at the imaging epoch  (≈ 2025-01-01 00:00 UTC)
T_J2000_EPOCH = 788_918_400.0

# Orbit
ALT = 500_000.0          # [m]  orbit altitude
INCL = np.deg2rad(97.4)  # [rad] sun-synchronous inclination
RAAN = np.deg2rad(165.0) # [rad] right ascension of ascending node

# Image dimensions
N_ROWS = 1000
N_COLS = 500

# Camera FOV
FOV_HALF_DEG = 3.0       # cross-track half-angle [°]

# Orbit sampling for data files
ORBIT_MARGIN_S = 60.0    # seconds before/after imaging to include in orbit file
ORBIT_STEP_S   = 10.0    # [s] sampling interval for orbit data
ATTITUDE_STEP_S = 1.0    # [s] sampling interval for attitude data


# ---------------------------------------------------------------------------
# Helper: circular orbit in J2000
# ---------------------------------------------------------------------------

def _orbit_state_j2000(t_rel, a, incl, raan, u0=0.0):
    """Position and velocity in J2000 for a circular orbit.

    Parameters
    ----------
    t_rel : float or ndarray
        Time relative to imaging epoch [s].
    a : float
        Semi-major axis [m].
    incl, raan, u0 : float
        Inclination [rad], RAAN [rad], initial argument of latitude [rad].

    Returns
    -------
    pos : ndarray (..., 3)  ECEF position [m]
    vel : ndarray (..., 3)  ECEF velocity [m/s]
    """
    t_rel = np.atleast_1d(np.asarray(t_rel, dtype=float))
    n = np.sqrt(GM / a**3)            # mean motion [rad/s]
    u = u0 + n * t_rel                 # argument of latitude

    cos_u, sin_u = np.cos(u), np.sin(u)
    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(incl), np.sin(incl)

    # Position in J2000
    r_x = a * (cos_O * cos_u - sin_O * sin_u * cos_i)
    r_y = a * (sin_O * cos_u + cos_O * sin_u * cos_i)
    r_z = a * sin_u * sin_i
    r_j2k = np.stack([r_x, r_y, r_z], axis=-1)

    # Velocity in J2000
    v = a * n
    v_x = v * (-cos_O * sin_u - sin_O * cos_u * cos_i)
    v_y = v * (-sin_O * sin_u + cos_O * cos_u * cos_i)
    v_z = v * cos_u * sin_i
    v_j2k = np.stack([v_x, v_y, v_z], axis=-1)

    return r_j2k, v_j2k


def _nadir_attitude_j2000(r_j2k, v_j2k):
    """Body-to-J2000 rotation matrix for a nadir-pointing attitude.

    Body frame:  X_b = along-track, Y_b = cross-track (right), Z_b = nadir.
    """
    r_j2k = np.asarray(r_j2k, dtype=float)
    v_j2k = np.asarray(v_j2k, dtype=float)

    X_b = v_j2k / np.linalg.norm(v_j2k)         # along-track
    Z_b = -r_j2k / np.linalg.norm(r_j2k)         # nadir
    Y_b = np.cross(Z_b, X_b)
    Y_b /= np.linalg.norm(Y_b)

    # Columns are the body axes expressed in J2000
    R = np.column_stack([X_b, Y_b, Z_b])
    return R


def _add_attitude_noise(q, sigma_deg=0.02):
    """Add a small random rotation to a quaternion."""
    sigma = np.deg2rad(sigma_deg)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.normal(0.0, sigma)
    dq = np.array([
        np.cos(angle / 2),
        *(np.sin(angle / 2) * axis),
    ])
    # Quaternion multiplication: q ⊗ dq
    qw, qx, qy, qz = q
    dw, dx, dy, dz = dq
    qout = np.array([
        qw*dw - qx*dx - qy*dy - qz*dz,
        qw*dx + qx*dw + qy*dz - qz*dy,
        qw*dy - qx*dz + qy*dw + qz*dx,
        qw*dz + qx*dy - qy*dx + qz*dw,
    ])
    return qout / np.linalg.norm(qout)


# ---------------------------------------------------------------------------
# Main generation routine
# ---------------------------------------------------------------------------

def generate(out_dir=None, seed=42):
    """Generate all data files and return their paths.

    Parameters
    ----------
    out_dir : str or None
        Directory to write files into.  Defaults to the directory that
        contains this script.
    seed : int
        Random seed for reproducible attitude noise.

    Returns
    -------
    dict of str → str
        Mapping ``{name: filepath}`` for the four data files.
    """
    np.random.seed(seed)
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    a = WGS84_A + ALT

    # ------------------------------------------------------------------
    # Scan times (1 row per scan line)
    # ------------------------------------------------------------------
    v_orbital = np.sqrt(GM / a)
    gsd_along = 5.0   # [m] desired along-track GSD
    dt_line = gsd_along / v_orbital  # [s] per line
    scan_times = np.arange(N_ROWS) * dt_line   # relative to imaging start

    # ------------------------------------------------------------------
    # Orbit file (dense enough: every ORBIT_STEP_S seconds)
    # ------------------------------------------------------------------
    t_lo = -ORBIT_MARGIN_S
    t_hi = scan_times[-1] + ORBIT_MARGIN_S
    orbit_times = np.arange(t_lo, t_hi + ORBIT_STEP_S, ORBIT_STEP_S)
    t_j2000_orbit = T_J2000_EPOCH + orbit_times

    r_j2k, v_j2k = _orbit_state_j2000(orbit_times, a, INCL, RAAN)

    # Convert J2000 positions to ECEF (WGS84) for the orbit file
    r_ecef = np.array([
        j2000_to_ecef_matrix(t) @ r_j2k[i]
        for i, t in enumerate(t_j2000_orbit)
    ])
    v_ecef = np.array([
        j2000_to_ecef_matrix(t) @ v_j2k[i]
        for i, t in enumerate(t_j2000_orbit)
    ])

    orbit_path = os.path.join(out_dir, "orbit.csv")
    header_o = "time_s,X_m,Y_m,Z_m,Vx_ms,Vy_ms,Vz_ms"
    orbit_data = np.column_stack([orbit_times, r_ecef, v_ecef])
    np.savetxt(orbit_path, orbit_data, delimiter=",", header=header_o, comments="")

    # ------------------------------------------------------------------
    # Attitude file (quaternion body → J2000, with small noise)
    # ------------------------------------------------------------------
    att_times = np.arange(t_lo, t_hi + ATTITUDE_STEP_S, ATTITUDE_STEP_S)
    r_att, v_att = _orbit_state_j2000(att_times, a, INCL, RAAN)

    quats = []
    for i in range(len(att_times)):
        R = _nadir_attitude_j2000(r_att[i], v_att[i])
        q = rotation_matrix_to_quaternion(R)
        q = _add_attitude_noise(q, sigma_deg=0.02)
        quats.append(q)
    quats = np.array(quats)

    att_path = os.path.join(out_dir, "attitude.csv")
    header_a = "time_s,qw,qx,qy,qz"
    att_data = np.column_stack([att_times, quats])
    np.savetxt(att_path, att_data, delimiter=",", header=header_a, comments="")

    # ------------------------------------------------------------------
    # Camera pointing angle table
    # ------------------------------------------------------------------
    fov_half = np.deg2rad(FOV_HALF_DEG)
    col_indices = np.arange(N_COLS)
    pointing_angles = np.linspace(-fov_half, fov_half, N_COLS)

    cam_path = os.path.join(out_dir, "camera_angles.csv")
    header_c = "col_index,angle_rad"
    cam_data = np.column_stack([col_indices, pointing_angles])
    np.savetxt(cam_path, cam_data, delimiter=",", header=header_c, comments="")

    # ------------------------------------------------------------------
    # Scan-time file
    # ------------------------------------------------------------------
    row_indices = np.arange(N_ROWS)
    scan_path = os.path.join(out_dir, "scan_times.csv")
    header_s = "row_index,time_s"
    scan_data = np.column_stack([row_indices, scan_times])
    np.savetxt(scan_path, scan_data, delimiter=",", header=header_s, comments="")

    print(f"[simulate_data] Generated orbit file    : {orbit_path}  ({len(orbit_times)} rows)")
    print(f"[simulate_data] Generated attitude file : {att_path}  ({len(att_times)} rows)")
    print(f"[simulate_data] Generated camera file   : {cam_path}  ({N_COLS} columns)")
    print(f"[simulate_data] Generated scan-time file: {scan_path}  ({N_ROWS} scan lines)")
    print(f"[simulate_data] dt_line = {dt_line*1e3:.4f} ms  "
          f"(imaging duration = {scan_times[-1]:.3f} s)")

    return dict(
        orbit=orbit_path,
        attitude=att_path,
        camera=cam_path,
        scan_times=scan_path,
        T_J2000_EPOCH=T_J2000_EPOCH,
        N_ROWS=N_ROWS,
        N_COLS=N_COLS,
    )


if __name__ == "__main__":
    generate()
