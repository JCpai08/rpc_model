"""
Coordinate transformations:

* WGS84 geodetic (lon, lat, h) ↔ ECEF XYZ
* J2000 ECI ↔ ECEF via Greenwich Mean Sidereal Time (simplified rotation model)
* Quaternion → rotation matrix

Reference conventions
---------------------
- The J2000 frame is an Earth-centred inertial (ECI) frame whose x-axis points
  toward the mean vernal equinox at J2000.0 and z-axis toward the Celestial
  Intermediate Pole.
- ECEF (ITRF) rotates with the Earth; its x-axis points toward the Greenwich
  meridian.
- The transformation J2000 → ECEF uses only Earth rotation (GMST).  For this
  exercise precession, nutation, and polar motion are omitted.
- Quaternions follow the scalar-first convention: q = [qw, qx, qy, qz].
  The rotation matrix R satisfies  r_target = R · r_source.
"""

import numpy as np
from .constants import WGS84_A, WGS84_B, WGS84_E2, OMEGA_E, GMST_J2000


# ---------------------------------------------------------------------------
# Geodetic ↔ ECEF
# ---------------------------------------------------------------------------

def geodetic_to_ecef(lon_deg, lat_deg, h):
    """Convert geodetic (lon [°], lat [°], h [m]) to ECEF XYZ [m].

    Parameters
    ----------
    lon_deg, lat_deg : float or array-like
        Geodetic longitude and latitude in degrees.
    h : float or array-like
        Ellipsoidal height in metres.

    Returns
    -------
    numpy.ndarray, shape (..., 3)
        ECEF position vector(s) in metres.
    """
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    X = (N + h) * cos_lat * np.cos(lon)
    Y = (N + h) * cos_lat * np.sin(lon)
    Z = (N * (1.0 - WGS84_E2) + h) * sin_lat
    return np.stack([X, Y, Z], axis=-1)


def ecef_to_geodetic(X, Y, Z):
    """Convert ECEF XYZ [m] to geodetic (lon [°], lat [°], h [m]).

    Uses Bowring's iterative method (converges in < 5 iterations for
    altitudes up to 10 000 km).

    Parameters
    ----------
    X, Y, Z : float or array-like
        ECEF coordinates in metres.

    Returns
    -------
    lon_deg, lat_deg, h : arrays
        Geodetic longitude (°), latitude (°), and ellipsoidal height (m).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    lon = np.arctan2(Y, X)
    p = np.hypot(X, Y)

    # Initial latitude estimate (Bowring)
    lat = np.arctan2(Z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        sin_lat = np.sin(lat)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
        lat_new = np.arctan2(Z + WGS84_E2 * N * sin_lat, p)
        if np.all(np.abs(lat_new - lat) < 1e-12):
            break
        lat = lat_new

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    # Choose numerically stable formula for height
    with np.errstate(invalid="ignore"):
        h_eq = p / np.where(np.abs(cos_lat) > 1e-10, cos_lat, 1.0) - N
        h_po = Z / np.where(np.abs(sin_lat) > 1e-10, sin_lat, 1.0) - N * (1.0 - WGS84_E2)
    h = np.where(np.abs(lat) <= np.pi / 4.0, h_eq, h_po)

    return np.rad2deg(lon), np.rad2deg(lat), h


# ---------------------------------------------------------------------------
# J2000 (ECI) ↔ ECEF  (Earth-rotation only, simplified GMST model)
# ---------------------------------------------------------------------------

def _gmst(t_sec):
    """Greenwich Mean Sidereal Time [rad] at *t_sec* seconds after J2000 epoch."""
    return GMST_J2000 + OMEGA_E * t_sec


def rot_z(angle):
    """Active rotation matrix around the Z-axis by *angle* [rad].

    r' = rot_z(θ) @ r  rotates a vector by +θ around Z (right-hand rule).
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def j2000_to_ecef_matrix(t_sec):
    """3 × 3 rotation matrix from J2000 to ECEF at *t_sec* seconds after J2000.

    ECEF = R_z(−GMST) · ECI, so we rotate the ECI frame clockwise by GMST.
    """
    theta = _gmst(t_sec)
    # R_z(-theta) rotates ECI vectors into ECEF
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,  s, 0.0],
                     [-s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def ecef_to_j2000_matrix(t_sec):
    """3 × 3 rotation matrix from ECEF to J2000 (transpose of j2000_to_ecef_matrix)."""
    return j2000_to_ecef_matrix(t_sec).T


def j2000_to_ecef(pos_j2000, t_sec):
    """Transform a position vector from J2000 to ECEF.

    Parameters
    ----------
    pos_j2000 : array-like, shape (3,) or (N, 3)
    t_sec : float or array of shape (N,)
        Seconds after J2000 epoch (2000-01-01 11:58:55.816 UTC).

    Returns
    -------
    numpy.ndarray, shape (3,) or (N, 3)
    """
    pos = np.asarray(pos_j2000, dtype=float)
    scalar = pos.ndim == 1
    if scalar:
        return j2000_to_ecef_matrix(t_sec) @ pos
    # Vectorised over multiple positions / times
    t_sec = np.asarray(t_sec, dtype=float)
    result = np.empty_like(pos)
    for i in range(len(pos)):
        result[i] = j2000_to_ecef_matrix(t_sec[i]) @ pos[i]
    return result


def ecef_to_j2000(pos_ecef, t_sec):
    """Transform a position vector from ECEF to J2000."""
    pos = np.asarray(pos_ecef, dtype=float)
    scalar = pos.ndim == 1
    if scalar:
        return ecef_to_j2000_matrix(t_sec) @ pos
    t_sec = np.asarray(t_sec, dtype=float)
    result = np.empty_like(pos)
    for i in range(len(pos)):
        result[i] = ecef_to_j2000_matrix(t_sec[i]) @ pos[i]
    return result


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(q):
    """Convert a unit quaternion to a 3 × 3 rotation matrix.

    Convention: q = [qw, qx, qy, qz] (scalar first).
    The matrix R satisfies  r_out = R · r_in  (active rotation).

    Parameters
    ----------
    q : array-like, shape (4,)

    Returns
    -------
    numpy.ndarray, shape (3, 3)
    """
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    R = np.array([
        [1.0 - 2.0*(qy**2 + qz**2),  2.0*(qx*qy - qw*qz),        2.0*(qx*qz + qw*qy)],
        [2.0*(qx*qy + qw*qz),         1.0 - 2.0*(qx**2 + qz**2),  2.0*(qy*qz - qw*qx)],
        [2.0*(qx*qz - qw*qy),         2.0*(qy*qz + qw*qx),         1.0 - 2.0*(qx**2 + qy**2)],
    ])
    return R


def rotation_matrix_to_quaternion(R):
    """Convert a rotation matrix to a unit quaternion [qw, qx, qy, qz].

    Uses Shepperd's method for numerical stability.
    """
    R = np.asarray(R, dtype=float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)
