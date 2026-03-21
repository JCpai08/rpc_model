"""
Coordinate transformations:

* WGS84 geodetic (lon, lat, h) ↔ ECEF XYZ
* J2000 ECI ↔ ECEF via a 3-angle Z-X-Z chain
* Quaternion → rotation matrix

Reference conventions
---------------------
- The J2000 frame is an Earth-centred inertial (ECI) frame whose x-axis points
  toward the mean vernal equinox at J2000.0 and z-axis toward the Celestial
  Intermediate Pole.
- ECEF (ITRF) rotates with the Earth; its x-axis points toward the Greenwich
  meridian.
- The transformation J2000 → ECEF uses the project ZXZ angle-chain model.
- Quaternions follow the scalar-first convention: q = [qw, qx, qy, qz].
  The rotation matrix R satisfies  r_target = R · r_source.
"""

import numpy as np
from datetime import datetime
from .constants import WGS84_A, WGS84_B, WGS84_E2


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
# J2000 (ECI) ↔ ECEF  (ZXZ angle-chain model)
# ---------------------------------------------------------------------------

def rot_z(angle):
    """Active rotation matrix around the Z-axis by *angle* [rad].

    r' = rot_z(θ) @ r  rotates a vector by +θ around Z (right-hand rule).
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def rot_x(angle):
    """Active rotation matrix around the X-axis by *angle* [rad]."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s,  c]])


def datetime_to_julian_params(imaging_time):
    """Convert imaging time to Julian parameters (JD, T, d).

    Parameters
    ----------
    imaging_time : str or datetime
        UTC timestamp. Supported string formats include:
        - ``YYYY MM DD HH:MM:SS(.ffffff)``
        - ``YYYY-MM-DD HH:MM:SS(.ffffff)``
        - ``YYYY/MM/DD HH:MM:SS(.ffffff)``

    Returns
    -------
    tuple[float, float, float]
        ``(jd, T, d)`` where:
        - ``jd``: Julian day
        - ``T``: Julian century from J2000.0
        - ``d``: Julian day offset from J2000.0
    """
    if isinstance(imaging_time, datetime):
        dt = imaging_time
    else:
        text = str(imaging_time).strip()
        text = text.replace("T", " ")
        date_part, *time_part = text.split()
        if "-" in date_part:
            y, m, d0 = [int(x) for x in date_part.split("-")]
        elif "/" in date_part:
            y, m, d0 = [int(x) for x in date_part.split("/")]
        else:
            if len(text.split()) < 4:
                raise ValueError(f"Unsupported imaging_time format: {imaging_time}")
            parts = text.split()
            y, m, d0 = [int(x) for x in parts[:3]]
            time_part = [parts[3]]

        if not time_part:
            hour = minute = 0
            second = 0.0
        else:
            hms = time_part[0].split(":")
            if len(hms) != 3:
                raise ValueError(f"Unsupported imaging_time format: {imaging_time}")
            hour = int(hms[0])
            minute = int(hms[1])
            second = float(hms[2])

        second_int = int(np.floor(second))
        microsecond = int(round((second - second_int) * 1_000_000))
        if microsecond == 1_000_000:
            second_int += 1
            microsecond = 0

        dt = datetime(y, m, d0, hour, minute, second_int, microsecond)

    y, m = dt.year, dt.month
    day_fraction = (
        dt.day
        + (dt.hour + (dt.minute + (dt.second + dt.microsecond / 1_000_000.0) / 60.0) / 60.0) / 24.0
    )

    if m <= 2:
        y -= 1
        m += 12

    a = int(np.floor(y / 100.0))
    b = 2 - a + int(np.floor(a / 4.0))
    jd = (
        int(np.floor(365.25 * (y + 4716)))
        + int(np.floor(30.6001 * (m + 1)))
        + day_fraction
        + b
        - 1524.5
    )

    d = jd - 2451545.0
    T = d / 36525.0
    return jd, T, d


def j2000_to_ecef_matrix(julian_day_offset, julian_century=None):
    """3 × 3 rotation matrix from J2000 to ECEF using the ZXZ model.

    This implements ``plan/coord_transform_J2000_ECES.md`` directly:

      alpha_0 = 0.00 - 0.641 T
      delta_0 = 90.00 - 0.557 T
      W  = 190.147 + 360.9856235 d

      R_j2e = M_W · M_(90° - delta_0) · M_(alpha_0 + 90°)

    Parameters
    ----------
    julian_day_offset : float
        Day offset ``d`` from J2000 epoch.
    julian_century : float or None
        Julian centuries ``T`` from J2000. If None, ``T = d / 36525``.

    Time type note
    --------------
    In strict astronomy workflows, ``d``/``T`` are typically based on UT1.
    If current data time tags are UTC/GPS/other, this mapping should be
    adjusted later once the exact time type is confirmed.
    """
    d = float(julian_day_offset)
    T = d / 36525.0 if julian_century is None else float(julian_century)

    alpha0_deg = 0.00 - 0.641 * T
    delta0_deg = 90.00 - 0.557 * T
    w_deg = 190.147 + 360.9856235 * d

    a1 = np.deg2rad(alpha0_deg + 90.0)
    a2 = np.deg2rad(90.0 - delta0_deg)
    a3 = np.deg2rad(w_deg)
    return rot_z(a3) @ rot_x(a2) @ rot_z(a1)


def ecef_to_j2000_matrix(julian_day_offset, julian_century=None):
    """3 × 3 rotation matrix from ECEF to J2000 (transpose of j2000_to_ecef_matrix)."""
    return j2000_to_ecef_matrix(julian_day_offset=julian_day_offset,
                                julian_century=julian_century).T


def j2000_to_ecef(vector_j2000, julian_day_offset, julian_century=None):
    """Transform vector(s) from J2000 to ECEF."""
    vec = np.asarray(vector_j2000, dtype=float)
    r_j2e = j2000_to_ecef_matrix(julian_day_offset=julian_day_offset,
                                 julian_century=julian_century)
    return np.einsum("ij,...j->...i", r_j2e, vec)


def ecef_to_j2000(vector_ecef, julian_day_offset, julian_century=None):
    """Transform vector(s) from ECEF to J2000."""
    vec = np.asarray(vector_ecef, dtype=float)
    r_e2j = ecef_to_j2000_matrix(julian_day_offset=julian_day_offset,
                                 julian_century=julian_century)
    return np.einsum("ij,...j->...i", r_e2j, vec)


def attitude_j2000_to_ecef_quaternion(q_body_to_j2000, julian_day_offset=None,
                                      julian_century=None, imaging_time=None):
    """Convert body→J2000 quaternion to body→ECEF quaternion.

    Time notes
    ----------
    Same as :func:`j2000_to_ecef_matrix`: current project may still need to
    refine whether time tags are UTC/UT1/GPS and the exact epoch mapping.
    """
    if imaging_time is not None:
        _, inferred_T, inferred_d = datetime_to_julian_params(imaging_time)
        if julian_day_offset is None:
            julian_day_offset = inferred_d
        if julian_century is None:
            julian_century = inferred_T

    if julian_day_offset is None:
        raise ValueError("Either julian_day_offset or imaging_time must be provided.")

    r_b2j = quaternion_to_rotation_matrix(q_body_to_j2000)
    r_j2e = j2000_to_ecef_matrix(julian_day_offset=julian_day_offset,
                                 julian_century=julian_century)
    r_b2e = r_j2e @ r_b2j
    return rotation_matrix_to_quaternion(r_b2e)


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
