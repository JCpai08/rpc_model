"""
Strict pushbroom imaging model.

Physical model
--------------
* The satellite carries a linear-array (pushbroom) sensor.
* Each scan line *r* is acquired at time ``t_r = scan_times[r]``.
* The satellite's ECEF position at *t_r* is ``P_s(t_r)`` (from orbit data).
* The satellite's orientation at *t_r* is given by a quaternion in J2000;
  after composing with the J2000→ECEF rotation we obtain ``R_b2e(t_r)``:
  the body-to-ECEF rotation matrix.
* Body frame axes:
    - X_b : along-track  (flight direction)
    - Y_b : cross-track  (to the right)
    - Z_b : nadir        (toward Earth)
* Each pixel column *c* has a look direction in the body frame defined by
  a pointing angle table:
    ``u_body(c) = normalize([0, sin(α_c), cos(α_c)])``
  where ``α_c`` is the cross-track pointing angle (positive to the right).

Backward projection (image → ground)
--------------------------------------
Given (row r, col c, ellipsoidal height h):
  1. Get P_s and R_b2e at t_r.
  2. Compute look direction in ECEF: d = R_b2e · u_body(c).
  3. Intersect the ray (P_s, d) with the WGS84 ellipsoid offset by h.
  4. Convert the intersection point to geodetic coordinates.

Forward projection (ground → image)
--------------------------------------
Given (lon, lat, h):
  1. Convert to ECEF: P_ground.
  2. Find row r* such that the along-track component of (P_ground − P_s(t_r))
     in the body frame is zero (Newton iteration with bisection fallback).
  3. Derive column c* from the cross-track angle and the pointing table.
"""

import numpy as np
from .constants import WGS84_A, WGS84_B, WGS84_E2
from .coord_transform import geodetic_to_ecef, ecef_to_geodetic


# ---------------------------------------------------------------------------
# Ray–ellipsoid intersection
# ---------------------------------------------------------------------------

def _ray_ellipsoid_intersect(P_s, d, h):
    """Intersect ray ``P_s + t·d`` with the WGS84 ellipsoid at height *h* [m].

    Parameters
    ----------
    P_s : array-like, shape (3,)
        Ray origin (satellite ECEF position) in metres.
    d : array-like, shape (3,)
        Unit look direction in ECEF.
    h : float
        Constant ellipsoidal height of the target surface [m].

    Returns
    -------
    numpy.ndarray, shape (3,) or None
        ECEF intersection point, or *None* if no real intersection exists.
    """
    A = WGS84_A + h
    B = WGS84_B + h
    A2, B2 = A * A, B * B

    Px, Py, Pz = P_s
    dx, dy, dz = d

    aa = (dx * dx + dy * dy) / A2 + dz * dz / B2
    bb = 2.0 * ((Px * dx + Py * dy) / A2 + Pz * dz / B2)
    cc = (Px * Px + Py * Py) / A2 + Pz * Pz / B2 - 1.0

    disc = bb * bb - 4.0 * aa * cc
    if disc < 0.0:
        return None

    sqrt_disc = np.sqrt(disc)
    t1 = (-bb - sqrt_disc) / (2.0 * aa)
    t2 = (-bb + sqrt_disc) / (2.0 * aa)

    # Take the smallest positive root (forward intersection)
    if t1 > 0.0:
        t = t1
    elif t2 > 0.0:
        t = t2
    else:
        return None

    return np.asarray(P_s, dtype=float) + t * np.asarray(d, dtype=float)


# ---------------------------------------------------------------------------
# Main imaging model class
# ---------------------------------------------------------------------------

class PushbroomImagingModel:
    """Strict pushbroom imaging model for a satellite with J2000 attitude data.

    Parameters
    ----------
    scan_times : array-like, shape (N_rows,)
        Imaging time [s] for each scan line (same epoch reference as orbit /
        attitude data).
    orbit_interp : OrbitInterpolator
        Provides ECEF position and velocity at arbitrary times.
    attitude_interp : AttitudeInterpolator
        Provides body→J2000 quaternion at arbitrary times.
    pointing_angles : array-like, shape (N_cols,)
        Cross-track pointing angle [rad] for each pixel column.
        Positive angles point to the right (increasing Y_b).
    julian_day_base : float or None
        Day offset ``d0`` from J2000 corresponding to ``t=0`` of ``scan_times``.
        Actual day offset at time ``t`` is ``d = d0 + t / 86400``.

        Time type note:
        data timestamps may be UTC/UT1/GPS depending on source metadata.
        This parameter keeps epoch mapping explicit for later adjustment.
    """

    def __init__(self, scan_times, orbit_interp, attitude_interp,
                 pointing_angles, julian_day_base=0.0):
        self.scan_times = np.asarray(scan_times, dtype=float)
        self.orbit = orbit_interp
        self.attitude = attitude_interp
        self.pointing_angles = np.asarray(pointing_angles, dtype=float)
        self.julian_day_base = float(julian_day_base)

        self.n_rows = len(self.scan_times)
        self.n_cols = len(self.pointing_angles)
        self._row_idx = np.arange(self.n_rows, dtype=float)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _t_from_row(self, row):
        """Imaging time [s] for a (possibly fractional) row index."""
        return float(np.interp(row, self._row_idx, self.scan_times))

    def _sat_state(self, t):
        """Return (P_s [m], R_b2e [3×3]) at time *t* [s]."""
        P_s = self.orbit.get_position(np.array([t]))[0]
        d = self.julian_day_base + float(t) / 86400.0
        R_b2e = self.attitude.get_rotation_body2ecef(t, julian_day_offset=d)
        return P_s, R_b2e

    def _col_to_look_body(self, col):
        """Unit look vector in the body frame for a (fractional) column index."""
        alpha = float(np.interp(col,
                                np.arange(self.n_cols, dtype=float),
                                self.pointing_angles))
        u = np.array([0.0, np.sin(alpha), np.cos(alpha)])
        return u / np.linalg.norm(u)

    def _look_body_to_col(self, u_body):
        """Column index corresponding to a look direction in the body frame.

        Derived from the cross-track angle atan2(u_body[1], u_body[2]).
        """
        alpha = np.arctan2(u_body[1], u_body[2])
        col = float(np.interp(alpha,
                              self.pointing_angles,
                              np.arange(self.n_cols, dtype=float)))
        return col

    # ------------------------------------------------------------------
    # Backward projection: image → ground
    # ------------------------------------------------------------------

    def backward_project(self, row, col, h=0.0):
        """Project pixel (row, col) to ground at ellipsoidal height *h* [m].

        Parameters
        ----------
        row, col : float
            Image coordinates (0-based).
        h : float
            Target ellipsoidal height [m].

        Returns
        -------
        lon_deg, lat_deg, h_out : float
            Geodetic coordinates of the ground point.  Returns (nan, nan, nan)
            if the ray does not intersect the ellipsoid.
        """
        t = self._t_from_row(row)
        P_s, R_b2e = self._sat_state(t)

        u_body = self._col_to_look_body(col)
        d_ecef = R_b2e @ u_body
        d_ecef /= np.linalg.norm(d_ecef)

        P_ground = _ray_ellipsoid_intersect(P_s, d_ecef, h)
        if P_ground is None:
            return np.nan, np.nan, np.nan

        lon, lat, h_out = ecef_to_geodetic(P_ground[0], P_ground[1], P_ground[2])
        return float(lon), float(lat), float(h_out)

    # ------------------------------------------------------------------
    # Forward projection: ground → image
    # ------------------------------------------------------------------

    def _along_track_residual(self, row, P_ground):
        """Along-track component of the unit look vector in the body frame.

        This is zero when *row* is the correct scan line for *P_ground*.
        """
        t = self._t_from_row(row)
        P_s, R_b2e = self._sat_state(t)
        d = P_ground - P_s
        d_norm = d / np.linalg.norm(d)
        d_body = R_b2e.T @ d_norm
        return float(d_body[0])   # X_b component (along-track)

    def forward_project(self, lon_deg, lat_deg, h=0.0):
        """Project a ground point to image pixel coordinates.

        Parameters
        ----------
        lon_deg, lat_deg : float
            Geodetic longitude and latitude [°].
        h : float
            Ellipsoidal height [m].

        Returns
        -------
        row, col : float
            Image coordinates (0-based), or (nan, nan) if not imaged.
        """
        P_ground = geodetic_to_ecef(lon_deg, lat_deg, h).ravel()

        row_lo, row_hi = 0.0, float(self.n_rows - 1)
        f_lo = self._along_track_residual(row_lo, P_ground)
        f_hi = self._along_track_residual(row_hi, P_ground)

        # Check if the ground point is within the imaged along-track swath
        if f_lo * f_hi > 0.0:
            return np.nan, np.nan

        # Bisection to find the zero of f(row) = along_track_residual
        for _ in range(60):
            row_mid = 0.5 * (row_lo + row_hi)
            if row_hi - row_lo < 1e-6:
                break
            f_mid = self._along_track_residual(row_mid, P_ground)
            if f_lo * f_mid <= 0.0:
                row_hi = row_mid
                f_hi = f_mid
            else:
                row_lo = row_mid
                f_lo = f_mid

        row = 0.5 * (row_lo + row_hi)

        # Derive column from the cross-track angle
        t = self._t_from_row(row)
        P_s, R_b2e = self._sat_state(t)
        d = P_ground - P_s
        d_body = R_b2e.T @ (d / np.linalg.norm(d))
        col = self._look_body_to_col(d_body)

        # Check column is within the sensor width
        if col < 0.0 or col > float(self.n_cols - 1):
            return np.nan, np.nan

        return float(row), float(col)
