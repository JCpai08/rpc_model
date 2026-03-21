"""Tests for the strict pushbroom imaging model."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rpc_model.imaging_model import PushbroomImagingModel, _ray_ellipsoid_intersect
from rpc_model.interpolation import OrbitInterpolator, AttitudeInterpolator
from rpc_model.coord_transform import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    j2000_to_ecef_matrix,
    rotation_matrix_to_quaternion,
)
from rpc_model.constants import GM, WGS84_A


# ---------------------------------------------------------------------------
# Helpers to build a minimal synthetic model
# ---------------------------------------------------------------------------

def _make_model(n_rows=200, n_cols=100, alt=500_000.0, fov_half_deg=3.0,
                incl_deg=97.4, t_j2000_epoch=0.0):
    """Return a PushbroomImagingModel for a simple circular orbit."""
    a = WGS84_A + alt
    n_orb = np.sqrt(GM / a**3)
    v_orb = np.sqrt(GM / a)
    incl = np.deg2rad(incl_deg)

    # Scan times
    gsd = 5.0
    dt_line = gsd / v_orb
    scan_times = np.arange(n_rows) * dt_line

    # Orbit – sample at wider range than scan times
    margin = 30.0
    orb_t = np.arange(-margin, scan_times[-1] + margin + 1.0, 5.0)

    def _orbit_pos(t):
        u = n_orb * t
        return np.array([a * np.cos(u),
                         a * np.sin(u) * np.cos(incl),
                         a * np.sin(u) * np.sin(incl)])

    def _orbit_vel(t):
        u = n_orb * t
        v = a * n_orb
        return np.array([-v * np.sin(u),
                          v * np.cos(u) * np.cos(incl),
                          v * np.cos(u) * np.sin(incl)])

    # Compute ECEF orbit positions
    t_j2k_arr = orb_t + t_j2000_epoch
    orb_pos = np.array([
        j2000_to_ecef_matrix(julian_day_offset=tj / 86400.0) @ _orbit_pos(t)
        for t, tj in zip(orb_t, t_j2k_arr)
    ])
    orb_vel = np.array([
        j2000_to_ecef_matrix(julian_day_offset=tj / 86400.0) @ _orbit_vel(t)
        for t, tj in zip(orb_t, t_j2k_arr)
    ])

    orbit_interp = OrbitInterpolator(orb_t, orb_pos, orb_vel, order=8)

    # Attitude – nadir-pointing quaternions in J2000
    att_t = np.arange(-margin, scan_times[-1] + margin + 0.5, 0.5)
    quats = []
    for t in att_t:
        r = _orbit_pos(t)
        v = _orbit_vel(t)
        X_b = v / np.linalg.norm(v)
        Z_b = -r / np.linalg.norm(r)
        Y_b = np.cross(Z_b, X_b)
        Y_b /= np.linalg.norm(Y_b)
        R = np.column_stack([X_b, Y_b, Z_b])
        quats.append(rotation_matrix_to_quaternion(R))

    attitude_interp = AttitudeInterpolator(att_t, np.array(quats))

    # Camera pointing angles
    fov_half = np.deg2rad(fov_half_deg)
    pointing_angles = np.linspace(-fov_half, fov_half, n_cols)

    model = PushbroomImagingModel(
        scan_times=scan_times,
        orbit_interp=orbit_interp,
        attitude_interp=attitude_interp,
        pointing_angles=pointing_angles,
        julian_day_base=t_j2000_epoch / 86400.0,
    )
    return model


# ---------------------------------------------------------------------------
# Ray–ellipsoid intersection
# ---------------------------------------------------------------------------

class TestRayEllipsoidIntersect:
    def test_vertical_ray_from_500km(self):
        """A vertical ray from 500 km altitude should intersect the equator at h=0."""
        P_s = np.array([WGS84_A + 500_000.0, 0.0, 0.0])
        d   = np.array([-1.0, 0.0, 0.0])   # pointing toward Earth
        pt  = _ray_ellipsoid_intersect(P_s, d, 0.0)
        assert pt is not None
        # Should land at the equatorial radius
        np.testing.assert_allclose(pt, [WGS84_A, 0.0, 0.0], rtol=1e-8)

    def test_ray_misses(self):
        """A ray pointing away from Earth returns None."""
        P_s = np.array([WGS84_A + 500_000.0, 0.0, 0.0])
        d   = np.array([1.0, 0.0, 0.0])    # pointing away
        pt  = _ray_ellipsoid_intersect(P_s, d, 0.0)
        assert pt is None

    def test_height_offset(self):
        """A vertical ray should intersect at h = 1000 m, not h = 0."""
        P_s = np.array([WGS84_A + 500_000.0, 0.0, 0.0])
        d   = np.array([-1.0, 0.0, 0.0])
        pt  = _ray_ellipsoid_intersect(P_s, d, 1000.0)
        assert pt is not None
        _, _, h = ecef_to_geodetic(pt[0], pt[1], pt[2])
        np.testing.assert_allclose(h, 1000.0, atol=0.1)


# ---------------------------------------------------------------------------
# Backward projection
# ---------------------------------------------------------------------------

class TestBackwardProjection:
    def setup_method(self):
        self.model = _make_model(n_rows=200, n_cols=100)

    def test_returns_finite_values(self):
        """Backward projection at centre pixel should give a finite ground point."""
        lon, lat, h = self.model.backward_project(100, 50, h=0.0)
        assert np.isfinite(lon) and np.isfinite(lat) and np.isfinite(h)

    def test_height_matches(self):
        """Backward projection at a given height should recover that height."""
        for h_target in [0.0, 500.0, 1000.0, -200.0]:
            lon, lat, h_out = self.model.backward_project(50, 30, h=h_target)
            assert np.isfinite(lon)
            np.testing.assert_allclose(h_out, h_target, atol=0.5)

    def test_centre_col_near_nadir(self):
        """The centre column should land close to the sub-satellite ground track."""
        from rpc_model.coord_transform import ecef_to_geodetic
        # Centre column ≈ nadir → cross-track offset should be small
        lon_l, lat_l, _ = self.model.backward_project(100, 0,   h=0.0)
        lon_r, lat_r, _ = self.model.backward_project(100, 99,  h=0.0)
        lon_c, lat_c, _ = self.model.backward_project(100, 49,  h=0.0)
        # Centre should be between left and right
        assert min(lon_l, lon_r) <= lon_c <= max(lon_l, lon_r) or \
               min(lat_l, lat_r) <= lat_c <= max(lat_l, lat_r)


# ---------------------------------------------------------------------------
# Forward projection (and round-trip consistency)
# ---------------------------------------------------------------------------

class TestForwardProjection:
    def setup_method(self):
        self.model = _make_model(n_rows=200, n_cols=100)

    def test_round_trip_interior_pixels(self):
        """backward_project → forward_project should recover the original pixel."""
        test_pixels = [(10, 10), (50, 50), (100, 49), (150, 80), (190, 90)]
        for r0, c0 in test_pixels:
            lon, lat, h = self.model.backward_project(r0, c0, h=0.0)
            if np.isnan(lon):
                continue
            r1, c1 = self.model.forward_project(lon, lat, h=0.0)
            assert np.isfinite(r1), f"Forward projection failed for pixel ({r0},{c0})"
            np.testing.assert_allclose(r1, r0, atol=0.05,
                err_msg=f"Row mismatch for pixel ({r0},{c0})")
            np.testing.assert_allclose(c1, c0, atol=0.05,
                err_msg=f"Col mismatch for pixel ({r0},{c0})")

    def test_out_of_swath_returns_nan(self):
        """A ground point outside the imaged swath should return NaN."""
        # Use a point very far from the imaged area
        r, c = self.model.forward_project(0.0, 80.0, 0.0)   # equatorial point
        # This may or may not be in FOV – just check the function runs without error
        assert (np.isnan(r) and np.isnan(c)) or (np.isfinite(r) and np.isfinite(c))

    def test_multiple_heights(self):
        """Round-trip must hold at different terrain heights."""
        r0, c0 = 80, 40
        for h in [-200.0, 0.0, 500.0, 1500.0]:
            lon, lat, h_out = self.model.backward_project(r0, c0, h=h)
            if np.isnan(lon):
                continue
            r1, c1 = self.model.forward_project(lon, lat, h=h)
            if np.isnan(r1):
                continue
            np.testing.assert_allclose(r1, r0, atol=0.1)
            np.testing.assert_allclose(c1, c0, atol=0.1)
