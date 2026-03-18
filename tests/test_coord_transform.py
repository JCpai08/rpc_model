"""Tests for coordinate transformation functions."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rpc_model.coord_transform import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    j2000_to_ecef,
    ecef_to_j2000,
    j2000_to_ecef_matrix,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


# ---------------------------------------------------------------------------
# WGS84 geodetic ↔ ECEF round-trip
# ---------------------------------------------------------------------------

class TestGeodeticECEF:
    def test_equator_prime_meridian(self):
        """Point on equator at prime meridian (lon=0, lat=0, h=0)."""
        from rpc_model.constants import WGS84_A
        xyz = geodetic_to_ecef(0.0, 0.0, 0.0)
        np.testing.assert_allclose(xyz, [WGS84_A, 0.0, 0.0], rtol=1e-10)

    def test_north_pole(self):
        """Geographic north pole should lie on the z-axis."""
        from rpc_model.constants import WGS84_B
        xyz = geodetic_to_ecef(0.0, 90.0, 0.0)
        np.testing.assert_allclose(xyz[0], 0.0, atol=1e-4)
        np.testing.assert_allclose(xyz[1], 0.0, atol=1e-4)
        np.testing.assert_allclose(xyz[2], WGS84_B, rtol=1e-8)

    def test_round_trip_scalar(self):
        """geodetic → ECEF → geodetic should recover the original values."""
        lon0, lat0, h0 = 116.4, 39.9, 100.0   # near Beijing
        xyz = geodetic_to_ecef(lon0, lat0, h0)
        lon1, lat1, h1 = ecef_to_geodetic(xyz[0], xyz[1], xyz[2])
        np.testing.assert_allclose(lon1, lon0, atol=1e-9)
        np.testing.assert_allclose(lat1, lat0, atol=1e-9)
        np.testing.assert_allclose(h1, h0, atol=1e-4)

    def test_round_trip_array(self):
        """Vectorised round-trip for several points."""
        lons = np.array([0.0, 90.0, 180.0, -90.0, 45.0])
        lats = np.array([0.0, 45.0, -45.0, 60.0, -30.0])
        hs   = np.array([0.0, 1000.0, 500.0, 200.0, 800.0])
        xyz  = geodetic_to_ecef(lons, lats, hs)
        lons2, lats2, hs2 = ecef_to_geodetic(xyz[..., 0], xyz[..., 1], xyz[..., 2])
        np.testing.assert_allclose(lons2, lons, atol=1e-9)
        np.testing.assert_allclose(lats2, lats, atol=1e-9)
        np.testing.assert_allclose(hs2, hs, atol=1e-4)

    def test_high_altitude(self):
        """Round-trip at satellite altitude (~500 km)."""
        lon0, lat0, h0 = 30.0, -10.0, 500_000.0
        xyz = geodetic_to_ecef(lon0, lat0, h0)
        lon1, lat1, h1 = ecef_to_geodetic(xyz[0], xyz[1], xyz[2])
        np.testing.assert_allclose(lon1, lon0, atol=1e-9)
        np.testing.assert_allclose(lat1, lat0, atol=1e-9)
        np.testing.assert_allclose(h1, h0, atol=1e-3)


# ---------------------------------------------------------------------------
# J2000 ↔ ECEF
# ---------------------------------------------------------------------------

class TestJ2000ECEF:
    def test_rotation_matrix_orthogonal(self):
        """j2000_to_ecef_matrix must be orthogonal (R · Rᵀ = I)."""
        for t in [0.0, 3600.0, 86400.0, 1e7]:
            R = j2000_to_ecef_matrix(t)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)

    def test_j2000_ecef_round_trip(self):
        """ecef_to_j2000(j2000_to_ecef(v)) == v."""
        v = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])
        for t in [0.0, 1000.0, 86400.0]:
            v_ecef = j2000_to_ecef(v, t)
            v_back = ecef_to_j2000(v_ecef, t)
            np.testing.assert_allclose(v_back, v, atol=1e-6)

    def test_earth_rotation_rate(self):
        """After one sidereal day the ECEF frame should be close to aligned."""
        sidereal_day = 2.0 * np.pi / (7.292_115_0e-5)   # ~86 164 s
        v = np.array([1.0, 0.0, 0.0])
        R0 = j2000_to_ecef_matrix(0.0)
        R1 = j2000_to_ecef_matrix(sidereal_day)
        # The two rotation matrices should be nearly identical
        np.testing.assert_allclose(R1, R0, atol=1e-8)


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

class TestQuaternion:
    def test_identity(self):
        """Identity quaternion should give identity matrix."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-14)

    def test_rotation_x_90(self):
        """90° rotation around X axis."""
        q = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        v = np.array([0.0, 1.0, 0.0])
        v_rot = R @ v
        np.testing.assert_allclose(v_rot, [0.0, 0.0, 1.0], atol=1e-14)

    def test_rotation_z_90(self):
        """90° rotation around Z axis."""
        q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        R = quaternion_to_rotation_matrix(q)
        v = np.array([1.0, 0.0, 0.0])
        v_rot = R @ v
        np.testing.assert_allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-14)

    def test_round_trip_matrix_quaternion(self):
        """quaternion → matrix → quaternion should be stable."""
        angle = np.deg2rad(35.0)
        axis  = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])
        q_in  = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])
        R     = quaternion_to_rotation_matrix(q_in)
        q_out = rotation_matrix_to_quaternion(R)
        # Sign ambiguity: compare |q_in ± q_out|
        diff_pos = np.linalg.norm(q_out - q_in)
        diff_neg = np.linalg.norm(q_out + q_in)
        assert min(diff_pos, diff_neg) < 1e-12

    def test_orthogonal_matrix(self):
        """The matrix from any unit quaternion must be orthogonal."""
        q = np.array([0.5, 0.5, 0.5, 0.5])   # 120° around (1,1,1)/√3
        R = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)
