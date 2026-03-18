"""Tests for orbit and attitude interpolation."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rpc_model.interpolation import (
    lagrange_interpolation,
    interpolate_attitude,
    OrbitInterpolator,
    AttitudeInterpolator,
    _slerp,
)


# ---------------------------------------------------------------------------
# Lagrange interpolation
# ---------------------------------------------------------------------------

class TestLagrangeInterpolation:
    def test_polynomial_exact(self):
        """Lagrange must reproduce a polynomial of degree ≤ order − 1 exactly."""
        t = np.linspace(0.0, 10.0, 20)
        # Cubic polynomial: f(t) = 2t³ - 3t² + t - 1
        values = 2 * t**3 - 3 * t**2 + t - 1
        t_query = np.array([1.5, 3.7, 7.2, 9.1])
        result  = lagrange_interpolation(t, values, t_query, order=8)
        expected = 2 * t_query**3 - 3 * t_query**2 + t_query - 1
        np.testing.assert_allclose(result, expected, rtol=1e-9)

    def test_at_nodes(self):
        """Interpolation at a node should return the node value."""
        t = np.linspace(0.0, 5.0, 10)
        values = np.sin(t)
        result = lagrange_interpolation(t, values, t[:5], order=6)
        np.testing.assert_allclose(result, values[:5], atol=1e-12)

    def test_vector_valued(self):
        """Works for vector-valued functions."""
        t = np.linspace(0.0, 2 * np.pi, 30)
        values = np.column_stack([np.sin(t), np.cos(t)])  # shape (30, 2)
        t_q = np.array([0.5, 1.0, 2.5])
        result = lagrange_interpolation(t, values, t_q, order=8)
        expected = np.column_stack([np.sin(t_q), np.cos(t_q)])
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_orbit_position(self):
        """Lagrange interpolation of a circular orbit position is accurate."""
        # Simple circular orbit in a plane
        n = 1e-3   # rad/s (fast for test)
        t = np.linspace(0.0, 100.0, 50)
        R = 7_000_000.0
        pos = np.column_stack([R * np.cos(n * t),
                               R * np.sin(n * t),
                               np.zeros_like(t)])
        t_q = np.array([10.5, 30.0, 70.3])
        result = lagrange_interpolation(t, pos, t_q, order=8)
        expected = np.column_stack([R * np.cos(n * t_q),
                                    R * np.sin(n * t_q),
                                    np.zeros_like(t_q)])
        np.testing.assert_allclose(result, expected, rtol=1e-8)


# ---------------------------------------------------------------------------
# SLERP
# ---------------------------------------------------------------------------

class TestSLERP:
    def test_endpoints(self):
        """SLERP at α=0 returns q0; at α=1 returns q1."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0])
        np.testing.assert_allclose(_slerp(q0, q1, 0.0), q0, atol=1e-12)
        np.testing.assert_allclose(_slerp(q0, q1, 1.0), q1, atol=1e-12)

    def test_midpoint_angle(self):
        """At α=0.5 the interpolated rotation should have half the total angle."""
        # Rotate 90° around Z
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        q_half = _slerp(q0, q1, 0.5)
        # Expected: 45° around Z
        q_exp = np.array([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)])
        np.testing.assert_allclose(np.abs(q_half), np.abs(q_exp), atol=1e-10)

    def test_unit_norm(self):
        """SLERP output is always a unit quaternion."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            q0 = rng.standard_normal(4); q0 /= np.linalg.norm(q0)
            q1 = rng.standard_normal(4); q1 /= np.linalg.norm(q1)
            alpha = rng.uniform(0.0, 1.0)
            q = _slerp(q0, q1, alpha)
            assert abs(np.linalg.norm(q) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# AttitudeInterpolator
# ---------------------------------------------------------------------------

class TestAttitudeInterpolator:
    def _make_interp(self):
        # Constant nadir-pointing attitude
        t = np.linspace(0.0, 10.0, 20)
        q_const = np.array([1.0, 0.0, 0.0, 0.0])
        quats = np.tile(q_const, (len(t), 1))
        return AttitudeInterpolator(t, quats)

    def test_constant_attitude(self):
        """Interpolating a constant attitude should return that attitude."""
        interp = self._make_interp()
        result = interp.get_quaternion(np.array([3.7, 7.1]))
        expected = np.array([[1.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_unit_norm_output(self):
        """Interpolated quaternions must be unit quaternions."""
        rng = np.random.default_rng(1)
        t = np.linspace(0.0, 20.0, 30)
        quats = rng.standard_normal((30, 4))
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        interp = AttitudeInterpolator(t, quats)
        t_q = np.linspace(0.5, 19.5, 100)
        out = interp.get_quaternion(t_q)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(t_q)), atol=1e-12)


# ---------------------------------------------------------------------------
# OrbitInterpolator
# ---------------------------------------------------------------------------

class TestOrbitInterpolator:
    def _make_circular(self, n_samples=60, R=7_000_000.0, n=7.3e-4):
        t = np.linspace(0.0, 2 * np.pi / n, n_samples)
        pos = np.column_stack([R * np.cos(n * t),
                               R * np.sin(n * t),
                               np.zeros_like(t)])
        vel = np.column_stack([-R * n * np.sin(n * t),
                                R * n * np.cos(n * t),
                                np.zeros_like(t)])
        return t, pos, vel

    def test_position_accuracy(self):
        """Lagrange interpolation of circular orbit position is sub-metre accurate."""
        t, pos, vel = self._make_circular()
        interp = OrbitInterpolator(t, pos, vel, order=8)
        R = 7_000_000.0
        n = 7.3e-4
        t_q = np.array([200.0, 500.0, 1000.0])
        result = interp.get_position(t_q)
        expected = np.column_stack([R * np.cos(n * t_q),
                                     R * np.sin(n * t_q),
                                     np.zeros_like(t_q)])
        np.testing.assert_allclose(result, expected, atol=0.5)   # sub-metre

    def test_velocity_accuracy(self):
        """Interpolated velocity should match analytical velocity."""
        t, pos, vel = self._make_circular()
        interp = OrbitInterpolator(t, pos, vel, order=8)
        R, n = 7_000_000.0, 7.3e-4
        t_q = np.array([300.0, 800.0])
        result = interp.get_velocity(t_q)
        expected = np.column_stack([-R * n * np.sin(n * t_q),
                                     R * n * np.cos(n * t_q),
                                     np.zeros_like(t_q)])
        np.testing.assert_allclose(result, expected, rtol=1e-6)
