"""Tests for RPC solver and RPCModel evaluation."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rpc_model.rpc_solver import RPCSolver, RPCModel, _monomials


# ---------------------------------------------------------------------------
# Monomial basis
# ---------------------------------------------------------------------------

class TestMonomials:
    def test_count(self):
        """Scalar input returns a 1-D array of 20 terms."""
        m = _monomials(0.1, 0.2, 0.3)
        assert m.shape == (20,)

    def test_first_term_is_one(self):
        """First monomial is always 1."""
        m = _monomials(0.0, 0.0, 0.0)
        assert float(m[0]) == 1.0

    def test_vectorised(self):
        """Works for N input points; array result has shape (N, 20)."""
        P = np.array([0.1, 0.5, -0.3])
        L = np.array([0.2, -0.1, 0.4])
        H = np.array([0.0, 0.3, -0.5])
        m = _monomials(P, L, H)
        assert m.shape == (3, 20)
        # Check second column = L
        np.testing.assert_allclose(m[:, 1], L)
        # Check third column = P
        np.testing.assert_allclose(m[:, 2], P)


# ---------------------------------------------------------------------------
# RPCSolver: identity / trivial cases
# ---------------------------------------------------------------------------

class TestRPCSolverTrivial:
    """Test solver on synthetically generated RPC data (no strict model needed)."""

    def _make_data(self, n=500, seed=0):
        """Generate random ground points and compute 'true' image coords
        using a known set of RPC coefficients."""
        rng = np.random.default_rng(seed)
        lons = rng.uniform(116.0, 117.0, n)
        lats = rng.uniform(39.0,  40.0,  n)
        hs   = rng.uniform(-500,   2000,  n)
        # Simple affine "true" model: row = A*lat + B*lon, col = C*lon + D*h
        rows = 500.0 + 2000.0 * (lats - 39.5) + 500.0 * (lons - 116.5)
        cols = 250.0 + 1000.0 * (lons - 116.5)
        return lons, lats, hs, rows, cols

    def test_affine_model_accuracy(self):
        """Solver should fit a linear ground-to-image mapping nearly exactly."""
        lons, lats, hs, rows, cols = self._make_data(n=500)
        solver = RPCSolver(lambda_reg=1e-6)
        rpc    = solver.fit(lons, lats, hs, rows, cols)
        acc    = rpc.assess_accuracy(lons, lats, hs, rows, cols)
        assert acc["row_rmse"] < 0.1, f"Row RMSE too large: {acc['row_rmse']}"
        assert acc["col_rmse"] < 0.1, f"Col RMSE too large: {acc['col_rmse']}"

    def test_check_set_generalises(self):
        """Model fit on training data should generalise to check points."""
        train = self._make_data(n=500, seed=0)
        check = self._make_data(n=300, seed=42)
        solver = RPCSolver(lambda_reg=1e-5)
        rpc    = solver.fit(*train)
        acc    = rpc.assess_accuracy(*check)
        assert acc["row_rmse"] < 0.5
        assert acc["col_rmse"] < 0.5


# ---------------------------------------------------------------------------
# RPCSolver with strict imaging model
# ---------------------------------------------------------------------------

class TestRPCSolverWithImagingModel:
    """Integration test: RPC fit to a synthetic pushbroom imaging model."""

    @pytest.fixture(scope="class")
    def rpc_and_check(self):
        from rpc_model.imaging_model import PushbroomImagingModel
        from rpc_model.interpolation import OrbitInterpolator, AttitudeInterpolator
        from rpc_model.coord_transform import (
            j2000_to_ecef_matrix, rotation_matrix_to_quaternion,
        )
        from rpc_model.control_grid import build_control_grid
        from rpc_model.constants import WGS84_A, GM

        alt  = 500_000.0
        incl = np.deg2rad(97.4)
        a    = WGS84_A + alt
        n_orb = np.sqrt(GM / a**3)
        v_orb = np.sqrt(GM / a)
        n_rows, n_cols = 300, 150

        gsd = 5.0
        dt  = gsd / v_orb
        scan_times = np.arange(n_rows) * dt
        margin = 30.0

        def orbit_j2000(t):
            u = n_orb * t
            return np.array([a * np.cos(u),
                             a * np.sin(u) * np.cos(incl),
                             a * np.sin(u) * np.sin(incl)])

        def vel_j2000(t):
            u = n_orb * t
            vv = a * n_orb
            return np.array([-vv * np.sin(u),
                              vv * np.cos(u) * np.cos(incl),
                              vv * np.cos(u) * np.sin(incl)])

        t_j2000_epoch = 788_918_400.0
        orb_t = np.arange(-margin, scan_times[-1] + margin + 1.0, 5.0)
        orb_pos = np.array([
            j2000_to_ecef_matrix(
                julian_day_offset=(t + t_j2000_epoch) / 86400.0
            ) @ orbit_j2000(t)
            for t in orb_t
        ])
        orb_vel = np.array([
            j2000_to_ecef_matrix(
                julian_day_offset=(t + t_j2000_epoch) / 86400.0
            ) @ vel_j2000(t)
            for t in orb_t
        ])

        att_t = np.arange(-margin, scan_times[-1] + margin + 0.5, 0.5)
        quats = []
        for t in att_t:
            r = orbit_j2000(t)
            v = vel_j2000(t)
            X_b = v / np.linalg.norm(v)
            Z_b = -r / np.linalg.norm(r)
            Y_b = np.cross(Z_b, X_b)
            Y_b /= np.linalg.norm(Y_b)
            R = np.column_stack([X_b, Y_b, Z_b])
            quats.append(rotation_matrix_to_quaternion(R))

        fov_half = np.deg2rad(3.0)
        pointing_angles = np.linspace(-fov_half, fov_half, n_cols)

        model = PushbroomImagingModel(
            scan_times=scan_times,
            orbit_interp=OrbitInterpolator(orb_t, orb_pos, orb_vel, order=8),
            attitude_interp=AttitudeInterpolator(att_t, np.array(quats)),
            pointing_angles=pointing_angles,
            julian_day_base=t_j2000_epoch / 86400.0,
        )

        # Build training and check grids
        train_lons, train_lats, train_hs, train_rows, train_cols = build_control_grid(
            model, heights=[-500.0, 0.0, 500.0, 1000.0, 2000.0],
            n_row_levels=8, n_col_levels=8,
        )
        check_lons, check_lats, check_hs, check_rows, check_cols = build_control_grid(
            model, heights=[-250.0, 250.0, 750.0, 1500.0],
            n_row_levels=12, n_col_levels=12,
        )

        solver = RPCSolver(lambda_reg=1e-4)
        rpc    = solver.fit(train_lons, train_lats, train_hs, train_rows, train_cols)

        acc_train = rpc.assess_accuracy(train_lons, train_lats, train_hs, train_rows, train_cols)
        acc_check = rpc.assess_accuracy(check_lons, check_lats, check_hs, check_rows, check_cols)

        return rpc, acc_train, acc_check

    def test_training_row_rmse(self, rpc_and_check):
        _, acc_train, _ = rpc_and_check
        assert acc_train["row_rmse"] < 0.1, f"Training row RMSE = {acc_train['row_rmse']:.4f}"

    def test_training_col_rmse(self, rpc_and_check):
        _, acc_train, _ = rpc_and_check
        assert acc_train["col_rmse"] < 0.1, f"Training col RMSE = {acc_train['col_rmse']:.4f}"

    def test_check_row_rmse(self, rpc_and_check):
        _, _, acc_check = rpc_and_check
        assert acc_check["row_rmse"] < 0.5, f"Check row RMSE = {acc_check['row_rmse']:.4f}"

    def test_check_col_rmse(self, rpc_and_check):
        _, _, acc_check = rpc_and_check
        assert acc_check["col_rmse"] < 0.5, f"Check col RMSE = {acc_check['col_rmse']:.4f}"


# ---------------------------------------------------------------------------
# RPCModel serialisation
# ---------------------------------------------------------------------------

class TestRPCModelSerialisation:
    def test_to_from_dict(self):
        """to_dict / from_dict round-trip preserves all coefficients."""
        rng = np.random.default_rng(5)
        rpc = RPCModel(
            a_L=rng.standard_normal(20),
            b_L=rng.standard_normal(19),
            a_S=rng.standard_normal(20),
            b_S=rng.standard_normal(19),
            offsets=dict(lon_off=116.5, lon_scale=0.5,
                         lat_off=39.5, lat_scale=0.5,
                         h_off=500.0, h_scale=2500.0,
                         row_off=500.0, row_scale=500.0,
                         col_off=250.0, col_scale=250.0),
        )
        d = rpc.to_dict()
        rpc2 = RPCModel.from_dict(d)
        np.testing.assert_array_equal(rpc2.a_L, rpc.a_L)
        np.testing.assert_array_equal(rpc2.b_L, rpc.b_L)
        np.testing.assert_array_equal(rpc2.a_S, rpc.a_S)
        np.testing.assert_array_equal(rpc2.b_S, rpc.b_S)
        assert rpc2.offsets == rpc.offsets
