"""
RPC (Rational Polynomial Coefficients) model solver and evaluator.

Mathematical formulation
------------------------
The RPC model expresses normalised image coordinates as ratios of
20-term third-degree polynomials of normalised ground coordinates:

    row_n = Num_L(P, L, H) / Den_L(P, L, H)
    col_n = Num_S(P, L, H) / Den_S(P, L, H)

where

    P = (lat  − LAT_OFF)    / LAT_SCALE
    L = (lon  − LON_OFF)    / LON_SCALE
    H = (h    − H_OFF)      / H_SCALE
    row_n = (row − ROW_OFF) / ROW_SCALE
    col_n = (col − COL_OFF) / COL_SCALE

All offsets are midpoint values; all scales are maximum deviations (so that
P, L, H, row_n, col_n ∈ [−1, 1]).

The 20 monomials in standard order (OGC / DigitalGlobe convention):
    [1, L, P, H, LP, LH, PH, L², P², H²,
     PLH, L³, LP², LH², P²L, P³, PH², L²H, P²H, H³]

Denominator constraint: the constant term of each denominator is fixed at 1.
This gives 78 free parameters total (4 × 20 − 2).

Parameter estimation
--------------------
For each training point, substituting the RPC equation and rearranging:

    row_n · Den_L − Num_L = 0
  ⟹ −[m] · a  +  row_n · [m₁…m₁₉] · b  =  −row_n

where *m* is the monomial vector and *b* = [b₂, …, b₂₀] (19 free denominator
coefficients, b₁ = 1 fixed).

This is a 39×1 linear system per training point for the row model (and
similarly for the column model).  We solve by regularised least squares
(Tikhonov / ridge regression).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Monomial basis
# ---------------------------------------------------------------------------

def _monomials(P, L, H):
    """Compute the 20 RPC monomial terms for scalar or array inputs.

    Parameters
    ----------
    P, L, H : float or ndarray, shape (N,)
        Normalised latitude, longitude, height.

    Returns
    -------
    numpy.ndarray, shape (20,) or (N, 20)
    """
    scalar = np.ndim(P) == 0
    P = np.atleast_1d(np.asarray(P, dtype=float))
    L = np.atleast_1d(np.asarray(L, dtype=float))
    H = np.atleast_1d(np.asarray(H, dtype=float))
    m = np.column_stack([
        np.ones_like(P),   # 1
        L,                 # L
        P,                 # P
        H,                 # H
        L * P,             # LP
        L * H,             # LH
        P * H,             # PH
        L ** 2,            # L²
        P ** 2,            # P²
        H ** 2,            # H²
        P * L * H,         # PLH
        L ** 3,            # L³
        L * P ** 2,        # LP²
        L * H ** 2,        # LH²
        P ** 2 * L,        # P²L
        P ** 3,            # P³
        P * H ** 2,        # PH²
        L ** 2 * H,        # L²H
        P ** 2 * H,        # P²H
        H ** 3,            # H³
    ])
    return m[0] if scalar else m


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _compute_offsets_scales(values):
    """Compute offset (midpoint) and scale (half-range) for normalisation."""
    lo, hi = np.min(values), np.max(values)
    offset = 0.5 * (lo + hi)
    scale = 0.5 * (hi - lo)
    if scale == 0.0:
        scale = 1.0
    return offset, scale


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class RPCSolver:
    """Fit RPC model coefficients from a set of ground ↔ image tie-points.

    Parameters
    ----------
    lambda_reg : float
        Tikhonov regularisation strength (default 1e-4).  A small positive
        value prevents singular matrices and reduces overfitting.
    """

    def __init__(self, lambda_reg=1e-4):
        self.lambda_reg = float(lambda_reg)

    def fit(self, lons, lats, hs, rows, cols):
        """Fit the RPC model from control-point data.

        Parameters
        ----------
        lons, lats, hs : array-like, shape (N,)
            Ground coordinates (°, °, m).
        rows, cols : array-like, shape (N,)
            Corresponding image coordinates (0-based pixel indices).

        Returns
        -------
        RPCModel
            Fitted RPC model ready for evaluation.
        """
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        hs   = np.asarray(hs,   dtype=float)
        rows = np.asarray(rows, dtype=float)
        cols = np.asarray(cols, dtype=float)

        # --- normalisation offsets and scales ---
        lon_off,  lon_scale  = _compute_offsets_scales(lons)
        lat_off,  lat_scale  = _compute_offsets_scales(lats)
        h_off,    h_scale    = _compute_offsets_scales(hs)
        row_off,  row_scale  = _compute_offsets_scales(rows)
        col_off,  col_scale  = _compute_offsets_scales(cols)

        offsets = dict(
            lon_off=lon_off,   lon_scale=lon_scale,
            lat_off=lat_off,   lat_scale=lat_scale,
            h_off=h_off,       h_scale=h_scale,
            row_off=row_off,   row_scale=row_scale,
            col_off=col_off,   col_scale=col_scale,
        )

        L = (lons - lon_off) / lon_scale
        P = (lats - lat_off) / lat_scale
        H = (hs   - h_off)   / h_scale
        row_n = (rows - row_off) / row_scale
        col_n = (cols - col_off) / col_scale

        # --- solve row model ---
        a_L, b_L = self._solve_model(P, L, H, row_n)

        # --- solve column model ---
        a_S, b_S = self._solve_model(P, L, H, col_n)

        return RPCModel(
            a_L=a_L, b_L=b_L,
            a_S=a_S, b_S=b_S,
            offsets=offsets,
        )

    def _solve_model(self, P, L, H, img_n):
        """Solve one RPC sub-model (row or column).

        Returns
        -------
        a : ndarray, shape (20,)   numerator coefficients
        b : ndarray, shape (19,)   denominator free coefficients (b₂…b₂₀)
        """
        N = len(P)
        m = _monomials(P, L, H)   # shape (N, 20)

        # Design matrix: [-m | img_n * m[:,1:]]  shape (N, 39)
        A = np.hstack([-m, img_n[:, np.newaxis] * m[:, 1:]])
        rhs = -img_n

        # Regularised least squares: (AᵀA + λI) x = Aᵀ rhs
        lam = self.lambda_reg
        x = np.linalg.solve(
            A.T @ A + lam * np.eye(39),
            A.T @ rhs,
        )

        a = x[:20]       # numerator
        b = x[20:]       # denominator (free coefficients b₂…b₂₀)
        return a, b


# ---------------------------------------------------------------------------
# RPC Model evaluation
# ---------------------------------------------------------------------------

@dataclass
class RPCModel:
    """RPC model with numerator / denominator polynomial coefficients.

    Attributes
    ----------
    a_L, b_L : ndarray
        Row model numerator (20 terms) and denominator free terms (19 terms).
    a_S, b_S : ndarray
        Column model numerator (20 terms) and denominator free terms (19 terms).
    offsets : dict
        Normalisation offsets and scales.
    """

    a_L: np.ndarray
    b_L: np.ndarray
    a_S: np.ndarray
    b_S: np.ndarray
    offsets: dict = field(default_factory=dict)

    def predict(self, lons, lats, hs):
        """Predict image coordinates for ground point(s).

        Parameters
        ----------
        lons, lats, hs : float or array-like, shape (N,)

        Returns
        -------
        rows, cols : ndarray, shape (N,)
        """
        lons = np.atleast_1d(np.asarray(lons, dtype=float))
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        hs   = np.atleast_1d(np.asarray(hs,   dtype=float))
        o = self.offsets

        L = (lons - o["lon_off"]) / o["lon_scale"]
        P = (lats - o["lat_off"]) / o["lat_scale"]
        H = (hs   - o["h_off"])   / o["h_scale"]

        m = _monomials(P, L, H)   # (N, 20)

        # Full denominator vectors (b₁ = 1 fixed)
        den_L = np.hstack([[1.0], self.b_L])
        den_S = np.hstack([[1.0], self.b_S])

        row_n = (m @ self.a_L) / (m @ den_L)
        col_n = (m @ self.a_S) / (m @ den_S)

        rows = row_n * o["row_scale"] + o["row_off"]
        cols = col_n * o["col_scale"] + o["col_off"]
        return rows, cols

    def assess_accuracy(self, lons, lats, hs, rows_ref, cols_ref):
        """Compute residuals between predicted and reference image coordinates.

        Parameters
        ----------
        lons, lats, hs : array-like
            Ground coordinates of check points.
        rows_ref, cols_ref : array-like
            Reference (strict-model) image coordinates.

        Returns
        -------
        dict with keys:
            row_rmse, col_rmse, row_max, col_max, row_res, col_res
        """
        rows_pred, cols_pred = self.predict(lons, lats, hs)
        row_res = rows_pred - np.asarray(rows_ref, dtype=float)
        col_res = cols_pred - np.asarray(cols_ref, dtype=float)
        return dict(
            row_rmse=float(np.sqrt(np.mean(row_res ** 2))),
            col_rmse=float(np.sqrt(np.mean(col_res ** 2))),
            row_max =float(np.max(np.abs(row_res))),
            col_max =float(np.max(np.abs(col_res))),
            row_res =row_res,
            col_res =col_res,
        )

    def to_dict(self):
        """Serialise model coefficients and offsets to a plain dictionary."""
        return dict(
            a_L=self.a_L.tolist(),
            b_L=self.b_L.tolist(),
            a_S=self.a_S.tolist(),
            b_S=self.b_S.tolist(),
            offsets={k: float(v) for k, v in self.offsets.items()},
        )

    @classmethod
    def from_dict(cls, d):
        """Reconstruct an RPCModel from a dictionary produced by :meth:`to_dict`."""
        return cls(
            a_L=np.array(d["a_L"]),
            b_L=np.array(d["b_L"]),
            a_S=np.array(d["a_S"]),
            b_S=np.array(d["b_S"]),
            offsets=d["offsets"],
        )
