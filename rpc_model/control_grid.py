"""
Terrain-independent ground control grid for RPC parameter solving.

Strategy ("地形无关")
----------------------
Rather than relying on ground control points from a DEM, we construct a
virtual 3-D grid of (lon, lat, h) points that uniformly samples the image
space.  For each grid point we obtain its image coordinates via the strict
backward projection model (pixel → ground at given height).

The grid is defined by:
  * ``n_row_levels`` evenly-spaced row indices across [0, n_rows − 1].
  * ``n_col_levels`` evenly-spaced column indices across [0, n_cols − 1].
  * ``heights``       list of representative ellipsoidal heights.

Each (row, col, h) triple is projected to ground using the strict backward
projection; the resulting (lon, lat, h) ↔ (row, col) pairs form the control
dataset used to fit the RPC coefficients.
"""

import numpy as np


def build_control_grid(imaging_model, heights, n_row_levels=10, n_col_levels=10):
    """Build a terrain-independent 3-D ground control grid.

    Parameters
    ----------
    imaging_model : PushbroomImagingModel
        The strict pushbroom model used for backward projection.
    heights : array-like
        List of ellipsoidal heights [m] to include in the grid.
    n_row_levels : int
        Number of evenly-spaced row samples (default 10).
    n_col_levels : int
        Number of evenly-spaced column samples (default 10).

    Returns
    -------
    lons : numpy.ndarray, shape (K,)
    lats : numpy.ndarray, shape (K,)
    hs   : numpy.ndarray, shape (K,)
    rows : numpy.ndarray, shape (K,)
    cols : numpy.ndarray, shape (K,)
        Matched ground and image coordinates.  Points where the projection
        failed (ray missed the ellipsoid) are excluded.
    """
    heights = np.asarray(heights, dtype=float)
    row_indices = np.linspace(0, imaging_model.n_rows - 1, n_row_levels)
    col_indices = np.linspace(0, imaging_model.n_cols - 1, n_col_levels)

    lons, lats, hs, rows, cols = [], [], [], [], []

    for h in heights:
        for r in row_indices:
            for c in col_indices:
                lon, lat, h_out = imaging_model.backward_project(r, c, float(h))
                if np.isnan(lon):
                    continue
                lons.append(lon)
                lats.append(lat)
                hs.append(h_out)
                rows.append(r)
                cols.append(c)

    lons = np.array(lons)
    lats = np.array(lats)
    hs = np.array(hs)
    rows = np.array(rows)
    cols = np.array(cols)

    if len(lons) == 0:
        raise RuntimeError("Control grid is empty – no valid backward projections.")

    return lons, lats, hs, rows, cols
