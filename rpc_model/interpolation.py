"""
Orbit and attitude interpolation.

* Orbit (position/velocity): Lagrange polynomial interpolation.
* Attitude (quaternions):     Piecewise SLERP (spherical linear interpolation).
"""

import numpy as np
from .coord_transform import quaternion_to_rotation_matrix


# ---------------------------------------------------------------------------
# Lagrange polynomial interpolation
# ---------------------------------------------------------------------------

def lagrange_interpolation(t_nodes, values, t_query, order=8):
    """Lagrange polynomial interpolation using the *order* nearest nodes.

    Parameters
    ----------
    t_nodes : array-like, shape (N,)
        Strictly increasing knot times.
    values : array-like, shape (N,) or (N, D)
        Function values at the knots.  Scalar (1-D) or vector (2-D) valued.
    t_query : float or array-like, shape (M,)
        Query times.
    order : int
        Number of neighbouring nodes used (polynomial degree = order − 1).

    Returns
    -------
    numpy.ndarray, shape (M,) or (M, D)
        Interpolated values.
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    t_query = np.atleast_1d(np.asarray(t_query, dtype=float))

    scalar = values.ndim == 1
    if scalar:
        values = values[:, np.newaxis]

    D = values.shape[1]
    M = len(t_query)
    results = np.empty((M, D))

    half = order // 2
    N = len(t_nodes)

    for i, t in enumerate(t_query):
        # Centre the window on the nearest node
        centre = int(np.searchsorted(t_nodes, t))
        lo = max(0, centre - half)
        hi = min(N, lo + order)
        lo = max(0, hi - order)

        t_loc = t_nodes[lo:hi]
        v_loc = values[lo:hi]
        n = len(t_loc)

        result = np.zeros(D)
        for j in range(n):
            L = 1.0
            for k in range(n):
                if k != j:
                    denom = t_loc[j] - t_loc[k]
                    if denom == 0.0:
                        raise ValueError(
                            f"Duplicate node times at index {lo+j} and {lo+k}"
                        )
                    L *= (t - t_loc[k]) / denom
            result += L * v_loc[j]
        results[i] = result

    return results[:, 0] if scalar else results


# ---------------------------------------------------------------------------
# SLERP attitude interpolation
# ---------------------------------------------------------------------------

def _slerp(q0, q1, alpha):
    """Spherical linear interpolation between two unit quaternions.

    Parameters
    ----------
    q0, q1 : array-like, shape (4,)
        Unit quaternions [qw, qx, qy, qz].
    alpha : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    numpy.ndarray, shape (4,)
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    # Ensure shortest-path rotation
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        # Linear interpolation for nearly identical quaternions
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = alpha * theta_0
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def interpolate_attitude(t_nodes, quaternions, t_query):
    """Piecewise-SLERP attitude interpolation.

    Parameters
    ----------
    t_nodes : array-like, shape (N,)
        Strictly increasing times at which quaternions are known.
    quaternions : array-like, shape (N, 4)
        Unit quaternions [qw, qx, qy, qz] at each node.
    t_query : float or array-like, shape (M,)
        Query times.

    Returns
    -------
    numpy.ndarray, shape (M, 4)
        Interpolated unit quaternions.
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    quaternions = np.asarray(quaternions, dtype=float)
    t_query = np.atleast_1d(np.asarray(t_query, dtype=float))

    M = len(t_query)
    result = np.empty((M, 4))

    for i, t in enumerate(t_query):
        idx = int(np.searchsorted(t_nodes, t))
        idx = np.clip(idx, 1, len(t_nodes) - 1)
        t0, t1 = t_nodes[idx - 1], t_nodes[idx]
        q0, q1 = quaternions[idx - 1], quaternions[idx]
        alpha = (t - t0) / (t1 - t0) if (t1 - t0) > 0.0 else 0.0
        result[i] = _slerp(q0, q1, alpha)

    return result


# ---------------------------------------------------------------------------
# High-level interpolator classes
# ---------------------------------------------------------------------------

class OrbitInterpolator:
    """Interpolates satellite position (and velocity) from discrete orbit data.

    Parameters
    ----------
    times : array-like, shape (N,)
        Epoch times [s] of the orbit samples (e.g. seconds after J2000).
    positions : array-like, shape (N, 3)
        ECEF position vectors [m] at each epoch.
    velocities : array-like, shape (N, 3) or None
        ECEF velocity vectors [m/s].  If *None*, velocities are estimated by
        numerical differentiation of the Lagrange interpolant.
    order : int
        Lagrange interpolation order (default 8).
    """

    def __init__(self, times, positions, velocities=None, order=8):
        self.times = np.asarray(times, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = (
            np.asarray(velocities, dtype=float) if velocities is not None else None
        )
        self.order = order

    def get_position(self, t):
        """Interpolated ECEF position [m] at time(s) *t*.

        Parameters
        ----------
        t : float or array-like, shape (M,)

        Returns
        -------
        numpy.ndarray, shape (M, 3)
        """
        return lagrange_interpolation(self.times, self.positions,
                                      np.atleast_1d(t), self.order)

    def get_velocity(self, t):
        """Interpolated ECEF velocity [m/s] at time(s) *t*.

        Parameters
        ----------
        t : float or array-like

        Returns
        -------
        numpy.ndarray, shape (M, 3)
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.velocities is not None:
            return lagrange_interpolation(self.times, self.velocities, t, self.order)
        # Numerical differentiation with small step
        dt = 1.0e-2  # 10 ms
        p_fwd = lagrange_interpolation(self.times, self.positions, t + dt, self.order)
        p_bwd = lagrange_interpolation(self.times, self.positions, t - dt, self.order)
        return (p_fwd - p_bwd) / (2.0 * dt)


class AttitudeInterpolator:
    """Interpolates satellite attitude quaternions using piecewise SLERP.

    Parameters
    ----------
    times : array-like, shape (N,)
        Epoch times [s].
    quaternions : array-like, shape (N, 4)
        Unit quaternions [qw, qx, qy, qz] giving the rotation from body
        frame to J2000 (ECI) frame at each epoch.
    """

    def __init__(self, times, quaternions):
        self.times = np.asarray(times, dtype=float)
        self.quaternions = np.asarray(quaternions, dtype=float)

    def get_quaternion(self, t):
        """Interpolated quaternion(s) at time(s) *t*.

        Returns
        -------
        numpy.ndarray, shape (M, 4)
        """
        return interpolate_attitude(
            self.times, self.quaternions, np.atleast_1d(t)
        )

    def get_rotation_body2j2000(self, t):
        """3 × 3 rotation matrix body → J2000 at scalar time *t*."""
        q = self.get_quaternion(np.array([float(t)]))[0]
        return quaternion_to_rotation_matrix(q)

    def get_rotation_body2ecef(self, t, t_j2000=None):
        """3 × 3 rotation matrix body → ECEF at scalar time *t*.

        Parameters
        ----------
        t : float
            Imaging epoch time [s] (same reference as *times* passed to __init__).
        t_j2000 : float or None
            Time [s] after J2000 epoch for the ECEF rotation.  If *None*,
            *t* is used directly as the J2000 offset (valid when the orbit
            epoch coincides with J2000).
        """
        from .coord_transform import j2000_to_ecef_matrix
        if t_j2000 is None:
            t_j2000 = t
        R_b2j = self.get_rotation_body2j2000(t)
        R_j2e = j2000_to_ecef_matrix(t_j2000)
        return R_j2e @ R_b2j
