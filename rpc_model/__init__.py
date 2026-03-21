"""
rpc_model – Terrain-independent RPC model construction for satellite imagery.

Workflow
--------
1. J2000 → ECEF coordinate transformation  (coord_transform)
2. Orbit and attitude interpolation         (interpolation)
3. Strict pushbroom imaging model           (imaging_model)
4. Ground control grid construction         (control_grid)
5. RPC parameter solving and assessment     (rpc_solver)
"""

from .constants import WGS84_A, WGS84_B, WGS84_E2, WGS84_F, GM, OMEGA_E
from .coord_transform import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    j2000_to_ecef,
    ecef_to_j2000,
    j2000_to_ecef_matrix,
    quaternion_to_rotation_matrix,
    attitude_j2000_to_ecef_quaternion,
)

from .data_parser import (
    OrbitSample,
    OrbitData,
    AttitudeSample,
    AttitudeData,
    ImagingTimeData,
    CBRData,
    NADBiasData,
    RPCTextData,
    NADFileConfig,
    NADDataParser,
    load_nad_bundle,
)

__all__ = [
    "WGS84_A", "WGS84_B", "WGS84_E2", "WGS84_F", "GM", "OMEGA_E",
    "geodetic_to_ecef", "ecef_to_geodetic",
    "j2000_to_ecef", "ecef_to_j2000", "j2000_to_ecef_matrix",
    "quaternion_to_rotation_matrix",
    "attitude_j2000_to_ecef_quaternion",
    "lagrange_interpolation", "interpolate_attitude",
    "OrbitInterpolator", "AttitudeInterpolator",
    "PushbroomImagingModel",
    "build_control_grid",
    "RPCSolver", "RPCModel",
    "OrbitSample", "OrbitData",
    "AttitudeSample", "AttitudeData",
    "ImagingTimeData", "CBRData", "NADBiasData", "RPCTextData",
    "NADFileConfig",
    "NADDataParser", "load_nad_bundle",
]
