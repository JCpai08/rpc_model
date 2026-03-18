"""Physical and geodetic constants."""

# WGS84 ellipsoid
WGS84_A = 6_378_137.0          # semi-major axis [m]
WGS84_F = 1.0 / 298.257_223_563  # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis [m]
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2  # first eccentricity squared

# Earth gravity / rotation
GM = 3.986_004_418e14    # gravitational parameter [m³/s²]
OMEGA_E = 7.292_115_0e-5  # Earth sidereal rotation rate [rad/s]

# GMST at J2000 epoch (2000 Jan 1.5 UT1) in radians
GMST_J2000 = 4.894_961_212_823_058  # ≈ 280.4606° in radians
