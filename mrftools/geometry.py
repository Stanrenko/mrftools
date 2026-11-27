"""Slice geometry of 2D radial raFin (MRF T1-FF) acquisitions.

Geometry is taken from the MDH of the raw data (slice centre position and
rotation quaternion), as done by siemens_to_ismrmrd. All vectors are in the
Siemens patient coordinate system (LPS: +Sag = L, +Cor = P, +Tra = H), which is
also the physical space of ITK / SimpleITK, so they can be written directly to
.mha files.

Convention of the reconstructed images (finufft type 1 on the radial
trajectory of trajectory.Radial): first image axis <-> k_x (gradient on the
READOUT axis, cos(theta)), second image axis <-> k_y (gradient on the PHASE
axis, sin(theta)). Image pixel n of N maps to (n - N/2) * pixel_size from the
slice centre. Signs READ_SIGN / PHASE_SIGN were calibrated against the
scanner Dixon DICOM (see python_bmy/mrf_qc/validate_geometry.py).
"""
import numpy as np

# calibrated against scanner DICOM, 2026-11-17: fkr003.v1 thighs axial vs VIBE (NCC 0.93 vs <=0.79
# for other signs) and dmd001.v2 thighs oblique 12.4 deg vs Dixon (NCC 0.60 vs <=0.44)
READ_SIGN = -1.0
PHASE_SIGN = -1.0


def quaternion_to_directions(quat):
    """MDH SliceData.Quaternion -> (read, phase, slice) unit vectors (LPS).

    Same convention as siemens_to_ismrmrd: quaternion (q1, q2, q3, q0) passed to
    ismrmrd_quaternion_to_directions(quat, phase_dir, read_dir, slice_dir).
    """
    q0, q1, q2, q3 = [float(v) for v in quat]
    a, b, c, d = q1, q2, q3, q0
    phase = np.array([1 - 2 * (b * b + c * c), 2 * (a * b + c * d), 2 * (a * c - b * d)])
    read = np.array([2 * (a * b - c * d), 1 - 2 * (a * a + c * c), 2 * (b * c + a * d)])
    slc = np.array([2 * (a * c + b * d), 2 * (b * c - a * d), 1 - 2 * (a * a + b * b)])
    return read, phase, slc


def slice_geometry_from_mdbs(mdbs, table_offset_tra=0.0):
    """Per-slice geometry from a list of twixtools mdb (image scans).

    table_offset_tra: header lScanRegionPosTra (mm). MDH positions are relative
    to the table position; adding it gives the patient coordinates used in the
    scanner DICOM (ImagePositionPatient).

    Returns a list of dicts, in ORDER OF ACQUISITION (order of first appearance
    in the raw data, which is the slice order of the kdata array built by
    read_rawdata_2D), with 'cslc', 'position' (mm, LPS, table offset
    included), 'position_mdh' (mm, as in MDH, used for the ADC off-centre),
    'read', 'phase', 'slice' unit vectors.
    """
    geo = {}  # dicts keep insertion order
    for m in mdbs:
        s = m.cSlc
        if s in geo:
            continue
        sd = m.mdh.SliceData
        pos_mdh = np.array([sd.SlicePos.Sag, sd.SlicePos.Cor, sd.SlicePos.Tra], dtype=float)
        pos = pos_mdh + np.array([0.0, 0.0, float(table_offset_tra or 0.0)])
        read, phase, slc = quaternion_to_directions(list(sd.Quaternion))
        geo[s] = {"cslc": int(s), "position": pos, "position_mdh": pos_mdh,
                  "read": read, "phase": phase, "slice": slc}
    return list(geo.values())


def spatial_order(slice_geometry):
    """Indices sorting the slices by position along the slice normal."""
    n = slice_geometry[0]["slice"]
    return np.argsort([g["position"] @ n for g in slice_geometry], kind="stable")


def inplane_offcentre(slice_geo):
    """Off-centre of the slice centre along the logical READOUT and PHASE axes (mm),
    relative to the magnet isocentre (MDH position, table offset excluded)."""
    p = slice_geo["position_mdh"]
    return float(p @ slice_geo["read"]), float(p @ slice_geo["phase"])


def physical_grid(slice_geometry, npix, pixel_size):
    """Physical coordinates (mm, LPS) of every voxel of a raw reconstructed
    volume of shape (nb_slices, npix, npix) [slice, x(k_x/read), y(k_y/phase)].

    Returns an array of shape (3, nb_slices, npix, npix).
    """
    n = np.arange(npix) - npix / 2
    grid = np.zeros((3, len(slice_geometry), npix, npix))
    for k, g in enumerate(slice_geometry):
        ux = READ_SIGN * g["read"]
        uy = PHASE_SIGN * g["phase"]
        grid[:, k] = (g["position"][:, None, None]
                      + ux[:, None, None] * n[None, :, None] * pixel_size
                      + uy[:, None, None] * n[None, None, :] * pixel_size)
    return grid


def affine_from_coordinates(coords, atol=1e-2):
    """Fit ITK origin/spacing/direction to a coordinate array (3, nz, ny, nx)
    laid out like the numpy array given to sitk.GetImageFromArray.

    Returns dict(origin, spacing, direction (row-major 3x3 tuple), max_error_mm).
    """
    o = coords[:, 0, 0, 0]
    ei = coords[:, 0, 0, 1] - o if coords.shape[3] > 1 else np.zeros(3)
    ej = coords[:, 0, 1, 0] - o if coords.shape[2] > 1 else np.zeros(3)
    ek = coords[:, 1, 0, 0] - o if coords.shape[1] > 1 else None
    si, sj = np.linalg.norm(ei), np.linalg.norm(ej)
    di, dj = ei / si, ej / sj
    if ek is None:
        dk = np.cross(di, dj)
        sk = 1.0
    else:
        sk = np.linalg.norm(ek)
        dk = ek / sk
    D = np.stack([di, dj, dk], axis=1)  # columns = axis directions
    # check that the volume is an exact affine grid (equal slice spacing, parallel slices)
    nz, ny, nx = coords.shape[1:]
    kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    pred = (o[:, None, None, None]
            + D[:, 0, None, None, None] * ii * si
            + D[:, 1, None, None, None] * jj * sj
            + D[:, 2, None, None, None] * kk * sk)
    err = float(np.abs(pred - coords).max())
    if err > atol:
        print(f"WARNING geometry: volume is not an exact affine grid (max error {err:.2f} mm) - "
              "unequal slice spacing or non-parallel slices")
    return {"origin": tuple(o), "spacing": (si, sj, sk),
            "direction": tuple(D.flatten()), "max_error_mm": err}
