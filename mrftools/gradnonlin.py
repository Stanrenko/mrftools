"""Gradient non-linearity (distortion) correction for 2D MRF maps.

Siemens images carry DIS2D / DIS3D corrections computed from the gradient coil
spherical-harmonic coefficients (coeff.grad). Offline MRF reconstructions do
not. This module:
  * reads a Siemens .grad file (Legendre coefficients A(n,m), B(n,m) per axis,
    radius R0) - e.g. python_bmy/config/coeff.grad (AS82, Prisma)
  * evaluates the non-linear displacement of each gradient axis (same
    expansion and Siemens normalization as gradunwarp, HCP):
        D_axis(r) = R0 * sum_n,m (r/R0)^n * Pnm(cos th) * (A cos m phi + B sin m phi)
    with Pnm the associated Legendre function without Condon-Shortley phase,
    Schmidt semi-normalized for m > 0. The linear terms are not in the file.
  * corrects 2D multi-slice maps IN-PLANE (as DIS2D): every output pixel at true
    position r takes the value of the distorted image at r + D(r) projected on
    the slice plane. The encoded position r + D(r) does not depend on the
    trajectory, so this also applies to radial MRF.

Coordinates: positions are Siemens patient coordinates (LPS) relative to the
isocentre (MDH positions, table offset excluded); they are converted to the
magnet (device) frame from the patient position (HFS, FFS, HFP, FFP).
"""
import math
import re

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import lpmv

from .geometry import READ_SIGN, PHASE_SIGN

# patient (LPS) -> device axes sign flips
PATIENT_TO_DEVICE = {"HFS": (1, 1, 1), "FFS": (-1, 1, -1), "HFP": (-1, -1, 1), "FFP": (1, -1, -1)}


def read_siemens_grad(path):
    """Return R0 (mm) and dict axis -> (alpha, beta) arrays (nmax+1, nmax+1)."""
    txt = open(path, encoding="latin1").read()
    r0 = float(re.search(r"([\d.]+)\s*m\s*=\s*R0", txt).group(1)) * 1000.0
    entries = re.findall(r"^\s*\d+\s+([AB])\(\s*(\d+),\s*(\d+)\)\s+([-+\d.eE]+)\s+([xyz])\s*$", txt, re.M)
    nmax = max(int(n) for _, n, _, _, _ in entries)
    coef = {ax: (np.zeros((nmax + 1, nmax + 1)), np.zeros((nmax + 1, nmax + 1))) for ax in "xyz"}
    for ab, n, m, v, ax in entries:
        coef[ax][0 if ab == "A" else 1][int(n), int(m)] = float(v)
    return r0, coef


def _harmonic_sum(alpha, beta, x, y, z, r0):
    x = x + 1e-4  # avoid r = 0
    r = np.sqrt(x * x + y * y + z * z)
    cth = np.clip(z / r, -1, 1)
    phi = np.arctan2(y, x)
    out = np.zeros_like(x, dtype=float)
    nmax = alpha.shape[0] - 1
    for n in range(nmax + 1):
        if not (np.any(alpha[n]) or np.any(beta[n])):
            continue
        f = (r / r0) ** n
        for m in range(n + 1):
            a, b = alpha[n, m], beta[n, m]
            if a == 0 and b == 0:
                continue
            p = lpmv(m, n, cth)
            if m > 0:
                p = p * (-1) ** m * math.sqrt((2 * n + 1) * math.factorial(n - m) / (2 * math.factorial(n + m)))
            out += f * p * (a * np.cos(m * phi) + b * np.sin(m * phi))
    return out


def displacement(points_lps, grad, patient_position="HFS"):
    """Non-linear displacement (mm, LPS) at positions (..., 3) in mm (LPS, isocentre-relative)."""
    r0, coef = grad
    s = np.array(PATIENT_TO_DEVICE[patient_position], float)
    p = points_lps * s
    d = np.stack([r0 * _harmonic_sum(*coef[ax], p[..., 0], p[..., 1], p[..., 2], r0) for ax in "xyz"], -1)
    return d * s


def correct_maps_2d(maps, slice_geometry, pixel_size, grad, patient_position="HFS", order=1, sign=1.0):
    """In-plane gradient non-linearity correction of 2D maps.

    sign: +1 (default, physical correction); -1 only for validation tests.

    maps: (nb_slices, N, N) in reconstruction layout (slice, k_x/read, k_y/phase),
          slices in the order of slice_geometry (geometry.slice_geometry_from_mdbs).
    order: interpolation order (1 linear for parameter maps, 0 for masks/labels).
    Returns corrected maps (same layout) and the in-plane displacement magnitude (mm).
    """
    maps = np.asarray(maps)
    nsl, n, _ = maps.shape
    idx = np.arange(n) - n / 2
    out = np.zeros_like(maps)
    dmag = np.zeros(maps.shape, float)
    for k, g in enumerate(slice_geometry):
        ux, uy = READ_SIGN * g["read"], PHASE_SIGN * g["phase"]
        c = g["position_mdh"]
        pts = (c[None, None, :] + idx[:, None, None] * pixel_size * ux[None, None, :]
               + idx[None, :, None] * pixel_size * uy[None, None, :])
        d = sign * displacement(pts, grad, patient_position)
        dx, dy = d @ ux, d @ uy                      # in-plane components (mm)
        src_i = idx[:, None] + dx / pixel_size + n / 2
        src_j = idx[None, :] + dy / pixel_size + n / 2
        src = np.stack([src_i, src_j])
        if np.iscomplexobj(maps):
            out[k] = (map_coordinates(maps[k].real, src, order=order, mode="constant")
                      + 1j * map_coordinates(maps[k].imag, src, order=order, mode="constant"))
        else:
            out[k] = map_coordinates(maps[k], src, order=order, mode="constant")
        dmag[k] = np.hypot(dx, dy)
    return out, dmag


def correct_masked_maps_2d(maps, mask, slice_geometry, pixel_size, grad, patient_position="HFS"):
    """Correction of parameter maps defined only inside `mask` (zeros outside):
    interpolates map*mask and mask, divides (no mixing with zeros at the
    border), keeps pixels whose interpolated mask > 0.5. Returns (maps, mask)."""
    m = np.asarray(mask, float)
    num, _ = correct_maps_2d(np.asarray(maps) * m, slice_geometry, pixel_size, grad, patient_position, order=1)
    den, _ = correct_maps_2d(m, slice_geometry, pixel_size, grad, patient_position, order=1)
    keep = den > 0.5
    out = np.zeros_like(num)
    out[keep] = num[keep] / den[keep]
    return out, keep
