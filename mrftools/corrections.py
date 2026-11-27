"""Raw-data corrections for 2D radial raFin (MRF T1-FF) acquisitions.

Missing off-centre ADC phase
----------------------------
The raFin kernel demodulates each spoke at the frequency of the in-plane slice
off-centre, f(theta) = gamma*G*(offRO*cos(theta) + offPE*sin(theta)), and is
supposed to reset the phase at the k-space centre with the ADC phase
(dPhaseValuePos, ImgKernelRadial.cpp). In some exams this phase was not
applied (cause on the scanner side not identified, see
python_bmy/mrf_qc/README.md): the spokes then carry
    phi_n = -c * (offRO*cos(theta_n) + offPE*sin(theta_n)),
    c = 2*pi * t_c / (dwell * FOV_os)   [rad/mm],  t_c = npoint/2 * dwell
and the images only show streaks. Removing phi_n restores the data.

`offcentre_phase_correction` detects the defect per slice (coherence of the
k-space centre phase over the first spokes, with and without correction) and
corrects only when it improves coherence clearly (mode="auto").
"""
import numpy as np

from .geometry import inplane_offcentre

GOLDEN_ANGLE_DEG = 111.246  # raFin ImgKernelRadial.cpp


def _coherence(z):
    return float(np.abs(np.mean(z)))


def offcentre_phase(theta, offro, offpe, c):
    return -c * (offro * np.cos(theta) + offpe * np.sin(theta))


def offcentre_phase_correction(kdata, slice_geometry, dwell_s, fov_mm, base_resolution,
                               mode="auto", n_detect=600, threshold=0.2, verbose=True):
    """Detect / correct the missing off-centre ADC phase.

    kdata: (nb_slices, nb_channels, nb_spokes, npoint), spokes in acquisition
           order (spoke n acquired at angle n*golden angle), no density
           compensation required.
    slice_geometry: list from geometry.slice_geometry_from_mdbs.
    mode: "auto" (correct slices where it helps), "force", "off".
    Returns (kdata_corrected, report) - kdata is modified in place.
    """
    report = []
    if mode == "off":
        return kdata, report
    nsl, nch, nsp, npt = kdata.shape
    fov_os = fov_mm * npt / base_resolution
    c = 2 * np.pi * (npt / 2 * dwell_s) / (dwell_s * fov_os)
    theta = np.deg2rad(np.arange(nsp) * GOLDEN_ANGLE_DEG)
    nd = min(n_detect, nsp)
    for s in range(nsl):
        offro, offpe = inplane_offcentre(slice_geometry[s])
        phi = offcentre_phase(theta, offro, offpe, c)
        ch = int(np.argmax((np.abs(kdata[s, :, :nd, npt // 2]) ** 2).sum(-1)))
        z = np.exp(1j * np.angle(kdata[s, ch, :nd, npt // 2]))
        coh_raw = _coherence(z)
        coh_cor = _coherence(z * np.exp(-1j * phi[:nd]))
        apply = mode == "force" or (coh_cor - coh_raw > threshold)
        if apply:
            kdata[s] *= np.exp(-1j * phi)[None, :, None].astype(kdata.dtype)
        rep = dict(slice=s, offRO_mm=round(offro, 2), offPE_mm=round(offpe, 2),
                   amplitude_rad=round(float(c * np.hypot(offro, offpe)), 1),
                   coherence_raw=round(coh_raw, 3), coherence_corrected=round(coh_cor, 3),
                   corrected=bool(apply))
        report.append(rep)
        if verbose:
            print(f"off-centre phase slice {s}: off (RO,PE)=({offro:.1f},{offpe:.1f}) mm, "
                  f"coherence {coh_raw:.3f} -> {coh_cor:.3f}, corrected={apply}")
    return kdata, report
