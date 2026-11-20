"""Tests for the 2D MRF pipeline entry points in mrftools.main_functions_2D (the functions behind scripts/cli.py).

Reconstruction steps run on a small synthetic phantom whose radial k-space is simulated with finufft, and the
dictionary steps use a tiny EPG dictionary generated with the real sequence file, so everything runs on CPU in a
few seconds. Reading Siemens .dat files is not tested (needs acquired data); extract_data is tested with the reader
replaced by a stub.
"""
import glob
import inspect
import os
import pickle

import finufft
import numpy as np
import pytest
import SimpleITK as sitk

from mrftools import main_functions_2D as mf
from mrftools.trajectory import Radial
from mrftools.utils_mrf import build_phi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQUENCE_FILE = os.path.join(ROOT, "mrftools", "dico_config", "mrf_sequence_adjusted.json")

# ---------------------------------------------------------------------------
# Synthetic acquisition
# ---------------------------------------------------------------------------

NPOINT = 64  # readout points per spoke -> images of NPOINT/2 x NPOINT/2
NSPOKES = 80
NTIMESTEPS = 10
NB_SLICES = 2
NB_CHANNELS = 3
IMAGE_SIZE = (NPOINT // 2, NPOINT // 2)


def disk(radius=0.55, center=(0.0, 0.0)):
    y, x = np.meshgrid(np.linspace(-1, 1, IMAGE_SIZE[0]), np.linspace(-1, 1, IMAGE_SIZE[1]), indexing="ij")
    return ((x - center[1]) ** 2 + (y - center[0]) ** 2 < radius ** 2).astype(float)


def coil_maps():
    """Smooth complex coil sensitivities, one per channel."""
    y, x = np.meshgrid(np.linspace(-1, 1, IMAGE_SIZE[0]), np.linspace(-1, 1, IMAGE_SIZE[1]), indexing="ij")
    maps = []
    for c in range(NB_CHANNELS):
        angle = 2 * np.pi * c / NB_CHANNELS
        cy, cx = 1.5 * np.sin(angle), 1.5 * np.cos(angle)
        maps.append(np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 4) * np.exp(1j * angle))
    return np.array(maps)


def simulate_kdata(images):
    """Density-adjusted radial k-space (nb_slices x nb_channels x nb_spokes x npoint) on the package trajectory."""
    traj = Radial(total_nspokes=NSPOKES, npoint=NPOINT).get_traj().reshape(-1, 2)
    coils = coil_maps()
    kdata = np.zeros((NB_SLICES, NB_CHANNELS, NSPOKES, NPOINT), dtype=np.complex64)
    for sl, image in enumerate(images):
        coil_images = np.ascontiguousarray((coils * image).astype(np.complex128))
        k = finufft.nufft2d2(np.ascontiguousarray(traj[:, 0]), np.ascontiguousarray(traj[:, 1]), coil_images)
        kdata[sl] = k.reshape(NB_CHANNELS, NSPOKES, NPOINT)
    # radial density adjustment, as applied by extract_data(dens_adj=True) before every reconstruction step
    kdata *= np.abs(np.linspace(-1, 1, NPOINT))
    return kdata


@pytest.fixture(scope="module")
def phantoms():
    return np.array([disk(0.55), disk(0.4, center=(0.15, -0.1))])


@pytest.fixture
def kdata(phantoms):
    return simulate_kdata(phantoms)


def correlation(a, b):
    a, b = np.abs(a).ravel(), np.abs(b).ravel()
    return np.corrcoef(a, b)[0, 1]


# ---------------------------------------------------------------------------
# extract_data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dens_adj", [True, False])
def test_extract_data_density_adjustment(monkeypatch, dens_adj):
    raw = np.ones((1, 2, 3, 8), dtype=np.complex64)
    params = {"dTR": 1.11}
    monkeypatch.setattr(mf, "read_rawdata_2D", lambda filename: (raw.copy(), params))

    data, seq_params = mf.extract_data("dummy.dat", dens_adj=dens_adj)

    assert seq_params is params
    assert data.shape == raw.shape
    expected = np.abs(np.linspace(-1, 1, 8)) if dens_adj else np.ones(8)
    np.testing.assert_allclose(data[0, 1, 2], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Reconstruction: sensitivity maps, masks, volumes
# ---------------------------------------------------------------------------


def test_calculate_sensitivity_map_shape_and_normalisation(kdata):
    b1 = mf.calculate_sensitivity_map(kdata.copy(), res=16)

    assert b1.shape == (NB_SLICES, NB_CHANNELS) + IMAGE_SIZE
    # maps are normalised across channels: root-sum-of-squares is 1 wherever there is signal
    rss = np.linalg.norm(b1, axis=1)
    np.testing.assert_allclose(rss[np.isfinite(rss) & (rss > 0)], 1.0, rtol=1e-4)


def test_calculate_sensitivity_map_does_not_modify_input_without_density_adj(kdata):
    original = kdata.copy()
    mf.calculate_sensitivity_map(kdata, res=16, density_adj=False)
    np.testing.assert_array_equal(kdata, original)


def test_calculate_sensitivity_map_follows_coil_phase(kdata):
    """Relative phase between channels of the estimated maps matches the simulated coils inside the object."""
    b1 = mf.calculate_sensitivity_map(kdata.copy(), res=32, hanning_filter=False)
    coils = coil_maps()
    inside = disk(0.3) > 0
    for c in range(1, NB_CHANNELS):
        est = np.angle(b1[0, c][inside] * b1[0, 0][inside].conj())
        true = np.angle(coils[c][inside] * coils[0][inside].conj())
        assert np.median(np.abs(np.angle(np.exp(1j * (est - true))))) < 0.3


@pytest.fixture
def b1(kdata):
    return mf.calculate_sensitivity_map(kdata.copy(), res=16)


PHANTOM_DISKS = [(0.55, (0.0, 0.0)), (0.4, (0.15, -0.1))]  # (radius, center) per slice


def test_build_masks_matches_phantom(kdata, b1):
    masks = mf.build_masks(kdata.copy(), b1)

    assert masks.shape == (NB_SLICES,) + IMAGE_SIZE
    for sl, (radius, center) in enumerate(PHANTOM_DISKS):
        inside = disk(radius - 0.1, center) > 0
        far_outside = disk(radius + 0.25, center) == 0
        assert masks[sl][inside].all()
        # the default threshold (1/25 of the histogram range) keeps some streaks around the object, not the background
        assert masks[sl][far_outside].mean() < 0.3
        assert not masks[sl][0, 0] and not masks[sl][-1, -1]


def test_build_volumes_shape_and_content(kdata, b1, phantoms):
    volumes = mf.build_volumes(kdata.copy(), b1, ntimesteps=NTIMESTEPS)

    assert volumes.shape == (NTIMESTEPS, NB_SLICES) + IMAGE_SIZE
    assert np.iscomplexobj(volumes)
    for sl in range(NB_SLICES):
        # each timestep is undersampled; the mean over the series recovers the static object
        assert correlation(volumes[:, sl].mean(axis=0), phantoms[sl]) > 0.9


def test_build_volumes_singular_shape_and_content(kdata, b1, phantoms):
    L0 = 2
    # orthonormal temporal basis: constant + linear ramp
    t = np.arange(NTIMESTEPS)
    phi = np.stack([np.ones(NTIMESTEPS), t - t.mean()]).astype(np.complex64)
    phi /= np.linalg.norm(phi, axis=1, keepdims=True)

    volumes = mf.build_volumes_singular(kdata.copy(), b1, phi, L0=L0)

    assert volumes.shape == (L0, NB_SLICES) + IMAGE_SIZE
    for sl in range(NB_SLICES):
        # static object: the energy is on the constant component; the ramp component only gets undersampling aliasing
        assert correlation(volumes[0, sl], phantoms[sl]) > 0.9
        assert np.linalg.norm(volumes[1, sl]) < 0.15 * np.linalg.norm(volumes[0, sl])


def test_build_volumes_singular_iterative_without_iterations_equals_singular(kdata, b1):
    phi = np.ones((1, NTIMESTEPS), dtype=np.complex64) / np.sqrt(NTIMESTEPS)
    direct = mf.build_volumes_singular(kdata.copy(), b1, phi, L0=1)
    iterative = mf.build_volumes_singular_iterative(kdata.copy(), b1, phi, L0=1, niter=0)
    # identical up to float32 NUFFT round-off
    np.testing.assert_allclose(iterative, direct, rtol=0, atol=1e-5 * np.abs(direct).max())


@pytest.mark.parametrize("regularizer, niter", [("unknown", 0), (None, 2)])
def test_build_volumes_singular_iterative_rejects_invalid_regularizer(kdata, b1, regularizer, niter):
    phi = np.ones((1, NTIMESTEPS), dtype=np.complex64)
    with pytest.raises(ValueError):
        mf.build_volumes_singular_iterative(kdata, b1, phi, L0=1, niter=niter, regularizer=regularizer)


def test_build_mask_from_singular_volume(phantoms):
    rng = np.random.default_rng(0)
    volumes = np.stack([np.zeros_like(phantoms), phantoms + 0.01 * rng.standard_normal(phantoms.shape)])

    mask = mf.build_mask_from_singular_volume(volumes, l=1, threshold=0.5, it=0)
    assert mask.shape == phantoms.shape
    np.testing.assert_array_equal(mask, phantoms.astype(int))

    # binary closing only adds pixels
    closed = mf.build_mask_from_singular_volume(volumes, l=1, threshold=0.5, it=1)
    assert closed[phantoms > 0].all()
    assert closed.sum() >= mask.sum()


# ---------------------------------------------------------------------------
# check_dico
# ---------------------------------------------------------------------------

DICO_HDR = {"sequence_config": {"TR": [3.50, 3.50], "TE": [2.39, 2.39]}}  # echo spacing 1.11 ms


def test_check_dico_accepts_matching_echo_spacing(capsys):
    mf.check_dico(DICO_HDR, {"dTR": 1.115})
    assert "Dictionary OK" in capsys.readouterr().out


def test_check_dico_reads_pickled_sequence_parameters(tmp_path):
    file_seq = tmp_path / "dico_seqParams.pkl"
    with open(file_seq, "wb") as f:
        pickle.dump({"dTR": 1.11}, f)
    mf.check_dico(DICO_HDR, str(file_seq))


def test_check_dico_rejects_mismatching_echo_spacing():
    with pytest.raises(ValueError, match=r"dictionary \(1.11 ms\).*sequence echo spacing \(1.5 ms\)"):
        mf.check_dico(DICO_HDR, {"dTR": 1.5})


def test_check_dico_rejects_invalid_input_type():
    with pytest.raises(ValueError):
        mf.check_dico(DICO_HDR, 1.11)


# ---------------------------------------------------------------------------
# generate_dictionaries -> build_maps -> save_maps
# ---------------------------------------------------------------------------

FAT = {"fat_T1": [300.0], "water_T2": 40.0, "fat_T2": 80.0,
       "fat_amp": [0.0586, 0.0109, 0.0618, 0.1412, 0.66, 0.0673],
       "fat_cshift": [-101.1, 208.3, 281.0, 305.7, 395.6, 446.2]}
DICT_CONF = dict(FAT, window_size=8, water_T1=[700, 900, 1100, 1300, 1500], B1_att=[0.6, 0.8, 1.0],
                 delta_freqs=[-30.0, 0.0, 30.0])
# light grid steps match the clustering windows of the two-step matching (wT1 +-400 ms, B1 +-0.2, strict bounds)
DICT_CONF_LIGHT = dict(FAT, window_size=8, water_T1=[700, 1100, 1500], B1_att=[0.6, 0.8, 1.0], delta_freqs=[-30.0, 0.0, 30.0])
MIN_TR_DELAY = 1.11  # ms
RECO = 5.0  # s


@pytest.fixture(scope="module")
def dico_dir(tmp_path_factory):
    dest = tmp_path_factory.mktemp("mrf_dict")
    mf.generate_dictionaries(SEQUENCE_FILE, RECO, MIN_TR_DELAY, DICT_CONF, DICT_CONF_LIGHT, dest=str(dest))
    return dest


@pytest.fixture(scope="module")
def dico_file(dico_dir):
    return str(dico_dir / "dico_TR1.11_reco5.0.pkl")


@pytest.fixture(scope="module")
def dico(dico_file):
    with open(dico_file, "rb") as f:
        return pickle.load(f)


def test_generate_dictionaries_writes_single_pickle(dico_dir, dico_file):
    assert glob.glob(str(dico_dir / "*")) == [dico_file]


def test_generate_dictionaries_content(dico):
    assert {"hdr", "hdr_light", "mrfdict", "mrfdict_light"} <= set(dico)

    n_full = len(DICT_CONF["water_T1"]) * len(DICT_CONF["fat_T1"]) * len(DICT_CONF["B1_att"]) * len(DICT_CONF["delta_freqs"])
    n_light = len(DICT_CONF_LIGHT["water_T1"]) * len(DICT_CONF_LIGHT["B1_att"]) * len(DICT_CONF_LIGHT["delta_freqs"])
    ntimesteps = 1400 // DICT_CONF["window_size"]
    assert dico["mrfdict"].values.shape == (n_full, ntimesteps, 2)
    assert dico["mrfdict_light"].values.shape == (n_light, ntimesteps, 2)

    keys = np.array(dico["mrfdict"].keys)
    assert sorted(set(keys[:, 0])) == DICT_CONF["water_T1"]
    assert sorted(set(keys[:, 2])) == DICT_CONF["B1_att"]
    # delta_freqs are given in Hz and stored in kHz with the opposite sign
    assert sorted(set(keys[:, 3])) == pytest.approx(sorted(-f / 1000 for f in DICT_CONF["delta_freqs"]))

    seq = dico["hdr"]["sequence_config"]
    assert len(seq["TE"]) == 1400
    assert seq["TR"][0] - seq["TE"][0] == pytest.approx(MIN_TR_DELAY)
    assert seq["T_recovery"] == RECO * 1000


def test_generate_dictionaries_header_passes_check_dico(dico):
    mf.check_dico(dico["hdr"], {"dTR": MIN_TR_DELAY})
    with pytest.raises(ValueError):
        mf.check_dico(dico["hdr"], {"dTR": 2.0})


def test_generated_fingerprints_depend_on_parameters(dico):
    values = dico["mrfdict"].values
    assert np.all(np.isfinite(values))
    water = values[:, :, 0]
    norm = water / np.linalg.norm(water, axis=1, keepdims=True)
    gram = np.abs(norm @ norm.conj().T)
    np.fill_diagonal(gram, 0)
    assert gram.max() < 0.9999  # no two dictionary entries are identical up to scale


def test_build_phi_covers_pure_water_and_pure_fat(dico):
    mrfdict = dico["mrfdict"]
    phi = build_phi(mrfdict)
    ntimesteps = mrfdict.values.shape[1]
    assert phi.shape == (ntimesteps, ntimesteps)
    np.testing.assert_allclose(phi @ phi.conj().T, np.eye(ntimesteps), atol=1e-6)  # orthonormal rows

    # the default fat fractions span the whole [0, 1] range used for matching, pure water included
    default_ffs = inspect.signature(build_phi).parameters["FFs"].default
    assert default_ffs[0] == 0.0 and default_ffs[-1] == pytest.approx(1.0)

    # pure water and pure fat fingerprints are well represented by the first components
    basis = phi[:6]
    for component in (0, 1):
        x = mrfdict.values[:, :, component]
        residual = np.linalg.norm(x - (x @ basis.T.conj()) @ basis, axis=1) / np.linalg.norm(x, axis=1)
        assert residual.mean() < 0.1, component


def test_generate_dictionaries_with_temporal_basis(tmp_path):
    L0 = 4
    mf.generate_dictionaries(SEQUENCE_FILE, RECO, MIN_TR_DELAY, DICT_CONF_LIGHT, DICT_CONF_LIGHT,
                             dest=str(tmp_path), is_build_phi=True, L0=L0)
    with open(tmp_path / "dico_TR1.11_reco5.0.pkl", "rb") as f:
        dico = pickle.load(f)

    ntimesteps = 1400 // DICT_CONF_LIGHT["window_size"]
    assert dico["phi"].shape[1] == ntimesteps
    water, fat, keys = dico["mrfdict_L0{}".format(L0)]
    assert water.shape == fat.shape == (len(dico["mrfdict"].keys), L0)
    assert len(keys) == len(dico["mrfdict"].keys)
    assert "mrfdict_light_L0{}".format(L0) in dico


# (index of the dictionary entry, fat fraction) for each simulated voxel; at least two voxels in each fat-fraction
# group (threshold 0.9) of the two-step matching
VOXELS = [(0, 0.0), (7, 0.1), (16, 0.3), (24, 0.5), (31, 0.7), (40, 0.95), (12, 1.0)]


FF_TOLERANCE = 0.03  # fat-fraction error from the 6-component PCA compression used for matching


def simulate_voxels(dico, voxels=VOXELS, scale=5.0, phase=0.4):
    values = dico["mrfdict"].values
    return np.stack([scale * np.exp(1j * phase) * ((1 - ff) * values[i, :, 0] + ff * values[i, :, 1])
                     for i, ff in voxels], axis=1)


@pytest.fixture(scope="module")
def maps(dico, dico_file):
    signals = simulate_voxels(dico)
    ntimesteps = signals.shape[0]
    # place the voxels in a 1 x 3 x 3 image with a mask
    mask = np.zeros((1, 3, 3), dtype=int)
    mask.flat[: len(VOXELS)] = 1
    volumes = np.zeros((ntimesteps,) + mask.shape, dtype=np.complex64)
    volumes[:, mask > 0] = signals
    return mf.build_maps(volumes, mask, dico_file, useGPU=False, split=100, pca=6), mask


def test_build_maps_recovers_simulated_parameters(dico, maps):
    all_maps, _ = maps
    map_rebuilt = all_maps[0][0]
    keys = np.array(dico["mrfdict"].keys)

    assert set(map_rebuilt) == {"wT1", "fT1", "attB1", "df", "ff"}
    for j, (i, ff) in enumerate(VOXELS):
        assert map_rebuilt["ff"][j] == pytest.approx(ff, abs=FF_TOLERANCE), j
        if ff < 0.9:  # water T1 is not identifiable in fat-dominated voxels (high-ff group of the matching)
            assert map_rebuilt["wT1"][j] == pytest.approx(keys[i, 0]), j
        assert map_rebuilt["attB1"][j] == pytest.approx(keys[i, 2]), j
        assert map_rebuilt["df"][j] == pytest.approx(keys[i, 3]), j


def test_build_maps_with_cost(dico, dico_file):
    signals = simulate_voxels(dico)
    all_maps = mf.build_maps(signals, np.ones(signals.shape[1]), dico_file, useGPU=False, pca=6, return_cost=True)
    map_rebuilt, mask, cost, phase, proton_density = all_maps[0][:5]
    assert cost.shape == phase.shape == proton_density.shape == (len(VOXELS),)
    np.testing.assert_allclose(cost, 1.0, atol=0.02)  # normalised correlation of near-perfect matches
    np.testing.assert_allclose(map_rebuilt["ff"], [ff for _, ff in VOXELS], atol=FF_TOLERANCE)


SEQ_PARAMS = {"dTR": MIN_TR_DELAY, "spacing": (1.5, 1.5, 5.0), "origin": (-10.0, -20.0, 30.0),
              "orientation": "transversal", "offset": 2.0, "is3D": False}


def test_save_maps_writes_pickle_and_images(tmp_path, maps):
    all_maps, mask = maps
    file_seq = tmp_path / "dico_seqParams.pkl"
    with open(file_seq, "wb") as f:
        pickle.dump(SEQ_PARAMS, f)

    mf.save_maps(all_maps, str(file_seq))

    with open(tmp_path / "maps.pkl", "rb") as f:
        saved = pickle.load(f)
    np.testing.assert_array_equal(saved[0][0]["ff"], all_maps[0][0]["ff"])

    for key in ["ff", "wT1", "attB1", "df"]:
        image = sitk.ReadImage(str(tmp_path / "{}_map.mha".format(key)))
        assert image.GetSpacing() == pytest.approx(SEQ_PARAMS["spacing"])
        expected_origin = SEQ_PARAMS["origin"][:2] + (SEQ_PARAMS["origin"][2] + SEQ_PARAMS["offset"],)
        assert image.GetOrigin() == pytest.approx(expected_origin)
        array = sitk.GetArrayFromImage(image)
        assert array.size == mask.size
        # all values of the map are written, background is zero
        np.testing.assert_allclose(np.sort(array[array != 0]), np.sort(all_maps[0][0][key][all_maps[0][0][key] != 0]))
