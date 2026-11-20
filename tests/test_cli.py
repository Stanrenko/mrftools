"""End-to-end tests of scripts/cli.py on a local raw data file.

Every command is run as a subprocess, in the order of the processing pipeline, on the smallest local acquisition
(1 slice, 16 channels, ~100 MB). The whole module takes about 40 s on CPU.

The raw data is not versioned (data/ is in .gitignore): the module is skipped when the file is missing. Another file
can be used by setting MRFTOOLS_TEST_DATFILE.

The CLI writes its outputs next to its input files, so the tests work in a temporary directory on a symlink to the
.dat file and never write into data/.
"""
import json
import os
import pathlib
import pickle
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLI = ROOT / "scripts" / "cli.py"
DATA = ROOT / "data"
DATFILE = pathlib.Path(os.environ.get("MRFTOOLS_TEST_DATFILE",
                                      DATA / "meas_MID00045_FID06215_MRF2D_bifurc_port_Sl1_2_BW934.dat"))
DICT_LIGHT_CONFIG = ROOT / "mrftools" / "dico_config" / "mrf_dictconf_Dico2_Invivo_light_for_matching.json"

pytestmark = pytest.mark.skipif(not DATFILE.is_file(), reason="local raw data file {} not found".format(DATFILE))

NTIMESTEPS = 175
L0 = 6


def run_cli(*args, cwd, check=True):
    result = subprocess.run([sys.executable, "-W", "ignore", str(CLI)] + [str(a) for a in args],
                            cwd=cwd, capture_output=True, text=True, timeout=600)
    if check and result.returncode != 0:
        raise AssertionError("cli.py {} failed:\n{}\n{}".format(" ".join(map(str, args)), result.stdout[-3000:],
                                                                 result.stderr[-3000:]))
    return result


def snapshot(directory):
    return {p.name: p.stat().st_mtime_ns for p in directory.iterdir()} if directory.is_dir() else {}


# ---------------------------------------------------------------------------
# Pipeline, one fixture per command (module scope: each command runs once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    work = tmp_path_factory.mktemp("cli")
    (work / "meas.dat").symlink_to(DATFILE.resolve())
    return work


@pytest.fixture(scope="module")
def data_dir_before():
    return snapshot(DATA)


@pytest.fixture(scope="module")
def extracted(workdir, data_dir_before):
    run_cli("extract_data", "--filename", workdir / "meas.dat", cwd=workdir)
    with open(workdir / "dico_seqParams.pkl", "rb") as f:
        seq_params = pickle.load(f)
    return np.load(workdir / "kdata.npy", mmap_mode="r"), seq_params


@pytest.fixture(scope="module")
def image_size(extracted):
    kdata, _ = extracted
    return (kdata.shape[-1] // 2, kdata.shape[-1] // 2)


@pytest.fixture(scope="module")
def sensi(workdir, extracted):
    run_cli("calculate_sensi", "--filekdata", workdir / "kdata.npy", cwd=workdir)
    return np.load(workdir / "b1.npy")


@pytest.fixture(scope="module")
def masks(workdir, sensi):
    run_cli("build_masks", "--filekdata", workdir / "kdata.npy", "--fileb1", workdir / "b1.npy", cwd=workdir)
    return np.load(workdir / "masks.npy")


@pytest.fixture(scope="module")
def volumes(workdir, sensi):
    run_cli("build_volumes", "--filekdata", workdir / "kdata.npy", "--fileb1", workdir / "b1.npy", cwd=workdir)
    return np.load(workdir / "volumes.npy", mmap_mode="r")


@pytest.fixture(scope="module")
def dict_config(workdir):
    """Small full dictionary grid (1650 entries) built from the light matching configuration."""
    with open(DICT_LIGHT_CONFIG) as f:
        config = json.load(f)
    config["water_T1"] = list(range(600, 2001, 100))
    config["B1_att"] = [round(0.3 + 0.1 * i, 1) for i in range(10)]
    path = workdir / "dictconf_small.json"
    with open(path, "w") as f:
        json.dump(config, f)
    return path, config


@pytest.fixture(scope="module")
def dico(workdir, extracted, dict_config):
    dictdir = workdir / "mrf_dict"
    result = run_cli("generate_dico", "--datafile", workdir / "meas.dat", "--dictdir", dictdir,
                     "--dictconf", dict_config[0], "--dictconflight", DICT_LIGHT_CONFIG, cwd=workdir)
    files = list(dictdir.glob("*.pkl"))
    assert len(files) == 1, files
    with open(files[0], "rb") as f:
        return dictdir, files[0], pickle.load(f), result.stdout


@pytest.fixture(scope="module")
def volumes_singular(workdir, sensi, dico):
    dictdir, dicofile, _, _ = dico
    run_cli("build_volumes_singular", "--filekdata", workdir / "kdata.npy", "--fileb1", workdir / "b1.npy",
            "--dictdir", dictdir, "--dictfiles", dicofile.name, "--L0", L0, cwd=workdir)
    return np.load(workdir / "volumes_singular.npy")


@pytest.fixture(scope="module")
def masks_singular(workdir, volumes_singular):
    run_cli("build_masks_singular", "--filevolumes", workdir / "volumes_singular.npy", cwd=workdir)
    return np.load(workdir / "masks_singular.npy")


@pytest.fixture(scope="module")
def maps(workdir, volumes, masks, dico):
    dictdir, dicofile, _, _ = dico
    run_cli("build_maps", "--filevolumes", workdir / "volumes.npy", "--filemasks", workdir / "masks.npy",
            "--fileseq", workdir / "dico_seqParams.pkl", "--dictdir", dictdir, "--dictfiles", dicofile.name,
            cwd=workdir)
    with open(workdir / "maps.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# extract_data
# ---------------------------------------------------------------------------


def test_extract_data_outputs(extracted):
    kdata, seq_params = extracted
    assert kdata.ndim == 4 and kdata.dtype == np.complex64  # slices x channels x spokes x npoint
    assert kdata.shape[2] == seq_params["nb_segments"] == 1400
    assert np.isfinite(kdata[0, 0]).all()
    for key in ["dTR", "TI", "spacing", "origin", "orientation", "offset", "is3D"]:
        assert key in seq_params
    assert 0.5 < seq_params["dTR"] < 3  # echo spacing in ms


def test_extract_data_applies_radial_density_adjustment(extracted):
    """The readout edges are weighted up and the k-space centre down, so energy peaks off-centre."""
    kdata, _ = extracted
    energy = np.mean(np.abs(kdata[0, :, :50]) ** 2, axis=(0, 1))
    npoint = kdata.shape[-1]
    assert energy[npoint // 2 - 1: npoint // 2 + 1].max() < energy.max()


# ---------------------------------------------------------------------------
# calculate_sensi, build_masks, build_volumes
# ---------------------------------------------------------------------------


def test_calculate_sensi_outputs(workdir, extracted, sensi, image_size):
    kdata, _ = extracted
    assert sensi.shape == kdata.shape[:2] + image_size
    rss = np.linalg.norm(sensi, axis=1)
    np.testing.assert_allclose(rss[rss > 0], 1.0, rtol=1e-3)
    assert (workdir / "b1.jpg").is_file()


def test_build_masks_outputs(workdir, extracted, masks, image_size):
    kdata, _ = extracted
    assert masks.shape == (kdata.shape[0],) + image_size
    assert masks.dtype == bool
    assert 0.05 < masks.mean() < 0.9
    assert (workdir / "masks.gif").is_file()


def test_build_volumes_outputs(workdir, extracted, volumes, masks, image_size):
    kdata, _ = extracted
    assert volumes.shape == (NTIMESTEPS, kdata.shape[0]) + image_size
    assert np.iscomplexobj(volumes)
    mean_image = np.abs(np.asarray(volumes[:, 0]).mean(axis=0))
    assert np.isfinite(mean_image).all()
    # signal is concentrated in the mask
    assert mean_image[masks[0]].mean() > 3 * mean_image[~masks[0]].mean()
    assert (workdir / "volumes.gif").is_file()


# ---------------------------------------------------------------------------
# generate_dico
# ---------------------------------------------------------------------------


def test_generate_dico_reads_sequence_timing_from_datafile(extracted, dico):
    _, seq_params = extracted
    _, dicofile, dictionary, stdout = dico
    echo_spacing = np.round(seq_params["dTR"], 2)
    assert dicofile.name == "dico_TR{}_reco5.0.pkl".format(echo_spacing)
    assert "Echo spacing of {} ms read in the .dat file".format(echo_spacing) in stdout

    seq = dictionary["hdr"]["sequence_config"]
    assert seq["TR"][0] - seq["TE"][0] == pytest.approx(echo_spacing)
    assert seq["TI"] == pytest.approx(np.round(seq_params["TI"], 2))


def test_generate_dico_uses_given_configurations(dico, dict_config):
    _, _, dictionary, _ = dico
    _, config = dict_config
    keys = np.array(dictionary["mrfdict"].keys)
    assert len(keys) == len(config["water_T1"]) * len(config["fat_T1"]) * len(config["B1_att"]) * len(config["delta_freqs"])
    assert sorted(set(keys[:, 0])) == config["water_T1"]
    with open(DICT_LIGHT_CONFIG) as f:
        light = json.load(f)
    assert sorted(set(np.array(dictionary["mrfdict_light"].keys)[:, 0])) == light["water_T1"]
    assert dictionary["mrfdict"].values.shape[1] == NTIMESTEPS


def test_generate_dico_refuses_non_empty_directory(dico, dict_config, workdir):
    dictdir, dicofile, _, _ = dico
    before = snapshot(dictdir)
    result = run_cli("generate_dico", "--datafile", workdir / "meas.dat", "--dictdir", dictdir,
                     "--dictconf", dict_config[0], "--dictconflight", DICT_LIGHT_CONFIG, cwd=workdir)
    assert "not emtpy. Aborting" in result.stdout
    assert snapshot(dictdir) == before


# ---------------------------------------------------------------------------
# build_volumes_singular, build_masks_singular
# ---------------------------------------------------------------------------


def test_build_volumes_singular_outputs(workdir, extracted, volumes_singular, masks, image_size):
    kdata, _ = extracted
    assert volumes_singular.shape == (L0, kdata.shape[0]) + image_size
    energy = np.linalg.norm(volumes_singular.reshape(L0, -1), axis=1)
    assert np.isfinite(energy).all()
    # the two leading temporal components carry most of the signal; which of the two dominates depends on the
    # tissue (water fingerprints project more on the second component of the water/fat mixture basis)
    assert energy[:2].min() > energy[2:].max()
    assert (workdir / "volumes_singular.gif").is_file()


def test_build_masks_singular_outputs(workdir, masks_singular, masks):
    assert masks_singular.shape == masks.shape
    assert set(np.unique(masks_singular)) <= {0, 1}
    assert 0.02 < masks_singular.mean() < 0.9
    # both masking methods find the same object
    overlap = (masks_singular.astype(bool) & masks).sum() / masks_singular.sum()
    assert overlap > 0.8
    assert (workdir / "masks_singular.gif").is_file()


# ---------------------------------------------------------------------------
# build_maps
# ---------------------------------------------------------------------------


def test_build_maps_outputs(maps, masks, dico):
    _, _, dictionary, _ = dico
    map_rebuilt, mask = maps[0][0], maps[0][1]
    np.testing.assert_array_equal(mask, masks)

    keys = np.array(dictionary["mrfdict"].keys)
    n_voxels = masks.sum()
    for i, name in enumerate(["wT1", "fT1", "attB1", "df"]):
        values = map_rebuilt[name]
        assert values.shape == (n_voxels,)
        # matched values are dictionary entries
        assert set(np.unique(values)) <= set(np.unique(keys[:, i])), name
    assert ((map_rebuilt["ff"] >= 0) & (map_rebuilt["ff"] <= 1)).all()
    # the maps are not degenerate
    assert len(np.unique(map_rebuilt["wT1"])) > 3
    assert 0 < np.mean(map_rebuilt["ff"]) < 1


def test_build_maps_writes_images_with_acquisition_geometry(workdir, maps, extracted, masks):
    _, seq_params = extracted
    for name in ["ff", "wT1", "attB1", "df"]:
        image = sitk.ReadImage(str(workdir / "{}_map.mha".format(name)))
        assert image.GetSpacing() == pytest.approx(seq_params["spacing"])
        origin = tuple(seq_params["origin"][:2]) + (seq_params["origin"][2] + seq_params["offset"],)
        assert image.GetOrigin() == pytest.approx(origin)
        array = sitk.GetArrayFromImage(image)
        assert array.size == masks.size
        assert np.count_nonzero(array) <= masks.sum()


def test_build_maps_return_cost(workdir, volumes, masks, dico):
    """--returncost adds the cost, phase and proton density maps (run in its own directory: maps are written next
    to --fileseq)."""
    dictdir, dicofile, _, _ = dico
    outdir = workdir / "returncost"
    outdir.mkdir()
    (outdir / "dico_seqParams.pkl").write_bytes((workdir / "dico_seqParams.pkl").read_bytes())
    run_cli("build_maps", "--filevolumes", workdir / "volumes.npy", "--filemasks", workdir / "masks.npy",
            "--fileseq", outdir / "dico_seqParams.pkl", "--dictdir", dictdir, "--dictfiles", dicofile.name,
            "--returncost", "True", cwd=workdir)
    with open(outdir / "maps.pkl", "rb") as f:
        all_maps = pickle.load(f)
    cost, phase, proton_density = all_maps[0][2:5]
    n_voxels = masks.sum()
    assert cost.shape == phase.shape == proton_density.shape == (n_voxels,)
    assert ((cost >= 0) & (cost <= 1 + 1e-6)).all()
    assert np.median(cost) > 0.5  # normalised correlation between signals and matched fingerprints


def test_build_maps_rejects_dictionary_with_other_echo_spacing(workdir, volumes, masks, dict_config, extracted):
    _, seq_params = extracted
    other_spacing = np.round(seq_params["dTR"], 2) + 0.2
    dictdir = workdir / "mrf_dict_other"
    run_cli("generate_dico", "--echospacing", other_spacing, "--dictdir", dictdir,
            "--dictconf", DICT_LIGHT_CONFIG, "--dictconflight", DICT_LIGHT_CONFIG, cwd=workdir)
    dicofile = next(dictdir.glob("*.pkl"))

    outdir = workdir / "other"
    outdir.mkdir()
    (outdir / "dico_seqParams.pkl").write_bytes((workdir / "dico_seqParams.pkl").read_bytes())
    result = run_cli("build_maps", "--filevolumes", workdir / "volumes.npy", "--filemasks", workdir / "masks.npy",
                     "--fileseq", outdir / "dico_seqParams.pkl", "--dictdir", dictdir, "--dictfiles", dicofile.name,
                     cwd=workdir, check=False)
    assert result.returncode != 0
    assert "does not match sequence echo spacing" in result.stderr
    assert not (outdir / "maps.pkl").exists()


# ---------------------------------------------------------------------------
# Side effects
# ---------------------------------------------------------------------------


def test_pipeline_does_not_write_into_data_directory(data_dir_before, extracted, maps, masks_singular):
    assert snapshot(DATA) == data_dir_before
