"""Tests for the clustered two-step dictionary matching (SimpleDictSearch.search_patterns_test_multi_2_steps_dico).

The tests build a small synthetic bi-component (water / fat) dictionary, so they run on CPU in a few seconds and
do not need any acquired data or EPG simulation.
"""
import inspect
import itertools
import pickle

import numpy as np
import pytest

from mrftools import utils_mrf
from mrftools.dictmodel import Dictionary
from mrftools.utils_mrf import (
    DEFAULT_CLUSTERING_WINDOWS,
    PCAComplex,
    SimpleDictSearch,
    match_signals_v2_clustered_on_dico,
    select_keys_in_clustering_window,
)

PARAM_NAMES = ("wT1", "fT1", "attB1", "df")
NTIMESTEPS = 40
T = np.linspace(10, 3000, NTIMESTEPS)  # ms
# Near-lossless PCA: with this small synthetic dictionary, stronger truncation loses too much information for the
# matched keys to be exact, which would hide what these tests check (argument passing and window selection).
THRESHOLD_PCA = NTIMESTEPS - 1


FA_TRAIN = 0.2 + 1.2 * np.sin(np.pi * T / T[-1]) ** 2  # rad, varying flip angle train
TE_EFF = np.linspace(0, 20, NTIMESTEPS)  # ms, off-resonance phase stays below one cycle over the grid


def fingerprint(wT1, fT1, attB1, df):
    """Synthetic water and fat fingerprints. B1 changes the curve shape (flip angle train and inversion
    efficiency) so that it is identifiable after normalisation, like in a real MRF sequence."""
    readout = np.sin(attB1 * FA_TRAIN) * np.exp(1j * 2 * np.pi * df * TE_EFF)
    water = readout * (1 - (1 + attB1) * np.exp(-T / wT1))
    fat = readout * (1 - (1 + attB1) * np.exp(-T / fT1)) * np.exp(1j * 2 * np.pi * 0.05 * np.sqrt(T))
    return water, fat


def build_dictionary(wT1s, fT1s, attB1s, dfs):
    keys = list(itertools.product(wT1s, fT1s, attB1s, dfs))
    values = np.stack([np.stack(fingerprint(*k), axis=-1) for k in keys], axis=0)  # (n_entries, T, 2)
    return Dictionary(keys, values)


FULL_GRID = dict(
    wT1s=[700, 900, 1100, 1300, 1500, 1700],
    fT1s=[250, 300, 350, 400],
    attB1s=[0.6, 0.7, 0.8, 0.9, 1.0],
    dfs=[-0.04, -0.02, 0.0, 0.02, 0.04],
)
LIGHT_GRID = dict(wT1s=[700, 1100, 1500], fT1s=[250, 350], attB1s=[0.6, 0.8, 1.0], dfs=[-0.04, -0.02, 0.0, 0.02, 0.04])


@pytest.fixture
def dicofull_file(tmp_path):
    hdr = {"param_names": PARAM_NAMES}
    dicofull = {
        "mrfdict": build_dictionary(**FULL_GRID),
        "mrfdict_light": build_dictionary(**LIGHT_GRID),
        "hdr": dict(hdr),
        "hdr_light": dict(hdr),
    }
    path = tmp_path / "dico_test.pkl"
    with open(path, "wb") as f:
        pickle.dump(dicofull, f)
    return str(path)


# (wT1, fT1, attB1, df, ff): low-ff voxels (incl. pure water) and high-ff voxels (incl. pure fat)
TRUE_PARAMS = [
    (900, 300, 0.9, 0.02, 0.0),
    (1300, 350, 0.7, -0.02, 0.0),
    (1100, 300, 0.8, 0.0, 0.2),
    (1500, 400, 1.0, 0.04, 0.4),
    (700, 250, 0.9, -0.04, 0.6),
    (1100, 350, 0.8, 0.02, 0.95),
    (1300, 300, 0.7, 0.0, 1.0),
]


def make_signals(params=TRUE_PARAMS, scale=3.0, global_phase=0.7):
    signals = []
    for wT1, fT1, attB1, df, ff in params:
        w, f = fingerprint(wT1, fT1, attB1, df)
        signals.append(scale * np.exp(1j * global_phase) * ((1 - ff) * w + ff * f))
    return np.stack(signals, axis=1)  # (T, n_signals), i.e. already masked volumes


def run_search(dicofull_file, signals, return_cost=False, clustering_windows=DEFAULT_CLUSTERING_WINDOWS):
    optimizer = SimpleDictSearch(
        mask=np.ones(signals.shape[1]),
        split=100,
        pca=True,
        threshold_pca=THRESHOLD_PCA,
        threshold_ff=0.9,
        return_cost=return_cost,
        useGPU_dictsearch=False,
        volumes_type="raw",
        clustering_windows=clustering_windows,
    )
    return optimizer.search_patterns_test_multi_2_steps_dico(dicofull_file, signals)


# ---------------------------------------------------------------------------
# select_keys_in_clustering_window
# ---------------------------------------------------------------------------

KEYS = np.array(
    [
        [1000.0, 300.0, 1.0, 0.00],
        [1300.0, 300.0, 1.0, 0.00],  # wT1 outside +-200
        [1000.0, 300.0, 0.5, 0.00],  # attB1 outside +-0.2
        [1100.0, 350.0, 0.9, 0.01],  # inside every window
        [1000.0, 300.0, 1.0, 0.05],  # df outside +-0.03
    ]
)
CENTER = np.array([1000.0, 300.0, 1.0, 0.0])


def test_window_selects_entries_inside_all_windows():
    windows = {"wT1": 200, "fT1": 100, "attB1": 0.2, "df": 0.03}
    mask = select_keys_in_clustering_window(KEYS, CENTER, PARAM_NAMES, windows)
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, [True, False, False, True, False])


def test_window_bounds_are_strict():
    windows = {"wT1": 300}
    mask = select_keys_in_clustering_window(KEYS, CENTER, PARAM_NAMES, windows)
    # entry 1 is exactly at center + 300 -> excluded
    np.testing.assert_array_equal(mask, [True, False, True, True, True])


def test_window_parameters_without_window_are_unrestricted():
    mask = select_keys_in_clustering_window(KEYS, CENTER, PARAM_NAMES, {})
    assert mask.all()


def test_window_without_first_parameter():
    """Regression: the mask used to be initialised only when param_names[0] had a window, which raised
    NameError on the first cluster and silently reused the previous cluster's mask afterwards."""
    windows = {"attB1": 0.2, "df": 0.03}
    mask = select_keys_in_clustering_window(KEYS, CENTER, PARAM_NAMES, windows)
    np.testing.assert_array_equal(mask, [True, True, False, True, False])


def test_window_is_independent_between_calls():
    windows = {"attB1": 0.2}
    select_keys_in_clustering_window(KEYS, np.array([1000.0, 300.0, 0.5, 0.0]), PARAM_NAMES, windows)
    mask = select_keys_in_clustering_window(KEYS, CENTER, PARAM_NAMES, windows)
    np.testing.assert_array_equal(mask, [True, True, False, True, True])


# ---------------------------------------------------------------------------
# match_signals_v2_clustered_on_dico (direct calls)
# ---------------------------------------------------------------------------


def assert_keys_match(found, true_params):
    """Compare matched dictionary keys to the true ones. Fat T1 is not identifiable without fat, nor water T1
    without water."""
    *true_keys, ff = true_params
    for name, f, t in zip(PARAM_NAMES, found, true_keys):
        if (name == "fT1" and ff < 0.01) or (name == "wT1" and ff > 0.99):
            continue
        assert f == pytest.approx(t), (name, f, t)


def _matching_inputs(dico, signals, threshold_pca=THRESHOLD_PCA):
    keys = np.array(dico.keys)
    array_water, array_fat = dico.values[:, :, 0], dico.values[:, :, 1]
    w_u, idx_w = np.unique(array_water, axis=0, return_inverse=True)
    f_u, idx_f = np.unique(array_fat, axis=0, return_inverse=True)
    var_w = np.sum(w_u * w_u.conj(), axis=1).real[idx_w].reshape(-1, 1)
    var_f = np.sum(f_u * f_u.conj(), axis=1).real[idx_f].reshape(-1, 1)
    sig_wf = np.sum(w_u[idx_w] * f_u[idx_f].conj(), axis=1).real.reshape(-1, 1)
    pca_w, pca_f = PCAComplex(n_components_=threshold_pca), PCAComplex(n_components_=threshold_pca)
    pca_w.fit(w_u)
    pca_f.fit(f_u)
    return (signals.astype("complex64"), keys, pca_w, pca_f, pca_w.transform(w_u), pca_f.transform(f_u),
            var_w, var_f, sig_wf, idx_w, idx_f)


@pytest.mark.parametrize("return_cost", [False, True])
def test_clustered_matching_recovers_low_ff_signals(return_cost):
    dico = build_dictionary(**FULL_GRID)
    low_ff = [p for p in TRUE_PARAMS if p[-1] < 0.9]
    signals = make_signals(low_ff)
    inputs = _matching_inputs(dico, signals)
    keys = inputs[1]
    # one cluster per signal, centred on its true parameters
    unique_keys = np.array([p[:4] for p in low_ff], dtype=float).T
    labels = np.arange(len(low_ff))

    out = match_signals_v2_clustered_on_dico(*inputs, False, unique_keys, labels, 100, param_names=PARAM_NAMES,
                                             clustering_windows=DEFAULT_CLUSTERING_WINDOWS, high_ff=False,
                                             return_cost=return_cost)
    assert len(out) == (4 if return_cost else 2)
    idx, alpha = out[0], out[1]
    for j, p in enumerate(low_ff):
        assert_keys_match(keys[int(idx[j])], p)
        assert alpha[j] == pytest.approx(p[-1], abs=1e-3)
    if return_cost:
        np.testing.assert_allclose(out[2], 1.0, atol=1e-3)  # normalised correlation of a perfect match


# ---------------------------------------------------------------------------
# Full two-step search: regression for the argument mix-up in the callers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("return_cost", [False, True])
def test_two_step_search_passes_correct_flags(dicofull_file, monkeypatch, return_cost):
    """Regression: the low-ff call without return_cost passed clustering_windows twice positionally, so the
    windows dict ended up in high_ff (truthy) and low-ff voxels were matched with the high-ff logic."""
    calls = []
    original = utils_mrf.match_signals_v2_clustered_on_dico

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(utils_mrf, "match_signals_v2_clustered_on_dico", spy)
    windows = {"wT1": 400, "fT1": 100, "attB1": 0.2, "df": 0.03}
    run_search(dicofull_file, make_signals(), return_cost=return_cost, clustering_windows=windows)

    assert len(calls) == 2
    bound = [inspect.signature(original).bind(*a, **k).arguments for a, k in calls]
    for arguments in bound:
        assert arguments["clustering_windows"] is windows
        assert tuple(arguments["param_names"]) == PARAM_NAMES
        assert arguments.get("return_cost", False) is return_cost
    assert [a.get("high_ff", False) for a in bound] == [False, True]


@pytest.mark.parametrize("return_cost", [False, True])
def test_two_step_search_recovers_parameters(dicofull_file, return_cost):
    signals = make_signals()
    all_maps = run_search(dicofull_file, signals, return_cost=return_cost)
    map_rebuilt = all_maps[0][0]

    assert set(map_rebuilt) == set(PARAM_NAMES) | {"ff"}
    for j, p in enumerate(TRUE_PARAMS):
        assert map_rebuilt["ff"][j] == pytest.approx(p[-1], abs=1e-2), j
        assert_keys_match([map_rebuilt[k][j] for k in PARAM_NAMES], p)


def test_two_step_search_with_window_missing_first_param(dicofull_file):
    """Regression: clustering windows without the first parameter (wT1) used to raise NameError."""
    windows = {"fT1": 100, "attB1": 0.2, "df": 0.03}
    all_maps = run_search(dicofull_file, make_signals(), clustering_windows=windows)
    map_rebuilt = all_maps[0][0]
    for j, p in enumerate(TRUE_PARAMS):
        assert map_rebuilt["ff"][j] == pytest.approx(p[-1], abs=1e-2), j
        assert_keys_match([map_rebuilt[k][j] for k in PARAM_NAMES], p)
