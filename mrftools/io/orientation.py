""" tools for managing the orientation of 3d volumes """

from . import volume
import numpy as np

ORIENTATIONS = [["R", "L"], ["A", "P"], ["I", "S"]]


def make_affine_transform(origin=None, spacing=None, rotation=None):
    """make transform matrix

    origin: a 3d vector
    spacing: a 3d vector
    rotation: a 3-sequence of 3d *columns*
        warning: when passing a 2d array, use the *transposed* rotation matrix

    """
    # source geometry
    if origin is None:
        origin = (0, 0, 0)
    if spacing is None:
        spacing = (1, 1, 1)
    if rotation is None:
        # assumes identify
        rotation = np.eye(3)

    origin = np.asarray(origin)
    spacing = np.asarray(spacing)

    # normalize rotation (assume it can be given as 1d array in column-major order)
    rotation = np.asarray(rotation).reshape(3, 3).T
    rotation = rotation / np.linalg.norm(rotation, axis=0, keepdims=True)

    # affine transform (rotation * scaling + translation)
    affine = np.r_[np.c_[rotation * spacing, origin], [[0, 0, 0, 1]]]

    return affine


def decompose_affine_transform(affine):
    """decompose affine transform into origin, spacing and rotation

    warning: rotation is the transposed of the rotation matrix
         ie. rotation can be seen as a sequence of *columns*
    """
    assert affine.shape == (4, 4)
    origin = affine[:-1, -1]
    spacing = np.linalg.norm(affine[:-1, :-1], axis=0)
    # transpose transform: return as sequence of columns
    transform = np.transpose(1 / spacing * affine[:-1, :-1])

    return origin, spacing, transform


#
# Orientation


def check_orientation(code):
    if len(code) != 3:
        raise ValueError(f"Invalid orientation code: {code}")
    if set(code.replace("L", "R").replace("P", "A").replace("S", "I")) != {
        "R",
        "A",
        "I",
    }:
        raise ValueError(f"Invalid orientation code: {code}")
    return True


def get_orientation(vol=None, *, transform=None):
    """get orientation from volume's tags"""
    if transform is None:
        transform = getattr(vol, "transform", np.eye(3))

    # assumes transform is a sequence of columns
    transform = np.asarray(transform).reshape(3, 3).T

    indices = np.argmax(np.abs(transform), axis=0)[:3]
    signs = 1 * (np.sign(transform[indices, np.arange(3)]) < 0)
    orient = "".join(ORIENTATIONS[i][s] for i, s in zip(indices, signs))

    check_orientation(orient)
    return orient


def set_orientation(vol, target, *, transform=None, copy=True):
    """Change volume/stack orientation"""

    check_orientation(target)
    if transform is None:
        transform = getattr(vol, "transform", np.eye(3))
    source = get_orientation(transform)

    vol = volume.asvolume(vol, transform=transform)
    if copy:
        vol = vol.copy()

    shape = vol.shape
    transform = vol.transform
    spacing = list(vol.spacing)
    center = [0, 0, 0]
    indices = [0, 1, 2]
    signs = [1, 1, 1]

    for i, code in enumerate(target):
        if code in source:
            index = source.find(code)
            sign = 1
        else:
            pair = [pair for pair in ORIENTATIONS if code in pair][0]
            alt = pair[1 - pair.index(code)]
            index = source.find(alt)
            sign = -1

        # invert direction and swap columns
        center[index] = (shape[index] - 1) if (sign < 0) else 0
        indices[i] = index
        signs[index] = sign

    # new origin and spacing
    origin = vol.get_coords(center)
    spacing = [spacing[i] for i in indices]

    # new transform
    transform = [[v * signs[i] for v in transform[i]] for i in indices]

    # flip and swap axes
    vol = np.flip(vol, axis=tuple(i for i in range(3) if signs[i] < 0))
    vol = np.moveaxis(vol, indices, [0, 1, 2])

    # set new values
    vol.origin = origin[:3]
    vol.transform = transform
    vol.spacing = spacing
    vol.info["orientation"] = target
    return vol
