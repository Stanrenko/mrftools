# coding=utf-8
""" read/write nibabel-compatible volumes (Niffti, Analyze, etc.) """

import numpy as np

# nibabel
import nibabel
from . import orientation

# so that nibabel recognizes format 132 (uint16)
_dtdefs = nibabel.analyze._dtdefs + ((132, "uint16", np.uint16),)
data_type_codes = nibabel.analyze.make_dt_codes(_dtdefs)
nibabel.analyze.AnalyzeHeader._data_type_codes = data_type_codes

SPM99_EXTENSIONS = [".hdr", ".img"]
NII_EXTENSIONS = [".nii", ".nii.gz"]
SPAR_EXTENSIONS = [".spar", ".sdat"]
FILE_EXTENSIONS = SPM99_EXTENSIONS + NII_EXTENSIONS + SPAR_EXTENSIONS


def load_volume(filepath, **kwargs):
    """load volume

    Accepted formats:
        formats:
            * ANALYZE_ (plain, SPM99, SPM2 and later)
            * GIFTI_, NIfTI1_, NIfTI2_, MINC1_, MINC2_, MGH_ and ECAT_
            * Philips PAR/REC
    """

    # load hdr
    filepath = str(filepath)
    niimage = nibabel.load(filepath)

    # raw volume
    data = np.asanyarray(niimage.dataobj)
    ndim = data.ndim

    # get affine transform
    affine = niimage.affine
    origin, spacing, rotation = orientation.decompose_affine_transform(affine)

    # invert x and y axes (Nifti assumes an LPI convention)
    origin *= np.array([-1, -1, 1])
    rotation *= np.array([[-1, -1, 1]])

    if ndim >= 4:
        time_offset = niimage.header["toffset"]
        time_spacing = niimage.header["pixdim"][4]
        _offset = np.stack([time_offset] + [0] * (ndim - 4))
        _spacing = np.stack([time_spacing] + [0] * (ndim - 4))
        origin = np.concatenate([origin, _offset])
        spacing = np.concatenate([spacing, _spacing])
        transform = np.eye(ndim)
        transform[:3, :3] = rotation
    else:
        origin = origin[:ndim]
        spacing = spacing[:ndim]
        transform = rotation[:ndim, :ndim]

    # rescale intercept, if defined
    try:
        intersect = niimage.header["scl_inter"]
        if not np.isnan(intersect):
            data = data - intersect
    except KeyError:
        pass

    # rescale slope, if defined
    try:
        slope = niimage.header["scl_slope"]
        if not np.isnan(slope):
            data = data / slope
    except:
        pass

    compression = filepath[-3:] == ".gz"
    header = {
        "origin": origin,
        "spacing": spacing,
        "transform": transform,
        "compression": compression,
    }
    return data, header


def save_volume(path, data, header):
    """write volume to file

    formats:
        * ANALYZE_ (plain, SPM99, SPM2 and later)
        * GIFTI_, NIfTI1_, NIfTI2_, MINC1_, MINC2_, MGH_ and ECAT_
        * Philips PAR/REC
    """
    data = np.asarray(data)
    ndim = data.ndim
    if ndim > 4:
        raise ValueError("Cannot store 5D+ array")

    spacing = header.get("spacing", (1,) * ndim)
    origin = header.get("origin", (0,) * ndim)
    transform = header.get("transform", np.eye(ndim))

    origin = np.asarray(origin)
    spacing = np.asarray(spacing)
    transform = np.asarray(transform)

    # create image
    image = nibabel.Nifti1Pair(data, None)
    image.default_x_flip = False
    # store pixdim and origin
    image.header["pixdim"][1 : ndim + 1] = list(spacing)
    if ndim == 4:
        image.header["toffset"] = origin[3]

    # fix
    if ndim < 3:
        origin = np.concatenate([origin, np.zeros(3 - ndim)])
        spacing = np.concatenate([spacing, np.ones(3 - ndim)])
        rotation = np.eye(3)
        rotation[:ndim, :ndim] = transform
    else:
        # crop dimensions
        origin = origin[:3]
        spacing = spacing[:3]
        rotation = transform[:3, :3]

    # invert x and y axes of origin (Nifti assumes an LPI convention)
    origin *= np.array([-1, -1, 1])
    rotation *= np.array([[-1, -1, 1]])

    # make affine transform matrix
    affine = orientation.make_affine_transform(
        origin=origin,
        spacing=spacing,
        rotation=rotation,
    )

    image.set_sform(affine)
    image.update_header()

    nibabel.save(image, path)
