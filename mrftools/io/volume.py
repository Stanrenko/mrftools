# coding=utf-8
""" module defining Volume and volume i/o """

import os
import numpy as np
from pathlib import Path

from dicomstack import DicomStack
from . import metaimage

# default value for compression flag
COMPRESSION_DEFAULT = True
DEFAULT_FORMAT = ".mha"

FILE_EXTENSIONS = metaimage.FILE_EXTENSIONS

try:
    from . import io_nibabel

    is_nibabel = True
    FILE_EXTENSIONS = FILE_EXTENSIONS + io_nibabel.FILE_EXTENSIONS
except ImportError:
    print(
        "Warning: could not import nibabel. \
       I/O for Analyze (.hdr/.img) or Nifti (.nii) is disabled"
    )
    is_nibabel = False


def dicomread(path, by=None, **filters):
    """convenience function for DICOM to volume"""
    stack = DicomStack(path, **filters)
    if by:
        header, vols = stack.as_volume(by=by)
        return header, [Volume(vol) for vol in vols]
    return Volume(stack.as_volume())


def read(
    filename, nan_as=None, default=..., astype=None, cls=None, tocomplex=False, **kwargs
):
    """read volume file

    Arguments:
    ---
        filename: path to volume
        default: any value.
            If provided, will return default instead of
             IOError in case of missing file
        nan_as: any value
            Convert maching values in data array to NaN
        default_format: file extension
            default file format for reading if ext is not known

    Returns:
    ---
        volume: Volume array (subtype of ndarray)

    """
    # parameters
    default_format = kwargs.pop("default_format", DEFAULT_FORMAT)

    filename = str(filename)

    # default value
    if not os.path.exists(filename):
        if default is Ellipsis:
            raise IOError('Could not find file at: "%s"' % filename)
        else:
            return default

    # file format
    for ext in FILE_EXTENSIONS:
        if filename.endswith(ext):
            break
    else:
        ext = os.path.splitext(filename)[-1]
        if not ext:
            ext = default_format
        else:
            raise ValueError(f"Unknown volume's extension: {ext}")

    if ext in metaimage.FILE_EXTENSIONS:
        # metaimage
        data, header = metaimage.read(filename, **kwargs)
        nchannel = header.get("ElementNumberOfChannels", 1)
        if tocomplex and nchannel == 2:
            # automatically convert to complex
            data = data[..., 0] + 1j * data[..., 1]
            nchannel = 1
        ndim = data.ndim - 1 if nchannel > 1 else data.ndim

        # get metadata from header
        transform = header.get("TransformMatrix")
        meta = {
            "origin": header.get("Offset"),
            "spacing": header.get("ElementSpacing"),
            "transform": tuple(zip(*[transform[i::ndim] for i in range(ndim)])),
            "info": {
                "compression": header["CompressedData"],
                "orientation": header.get("AnatomicalOrientation", "RAI"),
            },
        }
        if data.ndim > ndim:
            meta = resize_tags(data.ndim, meta)

        if not cls:
            # make Stack/Volume object
            cls = Volume if (ndim == 3 and nchannel <= 1) else Stack
        vol = cls(data, **meta)

    elif is_nibabel and ext in io_nibabel.FILE_EXTENSIONS:
        # .nii, .hdr
        data, header = io_nibabel.load_volume(filename, **kwargs)
        meta = {
            "origin": header["origin"],
            "spacing": header["spacing"],
            "transform": header["transform"],
        }
        info = {"compression": header.get("compression", False)}

        if not cls:
            # make Stack/Volume object
            cls = Volume if data.ndim == 3 else Stack
        vol = cls(data, info=info, **meta)

    else:
        raise ValueError("Unknown file extension: %s" % ext)

    # handle nans
    if (
        np.issubdtype(vol.dtype, np.floating)
        and nan_as is not None
        and np.isclose(vol.min(), nan_as)
    ):
        vol[np.isclose(vol, nan_as)] = np.nan

    if astype:
        return vol.astype(astype)

    return vol


def readfirst(dir, names, exts=None, default=..., **kwargs):
    """load first existing volume among several in same folder"""
    for name in names:
        path = Path(dir) / name
        vol = read(path, default=None, **kwargs)
        if vol is not None:
            return vol
    else:
        if default is ...:
            raise IOError(f"Could find none of {names} in {dir}")
        return default


def readfirstx(dir, name, exts, default=..., **kwargs):
    """load first matching volume for several extensions in same folder"""
    for ext in exts:
        path = (Path(dir) / name).with_suffix(ext)
        vol = read(path, default=None, **kwargs)
        if vol is not None:
            return vol
    else:
        if default is ...:
            raise IOError(f"Could find no matching volumes for {name}+{exts} in {dir}")
        return default


def write(filename, vol, tags=None, nan_as=0, cls=None, isvector=None, **kwargs):
    """write volume"""
    filename = Path(filename)

    # parameters
    default_format = kwargs.pop("default_format", DEFAULT_FORMAT)
    compression = kwargs.pop("compression", COMPRESSION_DEFAULT)

    if not tags:
        tags = {}

    # convert to stack
    vol = Stack(vol, **tags)

    # auto 4d scalar to 3d vector ?
    if vol.ndim == 4 and isvector is None:
        isvector = True

    # fix integer int64 case
    if vol.dtype == "int64":
        vol = vol.astype("int32")
    elif vol.dtype == "uint64":
        vol = vol.astype("uint32")

    # is complex ?
    iscomplex = np.iscomplexobj(vol)

    # handle nans
    vol[~np.isfinite(vol)] = nan_as

    # file format
    extensions = filename.suffixes
    if not extensions:
        ext = default_format
        filename = filename.with_suffix(ext)
    else:
        ext = "".join(extensions)

    if ext in metaimage.FILE_EXTENSIONS:
        # metaimage
        ndim = vol.ndim - 1 * (isvector == True)
        opts = {
            # MetaIO tags
            "Offset": vol.origin[:ndim],
            "ElementSpacing": vol.spacing[:ndim],
            # transform is stored in column major order
            "TransformMatrix": [
                value for vec in vol.transform[:ndim] for value in vec[:ndim]
            ],
            "CompressedData": compression,
            "AnatomicalOrientation": vol.info.get("orientation", "RAI"),
            # other
            "isvector": isvector,
            "ignore_errors": kwargs.get("ignore_errors", False),
        }

        if iscomplex:
            if isvector:
                raise ValueError("Cannot store complex vector data")
            # add new dimension
            vol = np.stack([vol.A.real, vol.A.imag], axis=-1)
            opts["isvector"] = True

        metaimage.write(filename, vol, **opts)

    elif is_nibabel and ext in io_nibabel.FILE_EXTENSIONS:
        header = {
            "spacing": vol.spacing,
            "origin": vol.origin,
            "transform": vol.transform,
        }
        io_nibabel.save_volume(filename, vol, header)

    else:
        raise ValueError("Unknown file extension: %s" % ext)


def asvolume(obj, **kwargs):
    """helper function similar to np.asarray

    Parameters: dtype, spacing, origin, transform, info
    """
    if isinstance(obj, Volume):
        if kwargs.get("dtype") and np.dtype(kwargs["dtype"]) != obj.dtype:
            pass
        elif kwargs.get("spacing") and tuple(kwargs["spacing"]) != obj.spacing:
            pass
        elif kwargs.get("origin") and tuple(kwargs["origin"]) != obj.origin:
            pass
        elif (
            kwargs.get("transform")
            and tuple(tuple(v) for v in kwargs["transform"]) != obj.transform
        ):
            pass
        elif kwargs.get("info") and dict(kwargs["info"]) != obj.info:
            pass
        else:
            # obj is the same
            return obj

    # copy if needed
    if not isinstance(obj, np.ndarray):
        obj = np.asarray(obj)

    # resize tags
    kwargs = resize_tags(obj.ndim, kwargs)

    cls = Volume if obj.ndim == 3 else Stack
    return cls(obj, **kwargs)


def stack(objs, *, axis=0, **kwargs):
    """like np.stack, but return Stack object"""
    # stack arrays
    arr = np.stack(objs, axis=axis)

    # get metadata
    ndim = np.ndim(objs[0])
    meta = {**getattr(objs[0], "meta", {}), **getattr(objs[0], "tags", {})}

    default = default_tags(ndim)
    tags = {
        "origin": meta.get("origin", kwargs.get("origin", default["origin"])),
        "spacing": meta.get("spacing", kwargs.get("spacing", default["spacing"])),
        "transform": meta.get(
            "transform", kwargs.get("transform", default["transform"])
        ),
        "info": meta.get("info", kwargs.get("info")),
    }
    if ndim != arr.ndim:
        tags = resize_tags(arr.ndim, tags)
    return Stack(arr, **tags)


def tovolume(values, mask=None, *, ref=None, **kwargs):
    """create and fill Volume at mask values, optionally using ref's metadata"""
    if mask is None and ref is None:
        return asvolume(values, **kwargs)

    if not isinstance(values, np.ndarray):
        values = np.asarray(values)
    if mask is not None and not isinstance(mask, np.ndarray):
        mask = np.asarray(mask) > 0
    if ref is not None and not isinstance(ref, np.ndarray):
        ref = np.asarray(ref)

    meta = {**getattr(mask, "meta", {}), **getattr(ref, "meta", {}), **kwargs}
    shape = getattr(mask, "shape", False) or getattr(ref, "shape", True)
    ndim = len(shape)
    cls = Volume if ndim == 3 else Stack
    vol = cls(np.zeros(shape), dtype=values.dtype, **meta)
    vol[mask] = values
    return vol


def default_tags(ndim):
    return {
        "origin": (0,) * ndim,
        "spacing": (1,) * ndim,
        "transform": tuple(
            tuple(1.0 * (i == j) for i in range(ndim)) for j in range(ndim)
        ),
    }


def resize_tags(ndim, tags={}, **kwargs):
    """resize tags to new ndim (crop or pad)"""
    tags = {**tags, **kwargs}
    default = default_tags(ndim)
    if "origin" in tags:
        origin = tags["origin"]
        diff = ndim - len(origin)
        if diff < 0:
            origin = origin[:ndim]
        elif diff > 0:
            origin = list(origin) + list(default["origin"][ndim - diff :])
        tags["origin"] = tuple(origin)

    if "spacing" in tags:
        spacing = tags["spacing"]
        diff = ndim - len(spacing)
        if diff < 0:
            spacing = spacing[:ndim]
        elif diff > 0:
            spacing = list(spacing) + list(default["spacing"][ndim - diff :])
        tags["spacing"] = tuple(spacing)

    if "transform" in tags:
        transform = tags["transform"]
        diff = ndim - len(transform)
        if diff < 0:
            transform = [[value for value in row[:ndim]] for row in transform[:ndim]]
        elif diff > 0:
            deftrf = [list(row) for row in default["transform"]]
            transform = [
                [value for value in list(row[: ndim - diff]) + deftrf[i][ndim - diff :]]
                for i, row in enumerate(list(transform) + deftrf[ndim - diff :])
            ]
        tags["transform"] = tuple(tuple(row) for row in transform)

    return tags


class Stack(np.ndarray):
    """Wrapper over numpy ndarray to add metadata"""

    @property
    def tags(self):
        """Geometrical tags"""
        tags = dict(self._meta, shape=self.shape)
        tags.pop("info", None)
        return tags

    @property
    def meta(self):
        """dict of meta data"""
        return dict(self._meta)

    @property
    def info(self):
        return self._meta["info"]

    @property
    def spacing(self):
        return self._meta["spacing"]

    @property
    def origin(self):
        return self._meta["origin"]

    @property
    def transform(self):
        """return transform matrix in column major order"""
        return self._meta["transform"]

    @property
    def transform_transposed(self):
        """return transform matrix in row major order"""
        transform = self._meta["transform"]
        return tuple(tuple(vec[i] for vec in transform) for i in range(self.ndim))

    @property
    def affine(self):
        """return affine transform matrix"""
        origin = np.asarray(self.origin)
        spacing = np.asarray(self.spacing)
        transform = np.asarray(self.transform)
        return np.r_[np.c_[transform.T * spacing, origin], [[0] * self.ndim + [1]]]

    def get_coords(self, indices):
        """compute real-world coordinates of a sequence of pixels"""
        is_flat = np.isscalar(indices[0])
        indices = np.atleast_2d(indices)
        coords = self.affine @ np.c_[indices, np.ones(len(indices))].T
        return coords[:3, 0] if is_flat else coords[:3].T

    def get_indices(self, coords):
        """compute real-world coordinates of a sequence of pixels"""
        is_flat = np.isscalar(coords[0])
        coords = np.atleast_2d(coords)
        indices = np.linalg.solve(self.affine, np.c_[coords, np.ones(len(coords))].T)
        return indices[:3, 0] if is_flat else indices[:3].T

    #
    # setters

    @origin.setter
    def origin(self, value):
        if len(value) != self.ndim:
            raise ValueError(
                f"'origin' tag must have size {self.ndim}, not {len(value)}"
            )
        self._meta["origin"] = tuple(float(v) for v in value)

    @spacing.setter
    def spacing(self, value):
        if len(value) != self.ndim:
            raise ValueError(f"'spacing' must have size {self.ndim}, not {len(value)}")
        self._meta["spacing"] = tuple(float(v) for v in value)

    @transform.setter
    def transform(self, value):
        """store transform matrix (sequence of orientation vectors)"""
        mat = np.asarray(value)
        try:
            mat = mat.reshape(self.ndim, self.ndim)
        except ValueError:
            raise ValueError(
                f"'transform' must have size: {self.ndim**2}, not {mat.size}"
            )
        # normalize transform
        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        # ignore axes that are 0
        norm[np.isclose(norm, 0)] = 1
        # storing in column-major order
        self._meta["transform"] = tuple(tuple(vec) for vec in (mat / norm))

    def __new__(
        cls,
        obj,
        *,
        spacing=None,
        origin=None,
        transform=None,
        info=None,
        **kwargs,
    ):
        """create Stack object"""
        # remove shape from kwargs
        kwargs.pop("shape", None)

        # get metadata if any
        meta = cls._get_meta(obj, attrs=["meta", "tags"])

        # Init stack
        stack = np.array(obj, **kwargs).view(cls)

        if stack.dtype is np.dtype("O"):
            raise ValueError(f"Invalid array type: {stack.dtype}")

        stack._meta = {"info": {}}
        origin = meta.get("origin") if origin is None else origin
        stack.origin = np.zeros(stack.ndim) if origin is None else origin

        spacing = meta.get("spacing") if spacing is None else spacing
        stack.spacing = np.ones(stack.ndim) if spacing is None else spacing

        transform = meta.get("transform") if transform is None else transform
        stack.transform = np.eye(stack.ndim) if transform is None else transform

        stack.info.update(meta.get("info", {}))
        if info:
            stack.info.update(info)
        return stack

    def __array_finalize__(self, obj):
        """make sure tags are correcty set"""
        if obj is None:
            return
        # copy tags
        self._meta = self._get_meta(obj)

    def __array_wrap__(self, out_arr, context=None):
        """wrap array after function"""
        if out_arr.ndim == 0:
            # return scalar (numpy dtype)
            return out_arr[()]  # out_arr.item()
        elif self.shape != out_arr.shape:
            # if not same shape: drop metadata
            return out_arr
        # else wrap out_array
        return np.ndarray.__array_wrap__(self, out_arr, context)

    @classmethod
    def _get_meta(cls, obj, attrs=["meta"]):
        """copy tags and info from another obj"""

        # try to get attributes directly
        spacing = getattr(obj, "spacing", None)
        origin = getattr(obj, "origin", None)
        transform = getattr(obj, "transform", None)
        info = getattr(obj, "info", {})

        for attr in attrs:
            # try other attributes
            _meta = getattr(obj, attr, None)
            if not _meta:
                continue
            if spacing is None and "spacing" in _meta:
                spacing = _meta["spacing"]
            if origin is None and "origin" in _meta:
                origin = _meta["origin"]
            if transform is None and "spacing" in _meta:
                transform = _meta["transform"]
            if "info" in _meta:
                info.update(_meta.get("info", {}))

        if origin is not None:
            origin = tuple(origin)
        if spacing is not None:
            spacing = tuple(spacing)
        if transform is not None:
            transform = tuple(tuple(vec) for vec in transform)
        if info:
            info = info.copy()
        return {
            "spacing": spacing,
            "origin": origin,
            "transform": transform,
            "info": info,
        }

    # for pickle
    def __reduce__(self):
        """store info"""
        reduced = super().__reduce__()
        attrs = {"_meta": self._meta}
        reduced = reduced[:2] + (reduced[2] + (attrs,),)
        return reduced

    def __setstate__(self, state):
        """retrieve info"""
        attrs = state[-1]
        super().__setstate__(state[:-1])

        self._meta = attrs["_meta"]

    @property
    def A(self):
        """return numpy.ndarray"""
        # return np.asarray(self)
        return self.view(np.ndarray)


class Volume(Stack):
    """3D-restricted Stack"""

    def __new__(cls, obj, **kwargs):
        """create Volume object"""
        array = np.asarray(obj)

        # check dims
        if array.ndim > 3:
            raise ValueError(
                "Volume class can only represent at most 3-dimensional arrays"
            )
        elif array.ndim < 3:
            diff = 3 - array.ndim
            obj = np.reshape(obj, array.shape + (1,) * diff)

        return super().__new__(cls, obj, **kwargs)
