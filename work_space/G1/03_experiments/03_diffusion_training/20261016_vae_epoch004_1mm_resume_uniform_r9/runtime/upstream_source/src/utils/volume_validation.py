"""Validate complete co-registered NIfTI case inputs before consuming them."""
import hashlib
import nibabel as nib
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_case(paths, allowed_labels=(0, 1, 2, 3, 4), affine_tol=0.1):
    images = [nib.load(str(path)) for path in paths]
    ref = images[-1]
    if len(ref.shape) != 3:
        raise ValueError("Segmentation must be 3D")
    corners = np.array([[i, j, k, 1.] for i in (0, ref.shape[0] - 1)
                        for j in (0, ref.shape[1] - 1) for k in (0, ref.shape[2] - 1)]).T
    arrays = []
    for path, img in zip(paths, images):
        if img.shape != ref.shape:
            raise ValueError(f"Shape mismatch: {path}")
        if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-10:
            raise ValueError(f"Invalid affine: {path}")
        spacing = np.linalg.norm(img.affine[:3, :3], axis=0)
        if not np.allclose(spacing, img.header.get_zooms()[:3], rtol=1e-5, atol=1e-5):
            raise ValueError(f"Header spacing differs from affine: {path}")
        shift = np.linalg.norm(((img.affine - ref.affine) @ corners)[:3], axis=0).max()
        if shift > affine_tol:
            raise ValueError(f"Modality grid mismatch ({shift:.4f} mm): {path}")
        array = img.get_fdata(dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"NaN/Inf: {path}")
        arrays.append(array)
    labels = arrays[-1]
    if not np.equal(labels, np.rint(labels)).all() or not np.isin(labels, list(allowed_labels)).all():
        raise ValueError("Illegal or non-integer segmentation labels")
    return images, arrays
