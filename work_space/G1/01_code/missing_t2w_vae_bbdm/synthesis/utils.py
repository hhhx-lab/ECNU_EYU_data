
import os
import re
import json
import numpy as np
import argparse
import nibabel as nib
import torch

import configs
from synthesis.spatial import (
    SpatialTransform,
    assert_support_contained,
    build_foreground_mask,
    build_spatial_transform,
    resample_labels_to_model,
    resample_to_model,
)
## ------------------ REPRODUCIBILITY

def set_seed(seed: int):
    # random.seed(seed)  # Semilla para Python
    np.random.seed(seed)  # Semilla para NumPy
    torch.manual_seed(seed)  # Semilla para PyTorch en CPU
    torch.cuda.manual_seed(seed)  # Semilla para PyTorch en GPU
    torch.cuda.manual_seed_all(seed)  # Semilla para todas las GPUs
    torch.backends.cudnn.deterministic = True  # Garantizar reproducibilidad en CNNs
    torch.backends.cudnn.benchmark = False  # Desactivar optimización no determinista



## ------------------ UTIL FUNCTIONS


def dict_to_args(dict_to_convert, deep_conversion=False):
    """converts a dictionary to an argparse.Namespace object to acces the values as attributes with dot notation"""
    if not isinstance(dict_to_convert, dict):
        raise ValueError("El argumento debe ser un diccionario.")

    args = argparse.Namespace()
    if not deep_conversion:
        for k, v in dict_to_convert.items():
            setattr(args, k, v)
    else:
        for k, v in dict_to_convert.items():
            if isinstance(v, dict):
                setattr(args, k, dict_to_args(v, deep_conversion=True))
            else:
                setattr(args, k, v)
    return args



def args_to_dict(args, deep_conversion=False):
    """Convierte un objeto argparse.Namespace a un diccionario, recursivamente si deep_conversion es True"""
    if not isinstance(args, argparse.Namespace):
        raise ValueError("El argumento debe ser un objeto argparse.Namespace.")

    result = {}

    for k, v in vars(args).items():
        if isinstance(v, argparse.Namespace) and deep_conversion:
            result[k] = args_to_dict(v, deep_conversion=True)  # Recursión para convertir sub-Namespace
        elif isinstance(v, list):  # Si es una lista, convertimos los Namespace dentro de ella (si los hay)
            result[k] = [args_to_dict(i, deep_conversion=True) if isinstance(i, argparse.Namespace) else i for i in v]
        else:
            result[k] = v  # Para valores simples (int, float, string, etc.)

    return result



## ------------------ PATHS

def get_chkpoint_path(path_modal_chk, modality=None):
    if not os.path.isdir(path_modal_chk):
        raise FileNotFoundError(f"Checkpoint directory not found: {path_modal_chk}")
    chk_files = [f for f in os.listdir(path_modal_chk) if f.endswith('.pt')]
    if len(chk_files) == 0:
        raise FileNotFoundError(f"No checkpoint file found in {path_modal_chk}")

    def checkpoint_key(filename):
        match = re.fullmatch(r"model_(\d+)\.pt", filename)
        step = int(match.group(1)) if match else -1
        path = os.path.join(path_modal_chk, filename)
        return step, os.stat(path).st_mtime_ns, filename

    selected = max(chk_files, key=checkpoint_key)
    selected_path = os.path.join(path_modal_chk, selected)
    if len(chk_files) > 1:
        print(
            f"Found {len(chk_files)} checkpoints in {path_modal_chk}; "
            f"selected latest training step: {selected}"
        )
    else:
        print(f"Selected checkpoint: {selected_path}")
    return selected_path



## ------------------ NIFTI FUNCTIONS

def load_nifti(path_name, transpose=False):
    imag_nifti = nib.load(path_name)
    img_data = imag_nifti.get_fdata(dtype=np.float32)
    if transpose:
        img_data = np.transpose(img_data, (1, 0, 2))
    return img_data, (imag_nifti.affine, imag_nifti.header)


def save_nifti(image_np, affine, img_path_name, transpose=False):
    if transpose:
        image_np = np.transpose(image_np, (1, 0, 2))

    img_nifti = nib.Nifti1Image(image_np, affine=affine[0], header=affine[1])
    nib.save(img_nifti, img_path_name)


def prepare_subject_space(
    image_paths,
    seg_path=None,
    target_shape=None,
    base_spacing_mm=1.0,
    margin_mm=5.0,
    foreground_indices=None,
):
    """Load aligned modalities and map them through one reversible transform."""
    if not image_paths:
        raise ValueError("at least one modality path is required")
    target_shape = tuple(target_shape or configs.SHAPE_PREPROCESS_IMG)

    loaded = [nib.load(str(path)) for path in image_paths]
    reference = loaded[0]
    native_shape = tuple(int(item) for item in reference.shape)
    native_affine = np.asarray(reference.affine, dtype=np.float64)
    raw_images = []
    for path, image in zip(image_paths, loaded):
        if tuple(image.shape) != native_shape:
            raise ValueError(
                f"shape mismatch for {path}: {tuple(image.shape)} != {native_shape}"
            )
        if not np.allclose(image.affine, native_affine, atol=1e-4, rtol=0.0):
            raise ValueError(f"affine mismatch for {path}")
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError(f"non-finite values in {path}")
        raw_images.append(data)

    segmentation = None
    if seg_path is not None:
        seg_image = nib.load(str(seg_path))
        if tuple(seg_image.shape) != native_shape:
            raise ValueError(
                f"shape mismatch for {seg_path}: {tuple(seg_image.shape)} != {native_shape}"
            )
        if not np.allclose(seg_image.affine, native_affine, atol=1e-4, rtol=0.0):
            raise ValueError(f"affine mismatch for {seg_path}")
        segmentation = np.asanyarray(seg_image.dataobj)
        if not np.isfinite(segmentation).all():
            raise ValueError(f"non-finite values in {seg_path}")
        rounded = np.rint(segmentation)
        if not np.allclose(segmentation, rounded, atol=1e-6):
            raise ValueError(f"non-integer segmentation labels in {seg_path}")
        segmentation = rounded.astype(np.int16)

    normalized_images = [robust_normalize(image) for image in raw_images]
    if foreground_indices is None:
        foreground_images = normalized_images
    else:
        foreground_indices = tuple(int(index) for index in foreground_indices)
        if not foreground_indices or any(
            index < 0 or index >= len(normalized_images) for index in foreground_indices
        ):
            raise ValueError(f"invalid foreground indices: {foreground_indices}")
        foreground_images = [normalized_images[index] for index in foreground_indices]

    transform = build_spatial_transform(
        foreground_images,
        native_affine,
        segmentation=segmentation,
        target_shape=target_shape,
        base_spacing_mm=base_spacing_mm,
        margin_mm=margin_mm,
    )
    foreground_support = build_foreground_mask(
        foreground_images,
        segmentation=segmentation,
    )
    foreground_audit = assert_support_contained(
        foreground_support, transform, "foreground"
    )
    lesion_audit = None
    if segmentation is not None and np.any(segmentation > 0):
        lesion_audit = assert_support_contained(segmentation > 0, transform, "lesion")

    model_images = [
        resample_to_model(image, transform, order=1) for image in normalized_images
    ]
    model_segmentation = (
        resample_labels_to_model(segmentation, transform)
        if segmentation is not None
        else None
    )
    return {
        "images": model_images,
        "segmentation": model_segmentation,
        "transform": transform,
        "native_affine": (native_affine, reference.header.copy()),
        "native_shape": native_shape,
        "foreground_support_audit": foreground_audit,
        "lesion_support_audit": lesion_audit,
    }


def _load_spatial_transform(transform_path):
    with open(transform_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    return SpatialTransform.from_dict(metadata.get("transform", metadata))


def load_image_in_model_space(image_path, transform_path):
    transform = _load_spatial_transform(transform_path)
    image = nib.load(str(image_path))
    if tuple(image.shape) != transform.native_shape:
        raise ValueError(
            f"image shape {tuple(image.shape)} != transform native shape {transform.native_shape}"
        )
    if not np.allclose(image.affine, transform.native_affine, atol=1e-4, rtol=0.0):
        raise ValueError("image affine does not match saved spatial transform")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite values in {image_path}")
    return resample_to_model(robust_normalize(data), transform, order=1)


def load_segmentation_in_model_space(seg_path, transform_path):
    transform = _load_spatial_transform(transform_path)
    seg_image = nib.load(str(seg_path))
    if tuple(seg_image.shape) != transform.native_shape:
        raise ValueError(
            f"segmentation shape {tuple(seg_image.shape)} != transform native shape "
            f"{transform.native_shape}"
        )
    if not np.allclose(seg_image.affine, transform.native_affine, atol=1e-4, rtol=0.0):
        raise ValueError("segmentation affine does not match saved spatial transform")
    segmentation = np.asanyarray(seg_image.dataobj)
    if not np.isfinite(segmentation).all():
        raise ValueError(f"non-finite values in {seg_path}")
    rounded = np.rint(segmentation)
    if not np.allclose(segmentation, rounded, atol=1e-6):
        raise ValueError(f"non-integer segmentation labels in {seg_path}")
    return resample_labels_to_model(rounded.astype(np.int16), transform)


## ------------------ PREPROCESSING/POSTPROCESSING FUNCTIONS


def robust_normalize(
    img,
    percentile=(0, 100),
    mask=None,
    reference_tensor=None,
    strictly_positive=True,
    clip_values = True
):
    """
    Normaliza una imagen al rango [0, 1] usando percentiles robustos.
    Soporta máscaras, tensor de referencia y opción de valores estrictamente positivos.

    Parámetros:
        img (np.ndarray): imagen o tensor a normalizar.
        percentile (tuple): percentiles para la normalización (p_min, p_max). Por defecto (0.5, 99.5).
        mask (np.ndarray, opcional): máscara booleana o binaria para calcular percentiles solo en regiones válidas.
        reference_tensor (np.ndarray, opcional): tensor del cual se obtendrán los percentiles. Si es None, se usa `img`.
        strictly_positive (bool): si es True, se fuerza que el valor mínimo no sea menor a 0.

    Retorna:
        np.ndarray: imagen normalizada en el rango [0, 1].
    """

    # Determinar tensor de referencia
    ref = reference_tensor if reference_tensor is not None else img

    # Aplicar máscara si se proporciona
    if mask is not None:
        ref = ref[mask > 0]

    # Calcular percentiles
    p_min, p_max = np.percentile(ref, percentile)

    # Ajuste para valores estrictamente positivos
    if strictly_positive and p_min < 0:
        p_min = 0

    # Clip antes de normalizar
    if clip_values:
        img_clipped = np.clip(img, p_min, p_max)
    else:
        img_clipped = img

    # Normalización al rango [0, 1]
    if p_max > p_min:
        img_normalized = (img_clipped - p_min) / (p_max - p_min)
    else:
        img_normalized = np.zeros_like(img)

    return img_normalized



def update_affine(original_affine: np.ndarray, offset: tuple) -> np.ndarray:
    """
    Modifica la matriz affine de acuerdo al desplazamiento en voxels.

    :param original_affine: Matriz affine original (4x4)
    :param offset: Desplazamiento en voxels (dx, dy, dz)
    :return: Matriz affine ajustada
    """
    dx, dy, dz = offset
    translation = original_affine[:3, :3] @ np.array([dx, dy, dz])
    new_affine = original_affine.copy()
    new_affine[:3, 3] -= translation
    return new_affine



def resize_center_crop_pad(image: np.ndarray, new_shape: tuple, affine=None) -> np.ndarray:
    """
    Ajusta el tamaño de una imagen 3D (x, y, z) recortando o rellenando con ceros.

    :param image: np.ndarray de tamaño (x, y, z)
    :param new_shape: Tuple (nx, ny, nz) con las nuevas dimensiones
    :return: np.ndarray de tamaño (nx, ny, nz)
    """
    x, y, z = image.shape
    nx, ny, nz = new_shape

    # Inicializar imagen de salida con ceros
    new_image = np.zeros((nx, ny, nz), dtype=image.dtype)

    # Calcular los índices de recorte o padding para cada dimensión
    def get_slices(old, new):
        if old > new:
            start = (old - new) // 2
            return slice(start, start + new), slice(0, new)
        else:
            start = (new - old) // 2
            return slice(0, old), slice(start, start + old)

    x_slice_old, x_slice_new = get_slices(x, nx)
    y_slice_old, y_slice_new = get_slices(y, ny)
    z_slice_old, z_slice_new = get_slices(z, nz)

    # Copiar los datos ajustados
    new_image[x_slice_new, y_slice_new, z_slice_new] = image[x_slice_old, y_slice_old, z_slice_old]
    offset = (x_slice_new.start-x_slice_old.start, y_slice_new.start-y_slice_old.start, z_slice_new.start-z_slice_old.start)

    if affine is not None:
        # Actualizar la matriz affine si se proporciona
        new_affine_matrix = update_affine(affine[0], offset)
        return new_image, offset, (new_affine_matrix, affine[1])
    else:
        return new_image, offset



def combine_masks(mask_list, combination="or"):
    """
    Combine multiple masks into a single mask.

    Parameters:
    - mask_list: List of numpy arrays representing the masks.
    - combination: Method to combine masks, either "or" or "and".

    Returns:
    - combined_mask: Numpy array representing the combined mask.
    """
    if not mask_list:
        return None

    combined_mask = mask_list[0].copy()

    for mask in mask_list[1:]:
        if combination == "or":
            combined_mask = np.logical_or(combined_mask, mask)
        elif combination == "and":
            combined_mask = np.logical_and(combined_mask, mask)
        elif combination == "min2" and len(mask_list) > 1:
            combined_mask = np.sum(mask_list, axis=0) >= 2
        else:
            raise ValueError("Combination method must be 'or' or 'and'.")

    return combined_mask.astype(np.uint8)



def preprocessing(img, affine=None):
    img = robust_normalize(img)
    img, _, affine = resize_center_crop_pad(img, configs.SHAPE_PREPROCESS_IMG, affine=affine)
    return img, affine


def postprocess_intensity(img, modality, bmask=None):
    img = np.asarray(img, dtype=np.float32).copy()
    if bmask is not None:
        img[bmask == 0] = 0
    else:
        img[img < 0.01] = 0
    img = np.clip(img, 0, 1)
    img = robust_normalize(img, strictly_positive=True)
    # img[img < 0.015] = 0

    if modality in ("t1n", "t2f"):
        img -= 0.015
        img = robust_normalize(img, strictly_positive=True)
    elif modality in ("t2w"):
        img[img < 0.015] = .0
        img[img > 0.98] = .98
        img = robust_normalize(img, strictly_positive=True)
    else:
        img[img < 0.015] = 0

    return img


def postprocessing(img, modality, org_shape, bmask=None):
    """Legacy fixed-voxel restoration retained for old artifacts only."""
    img = resize_center_crop_pad(img, org_shape)[0]
    return postprocess_intensity(img, modality, bmask=bmask)


def postprocessing_raw(img,org_shape):
    img = resize_center_crop_pad(img, org_shape)[0]
    return img




## ------------------ ENSAMBLE FUNCTIONS

def combine_images(img_mod_list, combination_type='mean', weights=None):

    imgs = np.array(img_mod_list)
    if combination_type == 'mean':
        return np.mean(imgs, axis=0)
    elif combination_type == "weighted_mean" and weights is not None:
        if len(weights) != imgs.shape[0]:
            raise ValueError("Weights length must match the number of images.")
        weighted_imgs = imgs * np.array(weights).T[:, np.newaxis, np.newaxis, np.newaxis]
        return np.sum(weighted_imgs, axis=0) / np.sum(weights)





## ------------------ GENARATIVE MODELS FUNCTIONS

def prepare_image(image, autoencoder):
    device = next(autoencoder.parameters()).device  # Obtiene el dispositivo del modelo
    image = torch.tensor(image).to(device)
    image = image.unsqueeze(0).unsqueeze(0)

    dtype = next(autoencoder.parameters()).dtype  # Obtiene el tipo de los pesos del modelo
    image = image.to(dtype)  # Convierte la entrada a ese tipo

    return image


def create_modality_one_hot(modality):
    """
    Crea un tensor one-hot para la modalidad especificada.
    """
    if modality not in configs.MODALITY_LIST:
        raise ValueError(f"Modality {modality} not recognized. Available modalities: {configs.MODALITY_LIST}")

    to_modality_index = configs.MODALITY_LIST.index(modality)
    to_modality_one_hot = np.zeros((1, len(configs.MODALITY_LIST)))
    to_modality_one_hot[0, to_modality_index] = 1.0
    return to_modality_one_hot


def preprare_bbdm_latens(latens_list, to_modality_index):
    n_modalities = len(configs.MODALITY_LIST)
    new_latens_list = []
    cur_latent_index = 0
    for i in range(n_modalities):
        if i == to_modality_index:
            new_latens_list.append(np.zeros_like(latens_list[0]))
        else:
            new_latens_list.append(latens_list[cur_latent_index])
            cur_latent_index += 1
    return new_latens_list

def gray_to_rgb(img, to_uint8=True, normalize=True):
    if normalize:
        img_min, img_max = np.min(img), np.max(img)

        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img)
    else:
        img_norm = img.copy()

    if to_uint8:
        img_norm = (255 * img_norm).astype(np.uint8)

    img_rgb = np.stack([img_norm] * 3, axis=-1)
    return img_rgb

### ------------------ TRAINING

class EMA:
    def __init__(self, model, decay, warm_up_steps=0, warm_up_decay=0.1):
        """
        Inicializa la clase EMA para gestionar la media móvil exponencial de los parámetros del modelo.
        Args:
            model (torch.nn.Module): El modelo cuyos parámetros se van a promediar.
            decay (float): Tasa de decaimiento para la EMA.
        """
        self.model = model
        self.decay = decay
        self.warm_up_steps = warm_up_steps
        self.warm_up_decay = warm_up_decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, step=None):
        """
        Actualiza los parámetros sombra utilizando la EMA.
        """
        decay = self.decay
        if step is not None and self.warm_up_steps > 0 and step < self.warm_up_steps:
            decay = self.warm_up_decay

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - decay) * param.data + decay * self.shadow[name]

    def apply_shadow(self):
        """
        Aplica los parámetros promediados (EMA) al modelo, guardando los originales.
        """
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """
        Restaura los parámetros originales del modelo.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
