from pathlib import Path
import concurrent.futures
from typing import Tuple, List

import cv2
import numpy as np
from tqdm import tqdm

def extract_coords(img_name: str) -> Tuple[int, int, int, int]:

    """
    Extracts the coordinates of a given patch from its filename.
    """

    stem = Path(img_name).stem
    y1, y2, x1, x2 = [int(c) for c in stem.split("-")[1:]]
    coords = (y1, y2, x1, x2)

    return coords


def merge_patches(
    patches: List[np.ndarray], 
    coords: List[Tuple[int]],
    target_patch_size: int
    ) -> np.ndarray:

    """
    Merges patches into the whole image given a list of coordinates.

    Parameters
    ----------
    patches: List[np.ndarray]
        A list containing the patches.

    coords: List[Tuple[int]]
        A list of coordinates specifying the location of each patch in the entire image.

    target_patch_size: int
        The resized patch size.

    Returns
    -------
    merged_img: np.ndarray
        The merged image.
    """
    
    orig_height = 384 * target_patch_size
    orig_width = 384 * target_patch_size
    
    num_channels = patches[0].shape[-1]
    merged_img = np.zeros((orig_height, orig_width, num_channels), dtype=np.float32)

    for i, coord in enumerate(coords):
        merged_img[coord[0] : coord[1], coord[2] : coord[3], :] = patches[i]

    return merged_img


def adjust_coords(
    coords: List[int], 
    src_patch_size: int, 
    target_patch_size: int
    ):

    """
    Adjust coordinates according to the change in patch size.

    Parameters
    ----------
    coords: List[int]
        The coordinates in the form [y1, y2, x1, x2]

    src_patch_size: int
        The original patch size.

    target_patch_size: int
        The resized patch size.

    Returns
    -------
    adjusted_coords: List[int]
        The adjusted coordinates per patch given the new patch size.
    """

    scaling_factor = src_patch_size / target_patch_size

    adjusted_coords = []

    for coord in coords:
        adjusted = [int(c / scaling_factor) for c in coord]

        adjusted_coords.append(adjusted)

    return adjusted_coords


def min_max_scale(attention_col):
    return (attention_col - attention_col.min()) / (attention_col.max() - attention_col.min())


def read_image(img_path, size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def multithread_read_img(img_paths, size):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(read_image, img_paths, [size] * len(img_paths)), total=len(img_paths), desc="processing images"))

    return results