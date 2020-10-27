"""Functionality for processing raw image data"""

import numpy as np

from scipy.ndimage import filters

try:
    import cv2

    CV2_BORDER_DEFAULT = cv2.BORDER_DEFAULT
except:
    cv2 = None
    CV2_BORDER_DEFAULT = 4


def hot_pixel_filter(img: np.ndarray, hot_pixel_thres: float, inplace: bool = False) -> np.ndarray:
    """Hot pixel filter as implemented in https://github.com/BodenmillerGroup/ImcPluginsCP

    Sets all hot pixels to the maximum of their 8-neighborhood. Hot pixels are defined as pixels, whose values are
    larger by ``hot_pixel_thres`` than the maximum of their 8-neighborhood.

    :param img: raw image data, shape: ``(c, y, x)``
    :param hot_pixel_thres: hot pixel threshold
    :param inplace: if ``True``, the image is modified in-place
    :return: The hot pixel-filtered image
    """
    if img.ndim != 3:
        raise ValueError(f'Invalid number of image dimensions: expected 3, got {img.ndim}')
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    if cv2 is not None:
        max_neighbor_img = cv2.dilate(np.moveaxis(img, 0, 2), kernel, iterations=1, borderType=cv2.BORDER_REFLECT101)
        if np.ndim(max_neighbor_img) == 2:
            max_neighbor_img = np.expand_dims(max_neighbor_img, 2)
        max_neighbor_img = np.moveaxis(max_neighbor_img, 2, 0)
    else:
        max_neighbor_img = filters.maximum_filter(kernel[None, :, :], footprint=kernel, mode='mirror')
    filter_mask = (img - max_neighbor_img) > hot_pixel_thres
    if not inplace:
        img = img.copy()
    img[filter_mask] = max_neighbor_img[filter_mask]
    return img


def median_filter_cv2(img: np.ndarray, size: int) -> np.ndarray:
    """Fast median blur using OpenCV

    :param img: raw image data, shape: ``(c, y, x)``
    :param size: size of the median filter, must be odd
    :return: the median-filtered image
    """
    if cv2 is None:
        raise RuntimeError('python-opencv is not installed')
    if img.ndim != 3:
        raise ValueError(f'Invalid number of image dimensions: expected 3, got {img.ndim}')
    if size <= 0:
        raise ValueError(f'Filter size {size} is not positive')
    if size % 2 == 0:
        raise ValueError(f'Filter size {size} is an even number')
    img = np.moveaxis(img, 0, 2)
    img = cv2.medianBlur(img, size)
    if np.ndim(img) == 2:
        img = np.expand_dims(img, 2)
    return np.moveaxis(img, 2, 0)


def gaussian_filter_cv2(img: np.ndarray, size: int = 0, sigma: float = 0) -> np.ndarray:
    """Fast Gaussian blur using OpenCV

    :param img: raw image data, shape: ``(c, y, x)``
    :param size: size of the median filter, must be odd
    :param sigma: Gaussian kernel standard deviation
    :return: the Gaussian-filtered image
    """
    if cv2 is None:
        raise RuntimeError('python-opencv is not installed')
    if img.ndim != 3:
        raise ValueError(f'Invalid number of image dimensions: expected 3, got {img.ndim}')
    if size <= 0:
        raise ValueError(f'Filter size {size} is not positive')
    if size % 2 == 0:
        raise ValueError(f'Filter size {size} is an even number')
    img = np.moveaxis(img, 0, 2)
    img = cv2.GaussianBlur(img, (size, size), sigma, borderType=cv2.BORDER_REFLECT101)
    if np.ndim(img) == 2:
        img = np.expand_dims(img, 2)
    return np.moveaxis(img, 2, 0)


def rotate_centered_cv2(img: np.ndarray, angle: float, border_mode: int = CV2_BORDER_DEFAULT,
                        expand_bbox: bool = False) -> np.ndarray:
    """Fast centered image rotation using OpenCV

    :param img: raw image data, shape: ``(c, y, x)``
    :param angle: rotation angle, in degrees
    :param border_mode: pixel extrapolation method, see :func:`cv2.warpAffine`
    :param expand_bbox: if set to ``True``, the bounding box of the resulting image is expanded to contain the full
        rotated image
    :return: the image, rotated around its center
    """
    if cv2 is None:
        raise RuntimeError('python-opencv is not installed')
    if img.ndim != 3:
        raise ValueError(f'Invalid number of image dimensions: expected 3, got {img.ndim}')
    height, width = img.shape[1], img.shape[2]
    center_x, center_y = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    if expand_bbox:
        width = int(height * np.abs(rotation_matrix[0, 1]) + width * np.abs(rotation_matrix[0, 0]))
        height = int(height * np.abs(rotation_matrix[0, 0]) + width * np.abs(rotation_matrix[0, 1]))
        rotation_matrix[0, 2] += (width / 2) - center_x
        rotation_matrix[1, 2] += (height / 2) - center_y
    img = np.moveaxis(img, 0, 2)
    img = cv2.warpAffine(img, rotation_matrix, (width, height), borderMode=border_mode, borderValue=0)
    if np.ndim(img) == 2:
        img = np.expand_dims(img, 2)
    return np.moveaxis(img, 2, 0)
