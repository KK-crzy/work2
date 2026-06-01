import cv2 
import numpy as np


def linear_normalization(image):
    image = image.astype(np.float32)
    min_value = np.min(image)
    max_value = np.max(image)

    if max_value == min_value:
        return np.zeros_like(image)

    normalized_image = (image - min_value) / (max_value - min_value)
    return normalized_image


def resize_image(image, scale, interpol='linear'):
    # scale: 0-1
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)

    if interpol == 'linear':
        interpol_method = cv2.INTER_LINEAR
    elif interpol == 'cubic':
        interpol_method = cv2.INTER_CUBIC
    elif interpol == 'area':
        interpol_method = cv2.INTER_AREA
    elif interpol == 'nearest':
        interpol_method = cv2.INTER_NEAREST

    resized = cv2.resize(image, dim, interpolation=interpol_method)  # default interpolation is cv2.INTER_LINEAR
    return cv2.resize(resized, (image.shape[1], image.shape[0]), interpolation=interpol_method)