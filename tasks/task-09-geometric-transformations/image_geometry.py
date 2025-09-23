# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    height, width = img.shape

    dx, dy = width // 4, height // 4
    translated_img = np.zeros_like(img)
    translated_img[dy:, dx:] = img[:-dy, :-dx]

    rotated_img = np.rot90(img, k=-1)

    new_width = int(width * 1.5)
    stretched_img = np.zeros((height, new_width), dtype=img.dtype)
    x_new = np.arange(new_width)
    x_original = x_new / 1.5
    x_original_int = np.round(x_original).astype(int)
    x_original_int = np.minimum(x_original_int, width - 1)
    stretched_img = img[:, x_original_int]

    mirrored_img = img[:, ::-1]

    distorted_img = np.zeros_like(img)
    center_x, center_y = width / 2, height / 2
    k = -0.0005

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    x_centered = x - center_x
    y_centered = y - center_y
    
    r = np.sqrt(x_centered**2 + y_centered**2)
    
    r_distorted_factor = 1 + k * r**2
    x_source = (x_centered / r_distorted_factor) + center_x
    y_source = (y_centered / r_distorted_factor) + center_y

    x_source_int = np.round(x_source).astype(int)
    y_source_int = np.round(y_source).astype(int)
    
    x_source_int = np.clip(x_source_int, 0, width - 1)
    y_source_int = np.clip(y_source_int, 0, height - 1)
    
    distorted_img[y, x] = img[y_source_int, x_source_int]

    return {
        "translated": translated_img,
        "rotated": rotated_img,
        "stretched": stretched_img,
        "mirrored": mirrored_img,
        "distorted": distorted_img
    }