# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2
import numpy as np
from skimage.exposure import match_histograms


def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:

    source_lab = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB)
    reference_lab = cv2.cvtColor(reference_img, cv2.COLOR_RGB2LAB)

    matched_lab = match_histograms(
        source_lab,
        reference_lab,
        channel_axis=-1
    )

    matched_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)

    return matched_rgb.astype(np.uint8)
