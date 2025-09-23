"""
task-11-blur-estimation-with-fourier-transform.py

>>> IMPORTANT <<<
Implement the function `frequency_blur_score` below.

Rules:
- Keep the function name and signature EXACTLY the same.
- Do NOT use any external network calls.
- You may ONLY use standard Python, NumPy, and OpenCV (cv2).
- Return a single float (higher = sharper OR lower = blurrier, but be consistent).

Tip (from the FFT blur-detection tutorial):
- Convert to grayscale
- 2D FFT -> shift DC to center
- Zero-out a centered square (low frequencies)
- Magnitude spectrum (e.g., log1p(abs(...)))
- Use the mean magnitude of the remaining spectrum as the score
"""

from typing import Union
import numpy as np
import cv2

def frequency_blur_score(
        image: Union[np.ndarray, "cv2.Mat"],
        center_size: int = 60
) -> float:
    if image.ndim == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    f_image = np.asarray(gray_image, dtype=np.float32)
    dft = cv2.dft(f_image, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2

    half_size = center_size // 2

    dft_shift[crow - half_size: crow + half_size, ccol - half_size: ccol + half_size] = 0

    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    total_pixels = rows * cols
    low_freq_pixels = center_size * center_size

    high_freq_pixels = total_pixels - low_freq_pixels
    if high_freq_pixels <= 0:
        return 0.0

    score = np.sum(magnitude_spectrum) / high_freq_pixels

    return score
