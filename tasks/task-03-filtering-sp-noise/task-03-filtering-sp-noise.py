import cv2
import numpy as np

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """
    denoised = cv2.medianBlur(image, 5)
    return denoised

if __name__ == "__main__":
    image_path = "../../img/head.png"
    noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    denoised_image = remove_salt_and_pepper_noise(noisy_image)
    cv2.imwrite("denoised_image.png", denoised_image)
