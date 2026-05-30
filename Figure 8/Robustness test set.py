"""
This code is used to generate robustness test sets of different levels
"""


import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import io
from PIL import Image


def compute_ssim(img1_array, img2_array):
    """
    Compute SSIM between two images represented as numpy arrays.

    :param img1_array: First image as a numpy array (3-channel).
    :param img2_array: Second image as a numpy array (3-channel).
    :return: SSIM score between the two images.
    """
    if img1_array.shape != img2_array.shape:
        raise ValueError("Images must have the same dimensions")

    data_range = 255  # Assuming images are in uint8 format
    return ssim(img1_array, img2_array, multichannel=True, win_size=3, data_range=data_range)


# The absolute path of the folder where the original images are stored
actual_image_root = "..."

# The absolute path of the folder storing robust images
robust_root = "..."

# Save the folder path of the generated robustness test set for the specified SSIM
save_root = "..."

# CIFAR-10
ssim_ranges = [
    (1.0, 0.95),
    (0.95, 0.9),
    (0.9, 0.85),
    (0.85, 0.8),
    (0.8, 0.75)
]

# RSCD
ssim_ranges = [
    (1.0, 0.99),
    (0.99, 0.98),
    (0.98, 0.97),
    (0.97, 0.96),
    (0.96, 0.95)
]

noise_types = [d for d in os.listdir(robust_root) if os.path.isdir(os.path.join(robust_root, d))]

for noise_type in noise_types:
    robust_folder = os.path.join(robust_root, noise_type, "5")
    save_folder = os.path.join(save_root, noise_type)

    subfolders = [os.path.join(save_folder, str(i + 1)) for i in range(5)]
    for subfolder in subfolders:
        for class_id in range(10):
            class_subfolder = os.path.join(subfolder, str(class_id))
            os.makedirs(class_subfolder, exist_ok=True)

    for class_id in range(10):
        class_folder = os.path.join(robust_folder, str(class_id))
        actual_image_folder = os.path.join(actual_image_root, str(class_id))
        for filename in os.listdir(class_folder):
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                actual_image_path = os.path.join(actual_image_folder, filename)
                robust_image_path = os.path.join(class_folder, filename)

                actual_image = io.imread(actual_image_path)
                robust_image = io.imread(robust_image_path)

                actual_image = np.array(Image.fromarray(actual_image).resize((32, 32)))
                robust_image = np.array(Image.fromarray(robust_image).resize((32, 32)))

                distance_matrix = actual_image.astype(np.float64) - robust_image.astype(np.float64)

                for i, (ssim_max, ssim_min) in enumerate(ssim_ranges):
                    low, high = 0.0, 1.0
                    scaling_factor = 0.5
                    max_iterations = 50
                    last_modified_image = None

                    for iteration in range(max_iterations):
                        modified_image = actual_image - (distance_matrix * scaling_factor)
                        modified_image = np.clip(modified_image, 0, 255).astype(np.uint8)
                        last_modified_image = modified_image

                        current_ssim = compute_ssim(actual_image, modified_image)

                        if ssim_min <= current_ssim <= ssim_max:
                            save_path = os.path.join(subfolders[i], str(class_id), filename)
                            Image.fromarray(modified_image).save(save_path)
                            break
                        elif current_ssim < ssim_min:
                            high = scaling_factor
                        else:
                            low = scaling_factor
                        scaling_factor = (low + high) / 2

                    else:
                        save_path = os.path.join(subfolders[i], str(class_id), filename)
                        Image.fromarray(last_modified_image).save(save_path)