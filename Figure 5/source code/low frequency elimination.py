from PIL import Image
import numpy as np
import math

pi = math.pi

def Low_pass_filter(single_channel_matrix, ratio):
    single_channel_matrix_copy = single_channel_matrix.copy()
    f_transform = np.fft.fft2(single_channel_matrix_copy)
    f_shift = np.fft.fftshift(f_transform)
    rows, cols = single_channel_matrix_copy.shape
    crow, ccol = rows // 2, cols // 2
    radius = int(np.sqrt(224 * 224 * 0.5 / pi))
    mask = np.zeros((rows, cols), dtype=np.uint8)
    y, x = np.ogrid[:rows, :cols]
    center_mask = (x - ccol)**2 + (y - crow)**2 <= radius**2
    mask[center_mask] = 1
    filtered_f_shift = f_shift * mask
    filtered_f = np.fft.ifftshift(filtered_f_shift)
    filtered_image = np.fft.ifft2(filtered_f)
    filtered_image = np.abs(filtered_image)
    indices = np.unravel_index(np.argsort(filtered_image.ravel())[int(-224 * 224 * ratio):], filtered_image.shape)
    single_channel_matrix_copy[indices] = 255
    return single_channel_matrix_copy

actual_image_absolute_path = "..."
image = Image.open(actual_image_absolute_path)
image = image.resize((224, 224))
actual_image = np.array(image)

# The R, G, B three channel matrix of the actual image
actual_image_channel_R = actual_image[:, :, 0]
actual_image_channel_R = actual_image_channel_R.astype(np.float64)
actual_image_channel_G = actual_image[:, :, 1]
actual_image_channel_G = actual_image_channel_G.astype(np.float64)
actual_image_channel_B = actual_image[:, :, 2]
actual_image_channel_B = actual_image_channel_B.astype(np.float64)

"""

"""
ratio = 0.15

image_lOW_frequency_information_R = Low_pass_filter(actual_image_channel_R, ratio)
image_lOW_frequency_information_G = Low_pass_filter(actual_image_channel_G, ratio)
image_lOW_frequency_information_B = Low_pass_filter(actual_image_channel_B, ratio)

# Combine three channels into an RGB image
image_rgb = np.stack([image_lOW_frequency_information_R, image_lOW_frequency_information_G, image_lOW_frequency_information_B], axis=-1)
# Convert data type to 8-bit unsigned integer
image_rgb = image_rgb.astype(np.uint8)
# Create PIL image object
image_pil = Image.fromarray(image_rgb)
image_pil.save("low frequency elimination.png")
