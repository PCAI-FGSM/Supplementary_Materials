import os
import numpy as np
from PIL import Image

def image_to_matrix(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    actual_image = np.array(image)

    matrix = np.zeros((3, 224, 224), dtype=np.float64)
    matrix[0] = actual_image[:, :, 0].astype(np.float64)
    matrix[1] = actual_image[:, :, 1].astype(np.float64)
    matrix[2] = actual_image[:, :, 2].astype(np.float64)

    return matrix

# Import the path of the original image folder
folder1 = r"..."

# Import adversarial example folder path
folder2 = r"..."

image_names = [f for f in os.listdir(folder1) if f.endswith('.png')]
image_names.sort(key=lambda x: int(x.split('.')[0]))

max_diffs = []

for name in image_names:
    path1 = os.path.join(folder1, name)
    path2 = os.path.join(folder2, name)

    matrix1 = image_to_matrix(path1)
    matrix2 = image_to_matrix(path2)

    diff = np.abs(matrix1 - matrix2)
    max_diff = np.max(diff)
    max_diffs.append(max_diff)

print(max_diffs)
average_max_diff = np.mean(max_diffs)
print("The average maximum perturbation intensity is：", average_max_diff)
