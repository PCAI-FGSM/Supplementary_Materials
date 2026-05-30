import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
import numpy as np
import math
import shutil
import torch.nn as nn
import pandas as pd
from collections import defaultdict

LOSS_RECORDER = defaultdict(list)


def count_differences(actual_image_matrix, attack_sample_matrix):
    diff = np.sum(actual_image_matrix != attack_sample_matrix)
    return diff


def calculate_gradient(image_path, label, weight_path, model_cnn, iter_idx, img_name):
    # Define transforms
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_tensor = data_transform(image).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    # Load model
    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Define loss
    loss_fn = nn.CrossEntropyLoss()
    label_tensor = torch.tensor([label], dtype=torch.long).to(device)

    # Forward pass
    outputs = model(img_tensor)

    # Calculate loss
    cost = loss_fn(outputs, label_tensor)

    loss_value = cost.item()

    LOSS_RECORDER[img_name].append(loss_value)

    # iter_idx = len(LOSS_RECORDER[img_name])
    print(f"[{img_name}] Iter {iter_idx} | Loss: {loss_value:.6f}")
    # ==================================

    # Calculate gradient
    grad = torch.autograd.grad(
        cost,
        img_tensor,
        retain_graph=False,
        create_graph=False
    )[0]

    return grad.squeeze(0).cpu().numpy()

def save_loss_to_excel(filename="loss_record.xlsx"):
    if len(LOSS_RECORDER) == 0:
        return

    df = pd.DataFrame(LOSS_RECORDER)
    df.index.name = "Iteration"
    df.to_excel(filename)


def divide_matrix(input_matrix, level):
    unique_elements = np.unique(input_matrix)
    num_unique = len(unique_elements)
    threshold = np.percentile(unique_elements, level)
    mark_matrix_actual_index = np.where(input_matrix > threshold, 1, 0)
    return mark_matrix_actual_index


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


def predict_image_path(image_path, index_path, weight_path, index, model_cnn):
    # Load image
    img = Image.open(image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    # Create model
    model = model_cnn(num_classes=1000).to(device)
    # Load model weights
    model.load_state_dict(torch.load(weight_path))
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    # Get the index of the class with the highest probability
    predicted_class_index = torch.argmax(classification_probability).item()
    return(predicted_class_index,output[index])
    # return(predicted_class_index)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


# The absolute path of index file
index_file_absolute_path = r"..."
assert os.path.exists(index_file_absolute_path), "file: '{}' dose not exist.".format(index_file_absolute_path)
with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)

# The absolute path of weight file for convolutional neural network model
weight_file_absolute_path = r"..."

# The maximum degree of tampering of a single pixel
degree = 3

# Iterative step size
iteration_step_size = 0.001

# Calculate the maximum number of iterations
max_num_iterative = 200

model_current = resnet18

ratio = 99

actual_images_folder_absolute_path = "..."
output_directory = "..."

# The total number of images in the folder that require tampering attacks
the_total_number_of_tampered_images = 0

image_files = sorted([f for f in os.listdir(actual_images_folder_absolute_path) if f.endswith('.png')], key=sort_func)

# Record the total number of original images
image_num = 0

# Record the total number of successfully attacked images
success_num = 0

for image_file in image_files:
    print("#############################################################################################")
    print(f"The image number currently being processed is: {image_file}")
    image_num = image_num + 1
    degree = degree + 1

    the_total_number_of_tampered_images = the_total_number_of_tampered_images + 1

    actual_image_absolute_path = os.path.join(actual_images_folder_absolute_path, image_file)

    img_name = os.path.basename(actual_image_absolute_path)

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

    # Standardize the actual image
    actual_image_transform_matrix_R = ((actual_image_channel_R / 255) - 0.485) / 0.229
    actual_image_transform_matrix_G = ((actual_image_channel_G / 255) - 0.456) / 0.224
    actual_image_transform_matrix_B = ((actual_image_channel_B / 255) - 0.406) / 0.225
    ####################################################################################################
    # Perform category prediction on the original image
    actual_image_index, x = predict_image_path(actual_image_absolute_path, index_file_absolute_path,
                                               weight_file_absolute_path, 0, model_current)

    flag = 0
    num_iterative = 0
    iterative_image_path = actual_image_absolute_path
    while num_iterative <= max_num_iterative:
        num_iterative = num_iterative + 1
        print("##########################################################################")
        print(f"Current iteration count：{num_iterative}")
        ###################################################################
        image = Image.open(iterative_image_path)
        image = image.resize((224, 224))
        iterative_image = np.array(image)

        # The R, G, B three channel matrix of the actual image
        iterative_image_channel_R = iterative_image[:, :, 0]
        iterative_image_channel_R = iterative_image_channel_R.astype(np.float64)
        iterative_image_channel_G = iterative_image[:, :, 1]
        iterative_image_channel_G = iterative_image_channel_G.astype(np.float64)
        iterative_image_channel_B = iterative_image[:, :, 2]
        iterative_image_channel_B = iterative_image_channel_B.astype(np.float64)

        # Standardize the actual image
        iterative_image_transform_matrix_R = ((iterative_image_channel_R / 255) - 0.485) / 0.229
        iterative_image_transform_matrix_G = ((iterative_image_channel_G / 255) - 0.456) / 0.224
        iterative_image_transform_matrix_B = ((iterative_image_channel_B / 255) - 0.406) / 0.225
        ####################################################################################################
        # Calculate the Top-1 category of the iterative image in this round
        iterative_image_top1_index, x = predict_image_path(iterative_image_path, index_file_absolute_path, weight_file_absolute_path, actual_image_index, model_current)

        gradient_matrix = calculate_gradient(iterative_image_path, actual_image_index, weight_file_absolute_path, model_current, num_iterative, img_name)
        iterative_image_pixel_weight_matrix_in_actual_label_R = gradient_matrix[0]
        iterative_image_pixel_weight_matrix_in_actual_label_G = gradient_matrix[1]
        iterative_image_pixel_weight_matrix_in_actual_label_B = gradient_matrix[2]

        mark_matrix_final = divide_matrix(np.abs(gradient_matrix), ratio)
        mark_matrix_final_R = mark_matrix_final[0]
        mark_matrix_final_G = mark_matrix_final[1]
        mark_matrix_final_B = mark_matrix_final[2]

        change_matrix_R = iterative_image_channel_R.copy()
        change_matrix_G = iterative_image_channel_G.copy()
        change_matrix_B = iterative_image_channel_B.copy()

        for i in range(224):
            for j in range(224):

                if mark_matrix_final_R[i][j] == 1:
                    if iterative_image_pixel_weight_matrix_in_actual_label_R[i][j] > 0:
                        iterative_image_transform_matrix_R[i][j] = iterative_image_transform_matrix_R[i][j] + iteration_step_size
                        change_matrix_R[i][j] = math.ceil((iterative_image_transform_matrix_R[i][j] * 0.229 + 0.485) * 255)
                    if iterative_image_pixel_weight_matrix_in_actual_label_R[i][j] < 0:
                        iterative_image_transform_matrix_R[i][j] = iterative_image_transform_matrix_R[i][j] - iteration_step_size
                        change_matrix_R[i][j] = int((iterative_image_transform_matrix_R[i][j] * 0.229 + 0.485) * 255)

                if mark_matrix_final_G[i][j] == 1:
                    if iterative_image_pixel_weight_matrix_in_actual_label_G[i][j] > 0:
                        iterative_image_transform_matrix_G[i][j] = iterative_image_transform_matrix_G[i][j] + iteration_step_size
                        change_matrix_G[i][j] = math.ceil((iterative_image_transform_matrix_G[i][j] * 0.224 + 0.456) * 255)
                    if iterative_image_pixel_weight_matrix_in_actual_label_G[i][j] < 0:
                        iterative_image_transform_matrix_G[i][j] = iterative_image_transform_matrix_G[i][j] - iteration_step_size
                        change_matrix_G[i][j] = int((iterative_image_transform_matrix_G[i][j] * 0.224 + 0.456) * 255)

                if mark_matrix_final_B[i][j] == 1:
                    if iterative_image_pixel_weight_matrix_in_actual_label_B[i][j] > 0:
                        iterative_image_transform_matrix_B[i][j] = iterative_image_transform_matrix_B[i][j] + iteration_step_size
                        change_matrix_B[i][j] = math.ceil((iterative_image_transform_matrix_B[i][j] * 0.225 + 0.406) * 255)
                    if iterative_image_pixel_weight_matrix_in_actual_label_B[i][j] < 0:
                        iterative_image_transform_matrix_B[i][j] = iterative_image_transform_matrix_B[i][j] - iteration_step_size
                        change_matrix_B[i][j] = int((iterative_image_transform_matrix_B[i][j] * 0.225 + 0.406) * 255)

        change_matrix_R = np.clip(
            change_matrix_R,
            actual_image_channel_R - degree,
            actual_image_channel_R + degree
        )

        change_matrix_G = np.clip(
            change_matrix_G,
            actual_image_channel_G - degree,
            actual_image_channel_G + degree
        )

        change_matrix_B = np.clip(
            change_matrix_B,
            actual_image_channel_B - degree,
            actual_image_channel_B + degree
        )

        change_matrix_R = np.clip(change_matrix_R, 0, 255)
        change_matrix_G = np.clip(change_matrix_G, 0, 255)
        change_matrix_B = np.clip(change_matrix_B, 0, 255)

        # Generate images that have been tampered with in this iteration
        image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B],axis=-1)
        image_rgb = image_rgb.astype(np.uint8)
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("target attack image.png")

        # Calculate and query whether the current attack image has been successfully attacked
        attack_image_path = "target attack image.png"
        attack_image_index, x = predict_image_path(attack_image_path, index_file_absolute_path,
                                                   weight_file_absolute_path, actual_image_index, model_current)


        if attack_image_index != actual_image_index:
            success_num = success_num + 1
            iterative_image_path = f"{output_directory}\\{image_num}.png"
            shutil.copy(attack_image_path, iterative_image_path)
        else:
            iterative_image_path = f"{output_directory}\\{image_num}.png"
            shutil.copy(attack_image_path, iterative_image_path)

save_loss_to_excel("loss_record.xlsx")
