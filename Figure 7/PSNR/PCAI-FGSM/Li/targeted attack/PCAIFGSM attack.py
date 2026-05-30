import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import alexnet
import os
from PCAIFGSM import PCAIFGSM
import json
import matplotlib.pyplot as plt
import numpy as np


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

def target_index_image_path(image_path, index_path, weight_path, model_cnn):
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
    top_probs, top_indices = torch.topk(classification_probability, 1000)
    attack_index = top_indices[499]
    return(attack_index)

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
    top500_indices = torch.topk(classification_probability, k=1000).indices
    top500_class_index = top500_indices[499].item()
    return(predicted_class_index, output[index], top500_class_index)

def pixel_weight_matrix_image_path(image_path, weight_path, index, model_cnn):
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Create model
    model = model_cnn(num_classes=1000).to(device)

    model.load_state_dict(torch.load(weight_path))

    # Set the model to evaluation mode
    model.eval()
    output = torch.squeeze(model(img.to(device))).cpu()
    classification_probability = torch.softmax(output, dim=0)

    top_probs, top_indices = torch.topk(classification_probability, 3)
    img = img.to(device)
    model.eval()
    img.requires_grad_()
    output = model(img)
    pred_score = output[0, index]
    pred_score.backward(retain_graph=True)
    gradients = img.grad

    channel_r = gradients[0, 0, :, :].cpu().detach().numpy()
    channel_g = gradients[0, 1, :, :].cpu().detach().numpy()
    channel_b = gradients[0, 2, :, :].cpu().detach().numpy()

    return channel_r, channel_g, channel_b

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The absolute path of the weight file
weights_path = r"..."

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# The maximum degree of tampering of a single pixel
degree = 16

psnr = 50
# Iterative step size
iteration_step_size = 0.001

# Calculate the maximum number of iterations
max_num_iterative = int(degree / 255 * (2.4285 - (-2.0357)) / iteration_step_size)

model_current = alexnet

# The absolute path to the folder where the original images are stored
folder_path = r"..."
# Save the folder path for the generated attack samples
save_path_for_adversarial_samples = r"..."

file_list = os.listdir(folder_path)
file_list = sorted(file_list, key=sort_func)

success_num = 0
image_num = 0

for file_name in file_list:
    image_num += 1
    print(file_name)
    image_path = os.path.join(folder_path, file_name)

    image = Image.open(image_path).convert('RGB')

    # actual image
    image = image.resize((224, 224))
    actual_image = np.array(image)

    # The R, G, B three channel matrix of the actual image
    actual_image_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    actual_image_matrix[0] = actual_image[:, :, 0]
    actual_image_matrix[0] = actual_image_matrix[0].astype(np.float64)
    actual_image_matrix[1] = actual_image[:, :, 1]
    actual_image_matrix[1] = actual_image_matrix[1].astype(np.float64)
    actual_image_matrix[2] = actual_image[:, :, 2]
    actual_image_matrix[2] = actual_image_matrix[2].astype(np.float64)

    actual_image_transform_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    actual_image_transform_matrix[0] = ((actual_image_matrix[0] / 255) - 0.485) / 0.229
    actual_image_transform_matrix[1] = ((actual_image_matrix[1] / 255) - 0.456) / 0.224
    actual_image_transform_matrix[2] = ((actual_image_matrix[2] / 255) - 0.406) / 0.225

    image = data_transform(image).unsqueeze(0).cuda()

    # Load image
    img_absolute_path = image_path
    assert os.path.exists(img_absolute_path), "file: '{}' dose not exist.".format(img_absolute_path)
    img = Image.open(img_absolute_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Read class_indict
    json_absolute_path = r"..."
    assert os.path.exists(json_absolute_path), "file: '{}' dose not exist.".format(json_absolute_path)

    with open(json_absolute_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = model_current(num_classes=1000).to(device)

    # Load model weights
    weights_absolute_path = weights_path
    assert os.path.exists(weights_absolute_path), "file: '{}' dose not exist.".format(weights_absolute_path)
    model.load_state_dict(torch.load(weights_absolute_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        top_probs, top_indices = torch.topk(predict, 1000)

    # When generating adversarial samples for aimless attacks, it is necessary to set the label in "[]" to the true label of the image
    label = torch.tensor([top_indices[499]]).cuda()

    attack_index = target_index_image_path(image_path, json_absolute_path, weights_path, model_current)

    atk = PCAIFGSM(model, alpha=0.001, steps=max_num_iterative, psnr=psnr)
    # Set to targeted attack mode
    atk.set_mode_targeted_by_label()
    adv_image = atk(image, label, actual_image_transform_matrix, actual_image_matrix, actual_image)


    image_rgb = np.stack([adv_image[0], adv_image[1], adv_image[2]], axis=-1)
    # Convert data type to 8-bit unsigned integer
    image_rgb = image_rgb.astype(np.uint8)
    # Create PIL image object
    image_pil = Image.fromarray(image_rgb)
    image_pil.save("target attack image.png")
    iterative_image_path = "target attack image.png"

    attack_image_index, x, _ = predict_image_path(iterative_image_path, json_absolute_path, weights_path, attack_index, model_current)
    if attack_image_index == attack_index:
        success_num = success_num + 1
    success_rate = success_num / image_num
    print(f"success_rate: {success_rate * 100:.2f}%")

    # Perform category prediction on the adversarial_sample_image
    actual_image_top1_index, x, actual_image_top500_index = predict_image_path(image_path, json_absolute_path,
                                                                               weights_path, 0, model_current)
    print(f"Top-1 category index number of the actual image:{actual_image_top1_index}")
    print(f"Top-500 category index number of the actual image:{actual_image_top500_index}")

    # Perform category prediction on the original image
    adversarial_sample_image_top1_index, x, _ = predict_image_path(iterative_image_path, json_absolute_path,
                                                                   weights_path, 0, model_current)
    print(f"Top-1 category index number of the adversarial sample:{adversarial_sample_image_top1_index}")

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
    #############################################################################################
    # adversarial sample image
    image = Image.open(iterative_image_path)
    image = image.resize((224, 224))
    adversarial_sample_image = np.array(image)

    # The R, G, B three channel matrix of the actual image
    adversarial_sample_image_channel_R = adversarial_sample_image[:, :, 0]
    adversarial_sample_image_channel_R = adversarial_sample_image_channel_R.astype(np.float64)
    adversarial_sample_image_channel_G = adversarial_sample_image[:, :, 1]
    adversarial_sample_image_channel_G = adversarial_sample_image_channel_G.astype(np.float64)
    adversarial_sample_image_channel_B = adversarial_sample_image[:, :, 2]
    adversarial_sample_image_channel_B = adversarial_sample_image_channel_B.astype(np.float64)

    # Standardize the actual image
    adversarial_sample_image_transform_matrix_R = ((adversarial_sample_image_channel_R / 255) - 0.485) / 0.229
    adversarial_sample_image_transform_matrix_G = ((adversarial_sample_image_channel_G / 255) - 0.456) / 0.224
    adversarial_sample_image_transform_matrix_B = ((adversarial_sample_image_channel_B / 255) - 0.406) / 0.225

    adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label = np.zeros((3, 224, 224), dtype=np.float64)
    adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label[0], \
        adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label[1], \
        adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label[2] = pixel_weight_matrix_image_path(
        iterative_image_path, weights_path, adversarial_sample_image_top1_index,
        model_current
    )
    #######################################################################################
    adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label = np.abs(
        adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label)
    # Flatten the matrix and calculate the absolute values
    flat_total_telationship = adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label.flatten()
    flat_abs_total_telationship = np.abs(flat_total_telationship)
    # Get the indices of the smallest 1000 absolute values
    smallest_indices = np.argpartition(flat_abs_total_telationship, 3 * 224 * 224 - 1)[:3 * 224 * 224]
    smallest_indices_sorted = smallest_indices[np.argsort(flat_abs_total_telationship[smallest_indices])]
    # Convert flat indices back to 3D indices (coordinates in the original matrix)
    index_matrix = np.array(
        [np.unravel_index(idx, adversarial_sample_image_pixel_weight_matrix_in_adversarial_sample_label.shape) for idx
         in smallest_indices_sorted])

    change_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    change_matrix[0] = adversarial_sample_image_channel_R.copy()
    change_matrix[1] = adversarial_sample_image_channel_G.copy()
    change_matrix[2] = adversarial_sample_image_channel_B.copy()

    actual_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    actual_matrix[0] = actual_image_channel_R.copy()
    actual_matrix[1] = actual_image_channel_G.copy()
    actual_matrix[2] = actual_image_channel_B.copy()

    iterative_num = 1
    flag = 1
    left = 0
    right = 3 * 224 * 224
    mid = (left + right) / 2

    last_matrix = np.zeros((3, 224, 224), dtype=np.float64)

    while (right - left != 1):
        print(iterative_num)
        print(left)
        print(right)
        print(mid)
        if flag == 1:
            last_matrix = change_matrix.copy()

        return_the_number_of_pixels = int(mid)
        change_matrix = np.zeros((3, 224, 224), dtype=np.float64)
        change_matrix[0] = adversarial_sample_image_channel_R.copy()
        change_matrix[1] = adversarial_sample_image_channel_G.copy()
        change_matrix[2] = adversarial_sample_image_channel_B.copy()

        for i in range(return_the_number_of_pixels):
            change_matrix[index_matrix[i][0]][index_matrix[i][1]][index_matrix[i][2]] = \
                actual_matrix[index_matrix[i][0]][index_matrix[i][1]][index_matrix[i][2]]

        # Combine three channels into an RGB image
        image_rgb = np.stack([change_matrix[0], change_matrix[1], change_matrix[2]], axis=-1)
        # Convert data type to 8-bit unsigned integer
        image_rgb = image_rgb.astype(np.uint8)
        # Create PIL image object
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("target.png")
        iterative_image_path = "target.png"

        iterative_image_current_top1_label, x, _ = predict_image_path(iterative_image_path, json_absolute_path,
                                                                      weights_path, 0, model_current)

        if iterative_image_current_top1_label != actual_image_top500_index:
            flag = 0
            right = mid
            mid = (left + right) // 2

        # iterative_num = iterative_num + 1
        else:
            flag = 1
            left = mid
            mid = (left + right) // 2

        iterative_num = iterative_num + 1


    image_rgb = np.stack([last_matrix[0], last_matrix[1], last_matrix[2]], axis=-1)
    # Convert data type to 8-bit unsigned integer
    image_rgb = image_rgb.astype(np.uint8)
    # Create PIL image object
    image_pil = Image.fromarray(image_rgb)
    new_image_name = str(file_name)
    new_image_path = os.path.join(save_path_for_adversarial_samples, new_image_name)
    image_pil.save(new_image_path)