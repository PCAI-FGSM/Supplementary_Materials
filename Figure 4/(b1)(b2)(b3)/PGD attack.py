import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import alexnet
import os
from PGD import PGD
import json
import matplotlib.pyplot as plt
import numpy as np


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model_current = alexnet

# The absolute path to the folder where the original images are stored
folder_path = r"..."

# Save the folder path for the generated attack samples
save_path_for_adversarial_samples = "..."

# The absolute path of the weight file
weights_path = "..."

file_list = os.listdir(folder_path)
file_list = sorted(file_list, key=sort_func)

for file_name in file_list:
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
    json_absolute_path = "..."
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
    label = torch.tensor([top_indices[0]]).cuda()
    mask_bank = {}

    for degree in range(2, 41, 2):
        # Iterative step size
        iteration_step_size = 0.01
        # Calculate the maximum number of iterations
        max_num_iterative = int(degree / 255 * (2.4285 - (-2.0357)) / iteration_step_size)

        if degree == 2:
            shape = (3, 224, 224)
            total_elements = torch.tensor(shape).prod().item()
            full_index_order = torch.randperm(total_elements)
            mask_template = torch.zeros(total_elements, dtype=torch.uint8)

            for step in range(1, 21):
                ratio = step * 0.05
                ratio_int = step * 5
                print(f"The current tampering ratio is：{ratio}")
                num_pixels_to_tamper = int(total_elements * ratio)
                current_indices = full_index_order[:num_pixels_to_tamper]
                mask = mask_template.clone()
                mask[current_indices] = 1
                mask_bank[ratio_int] = mask.view(shape)

                atk = PGD(model, alpha=0.01, steps=max_num_iterative)
                adv_image, flag = atk(image, label, actual_image_transform_matrix, actual_image_matrix,
                                      mask.view(shape), degree)
                if flag == 1:
                    break
        else:
            for ratio_int in sorted(mask_bank.keys()):
                mask = mask_bank[ratio_int]
                max_num_iterative = int(degree / 255 * (2.4285 - (-2.0357)) / 0.01)

                atk = PGD(model, alpha=0.01, steps=max_num_iterative)
                adv_image, flag = atk(image, label, actual_image_transform_matrix, actual_image_matrix, mask, degree)

                if flag == 1:
                    break
        image_rgb = np.stack([adv_image[0], adv_image[1], adv_image[2]], axis=-1)
        image_rgb = image_rgb.astype(np.uint8)
        image_pil = Image.fromarray(image_rgb)

        subfolder_path = os.path.join(save_path_for_adversarial_samples, str(degree))
        os.makedirs(subfolder_path, exist_ok=True)
        new_image_path = os.path.join(subfolder_path, file_name)
        image_pil.save(new_image_path)