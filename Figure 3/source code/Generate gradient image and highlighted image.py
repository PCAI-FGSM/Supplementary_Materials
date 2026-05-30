import json
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
from torchvision.models import alexnet
from scipy.linalg import eigh
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 图像预处理
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def get_top_gradients(image, weight_path, index_path, model_cnn):
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    with open(index_path, "r") as f:
        class_indict = json.load(f)

    model = model_cnn(num_classes=len(class_indict)).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)

    top2_probs, top2_indices = torch.topk(classification_probability, k=2)
    top1_index = top2_indices[0].item()
    top2_index = top2_indices[1].item()

    def calculate_gradient(index):
        img_copy = img.clone().to(device)
        img_copy.requires_grad_()
        model.eval()
        output = model(img_copy)
        pred_score = output[0, index]
        pred_score.backward(retain_graph=True)
        gradients = img_copy.grad.cpu().detach()
        return gradients

    top1_gradient = calculate_gradient(top1_index)
    top2_gradient = calculate_gradient(top2_index)
    return top1_gradient.squeeze(0), top2_gradient.squeeze(0), top1_index, top2_index


def apply_pca_analysis(grad):
    grad_np = grad.cpu().numpy()
    grad_np_abs = np.abs(grad_np)
    data = grad_np_abs


    flattened_data = data.reshape(3, -1)


    covariance_matrix = np.cov(flattened_data)
    eigenvalues, eigenvectors = eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)


    vcr = 0.9
    k = np.argwhere(cumulative_variance >= vcr)[0][0] + 1
    selected_eigenvectors = sorted_eigenvectors[:, :k]


    principal_components = np.dot(selected_eigenvectors.T, flattened_data)


    projection_strength = np.abs(principal_components)


    projection_per_pixel = np.sqrt(np.sum(projection_strength ** 2, axis=0))


    mean_projection = np.mean(projection_per_pixel)
    std_projection = np.std(projection_per_pixel)
    threshold = mean_projection + 4 * std_projection


    significant_pixel_indices = np.where(projection_per_pixel >= threshold)[0]


    original_data_flat_R = flattened_data[0]
    original_data_flat_G = flattened_data[1]
    original_data_flat_B = flattened_data[2]

    significant_pixel_values_R = original_data_flat_R[significant_pixel_indices]
    significant_pixel_values_G = original_data_flat_G[significant_pixel_indices]
    significant_pixel_values_B = original_data_flat_B[significant_pixel_indices]


    if significant_pixel_values_R.size == 0:
        min_value_R = 0
    else:
        min_value_R = np.min(significant_pixel_values_R)

    if significant_pixel_values_G.size == 0:
        min_value_G = 0
    else:
        min_value_G = np.min(significant_pixel_values_G)

    if significant_pixel_values_B.size == 0:
        min_value_B = 0
    else:
        min_value_B = np.min(significant_pixel_values_B)


    min_value = math.sqrt(min_value_R ** 2 + min_value_G ** 2 + min_value_B ** 2)


    mask = torch.zeros_like(grad)
    mask[torch.from_numpy(grad_np_abs) > min_value] = 1

    return mask, min_value


def highlight_important_regions(original_img, grad, output_path):

    mask, threshold = apply_pca_analysis(grad)


    if original_img.mode != 'RGB':
        original_img = original_img.convert('RGB')


    resized_img = original_img.resize((224, 224))
    img_np = np.array(resized_img).astype(np.float32) / 255.0


    red_highlight = np.zeros_like(img_np)
    red_highlight[:, :, 0] = 1.0


    mask_np = mask.numpy().transpose(1, 2, 0)
    mask_np = np.all(mask_np > 0, axis=2)


    alpha = 0.5
    alpha_mask = np.zeros_like(mask_np, dtype=np.float32)
    alpha_mask[mask_np] = alpha


    highlighted_img = img_np.copy()
    for i in range(3):
        highlighted_img[:, :, i] = (1 - alpha_mask) * img_np[:, :, i] + alpha_mask * red_highlight[:, :, i]

    highlighted_img_pil = Image.fromarray((highlighted_img * 255).astype(np.uint8))
    highlighted_img_pil.save(output_path)

    return output_path, threshold


def process_single_image(image_path, output_folder, index_path, weight_path, model_cnn):

    try:

        assert os.path.exists(image_path), "error"

        image_name = os.path.splitext(os.path.basename(image_path))[0]

        os.makedirs(output_folder, exist_ok=True)

        img = Image.open(image_path).convert('RGB')

        top1_gradient, top2_gradient, top1_index, top2_index = get_top_gradients(
            img, weight_path, index_path, model_cnn)

        output_path = os.path.join(output_folder, f"{image_name}_gradient.png")
        save_gradient_channels(top1_gradient, image_name, output_folder, "top1")

        highlighted_output_path = os.path.join(output_folder, f"{image_name}_highlighted.png")
        _, threshold = highlight_important_regions(img, top1_gradient, highlighted_output_path)


    except Exception as e:
        print(f"error: {str(e)}")


def save_gradient_channels(gradient, image_name, output_folder, prefix="top1"):

    os.makedirs(output_folder, exist_ok=True)


    gradient_np = gradient.permute(1, 2, 0).numpy()


    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = gradient_np[:, :, i]


        normalized_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)


        channel_img = Image.fromarray((normalized_data * 255).astype(np.uint8), mode='L')
        channel_output_path = os.path.join(output_folder, f"{image_name}_{prefix}_gradient_{channel}.png")
        channel_img.save(channel_output_path)


if __name__ == "__main__":

    single_image_path = r"..."

    index_file_absolute_path = r"..."
    weight_file_absolute_path = r"..."


    output_folder = os.path.join(os.getcwd(), "gradient_images")


    process_single_image(
        image_path=single_image_path,
        output_folder=output_folder,
        index_path=index_file_absolute_path,
        weight_path=weight_file_absolute_path,
        model_cnn=alexnet
    )