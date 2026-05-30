import torch
import torch.nn as nn
from PIL import Image
import json
import torchvision.transforms as transforms
from torchvision.models import alexnet
from attack import Attack
import numpy as np
from scipy.linalg import eigh
import math
from skimage.metrics import structural_similarity as ssim
from skimage import io

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def predict_image_from_rgb_matrices(r_matrix, g_matrix, b_matrix, index_path, weight_path, index, model_cnn):
    img_array = np.stack([r_matrix, g_matrix, b_matrix], axis=-1).astype(np.uint8)
    img = Image.fromarray(img_array)
    img_tensor = data_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img_tensor.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    predicted_class_index = torch.argmax(classification_probability).item()

    return predicted_class_index, output[index].item()


def con_transform(actual_image_transform_matrix, adversarial_sample_transform_matrix, actual_image_matrix):

    adv_image = actual_image_matrix.copy()


    adversarial_sample_transform_matrix = adversarial_sample_transform_matrix.cpu().numpy()


    factors = np.array([[0.229, 0.485], [0.224, 0.456], [0.225, 0.406]])
    scales = np.array([255, 255, 255])


    for c in range(3):

        actual_transform = actual_image_transform_matrix[c]
        adversarial_transform = adversarial_sample_transform_matrix[c]
        actual_image = actual_image_matrix[c]


        mask_greater = adversarial_transform > actual_transform
        mask_less = adversarial_transform < actual_transform
        mask_equal = adversarial_transform == actual_transform


        adv_image[c] = np.where(mask_greater,
                                np.ceil((adversarial_transform * factors[c][0] + factors[c][1]) * scales[c]),
                                np.where(mask_less,
                                         np.floor((adversarial_transform * factors[c][0] + factors[c][1]) * scales[c]),
                                         np.where(mask_equal, actual_image, adv_image[c])))


    adv_image = np.clip(adv_image, 0, 255)

    return adv_image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PCAIFGSM(Attack):
    def __init__(self, model, eps=float(np.sqrt((8 ** 2) * 3 * 224 * 224)), alpha=1 / 255, steps=10, random_start=True, vcr=0.9, ssim=0.9):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.vcr = vcr
        self.ssim = ssim

    def forward(self, images, labels, actual_image_transform_matrix, actual_image_matrix, actual_image):

        weights_path = r"..."
        json_absolute_path = r"..."

        model_current = alexnet

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images[0][0] = torch.clamp(images[0][0], min=-2.1179, max=2.2489).detach()
            adv_images[0][1] = torch.clamp(images[0][1], min=-2.0357, max=2.4285).detach()
            adv_images[0][2] = torch.clamp(images[0][2], min=-1.8044, max=2.64).detach()

        adv_image_copy = actual_image_matrix

        for _ in range(self.steps):
            print(_+1)
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

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
            k = np.argwhere(cumulative_variance >= self.vcr)[0][0] + 1
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

            # min_value = max(min_value_R, min_value_G, min_value_B)
            min_value = math.sqrt(min_value_R ** 2 + min_value_G ** 2 + min_value_B ** 2)
            average = np.mean(grad_np_abs)
            print(min_value, average)

            mask = torch.zeros_like(grad)
            mask[grad_np_abs > min_value] = 1

            grad = grad * mask

            adv_images = adv_images.detach() + self.alpha * grad.sign()

            delta = adv_images - images

            adv_images[0][0] = torch.clamp(images[0][0] + delta[0][0], min=-2.1179, max=2.2489).detach()
            adv_images[0][1] = torch.clamp(images[0][1] + delta[0][1], min=-2.0357, max=2.4285).detach()
            adv_images[0][2] = torch.clamp(images[0][2] + delta[0][2], min=-1.8044, max=2.64).detach()

            adv_image = adv_images.squeeze(0).cpu()
            adv_image = con_transform(actual_image_transform_matrix, adv_image, actual_image_matrix)

            actual_delta = adv_image - actual_image_matrix

            actual_delta_L2 = np.linalg.norm(actual_delta)

            if actual_delta_L2 > self.eps:
                return previous_adv_image

            previous_adv_image = adv_images

            iterative_image_top1_label, x = predict_image_from_rgb_matrices(adv_image[0], adv_image[1], adv_image[2],
                                                                            json_absolute_path, weights_path, labels.item(),
                                                                            model_current)
            image_np = actual_image_matrix.transpose((1, 2, 0))
            adv_image_np = adv_image.transpose((1, 2, 0))

            SSIM = ssim(image_np, adv_image_np, win_size=3, data_range=255)

            if SSIM < self.ssim:
                return adv_image_copy

            adv_image_copy = adv_image

            if iterative_image_top1_label == labels:
                return adv_image

        return adv_image
