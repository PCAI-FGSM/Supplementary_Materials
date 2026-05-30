"""
The current code file is used to calculate the recognition accuracy of convolutional neural network models on noisy image test sets
Firstly, you need to set the parameter 'weight_file_path' to the absolute path of the model weight file
Secondly, you need to set the parameter 'noisy_image_file_path' to the absolute path of the test set containing noisy images
Thirdly, you need to set the parameter 'label_file_path' to the absolute path of the noise test set image label file
"""

import torch
from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.models import alexnet
from tqdm import tqdm

# Perform initialization operations on the images
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
     ])

# Set the device based on CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


class CustomDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.label_file = label_file
        self.transform = transform
        self.image_paths = sorted(
            [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.png')],
            key=sort_func)
        with open(label_file, 'r', encoding='utf-8') as f:
            self.true_labels = [int(line.strip()) for line in f.readlines()]

        # Ensure label count matches image count
        if len(self.true_labels) != len(self.image_paths):
            raise ValueError(f"Label count {len(self.true_labels)} does not match image count {len(self.image_paths)}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.true_labels[idx]
        return img, label, os.path.basename(img_path)


def predict_batch(model, dataloader):
    all_predictions = []
    all_labels = []
    all_file_names = []

    model.eval()
    with torch.no_grad():
        for images, labels, file_names in tqdm(dataloader, desc="Predicting", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_file_names.extend(file_names)

    return all_predictions, all_labels, all_file_names


def calculate_accuracy(predicted_labels, true_labels):
    correct_predictions = sum(1 for p, t in zip(predicted_labels, true_labels) if p == t)
    total_images = len(true_labels)
    if total_images > 0:
        accuracy = correct_predictions / total_images * 100
        return accuracy, correct_predictions, total_images
    else:
        return 0, 0, 0


def main():
    weight_file_path = r"..."

    noisy_image_file_path = r"..."

    label_file_path = r"..."

    # Load the model once
    model = alexnet(num_classes=10).to(device)
    model.load_state_dict(torch.load(weight_file_path))

    # Iterate over each subfolder (1 to 5)
    subfolders = ['1', '2', '3', '4', '5']

    for subfolder in subfolders:
        print(f"\nProcessing folder: {subfolder}")
        # Prepare dataset and dataloader for the current subfolder
        current_folder_path = os.path.join(noisy_image_file_path, subfolder)
        dataset = CustomDataset(current_folder_path, label_file_path, transform=data_transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

        # Predict in batches
        predicted_labels, true_labels, file_names = predict_batch(model, dataloader)

        # Calculate accuracy for the current folder
        accuracy, correct_predictions, total_images = calculate_accuracy(predicted_labels, true_labels)

        # Output accuracy for the current folder
        print(f"Folder {subfolder} accuracy: {accuracy:.2f}% ({correct_predictions}/{total_images} images correctly classified)")

if __name__ == "__main__":
    main()
