

"""
This section of code is used for calculating the accuracy of the CIFAR-10 raw test set
"""
import torch
from torch.utils.data import DataLoader
import os
from torchvision import transforms, datasets
from torchvision.models import densenet121
from tqdm import tqdm  # Import tqdm for progress bar

# Set the device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Perform initialization operations on the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Import the testing dataset
test_dataset = datasets.ImageFolder(root="...", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load a pre-trained DenseNet121 model
model = densenet121(num_classes=10).to(device)
model.load_state_dict(torch.load("..."))
model.to(device)

# Set the model to evaluation mode
model.eval()

# Number of correctly predicted samples
correct = 0

# Total number of samples in the validation set
total = 0

# Disable gradient calculation during validation to save memory and computation time
with torch.no_grad():
    # Wrap the test_loader with tqdm for progress bar
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Compute accuracy
accuracy = 100 * correct / total
print(f"\nAccuracy: {accuracy:.2f}%")



"""This section of code is used for calculating the accuracy of the RSCD raw test set"""
import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import alexnet
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.cuda.amp as amp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def load_class_indices(file_path):
    class_dict = {}
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            parts = line.strip().rsplit('-', 1)
            class_name, class_index = parts[0], int(parts[1])
            class_dict[class_name] = class_index
    return class_dict

class ImageDataset(Dataset):
    def __init__(self, root_dir, class_dict, transform=None):
        self.root_dir = root_dir
        self.class_dict = class_dict
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file_path = os.path.join(self.root_dir, filename)
        class_name = '-'.join(filename.split('-')[1:]).replace('.png', '')
        label = self.class_dict.get(class_name, -1)
        img = Image.open(file_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def load_model(weight_path):
    model = alexnet(num_classes=8).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, dataloader):
    total_samples, correct_predictions = 0, 0
    scaler = amp.GradScaler() if device.type == 'cuda' else None

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", unit="batch", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if device.type == 'cuda':
                with amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    return accuracy, correct_predictions, total_samples

def main():

    index_file_path = r"..."
    weight_file_path = r"..."
    base_test_folder = r"..."

    dataset_classes = load_class_indices(index_file_path)

    model = load_model(weight_file_path)

    noise_types = [d for d in os.listdir(base_test_folder)
                   if os.path.isdir(os.path.join(base_test_folder, d))]

    intensities = ['1', '2', '3', '4', '5']

    accuracy_sums = {intensity: 0.0 for intensity in intensities}
    counts = {intensity: 0 for intensity in intensities}

    for noise_type in noise_types:
        noise_folder = os.path.join(base_test_folder, noise_type)

        for intensity in intensities:
            intensity_folder = os.path.join(noise_folder, intensity)

            dataset = ImageDataset(intensity_folder, dataset_classes, transform=data_transform)
            if len(dataset) == 0:
                continue

            dataloader = DataLoader(dataset, batch_size=256, shuffle=False,
                                   num_workers=8,
                                   pin_memory=True if device.type == 'cuda' else False,
                                   prefetch_factor=2 if device.type == 'cuda' else None)

            accuracy, _, _ = evaluate_model(model, dataloader)
            print(f"{noise_type} {intensity} {accuracy:.4f}%")

            accuracy_sums[intensity] += accuracy
            counts[intensity] += 1

    for intensity in intensities:
        if counts[intensity] > 0:
            avg_accuracy = accuracy_sums[intensity] / counts[intensity]
            print(f"accuracy_{intensity} = {avg_accuracy:.4f}%")
        else:
            print(f"accuracy_{intensity} = 0.00%")

if __name__ == "__main__":
    main()