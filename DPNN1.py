# Import Required Libraries
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# --- Custom Label Smoothing Loss ---
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, num_classes=4):  # 4 classes: Cyst, Stone, Tumor, Normal
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # Convert one-hot targets to class indices
        targets = targets.argmax(dim=1)
        # Smooth labels
        targets = torch.clamp(targets.long(), 0, self.num_classes - 1)
        n = inputs.size()[-1]
        log_preds = F.log_softmax(inputs, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, targets, reduction='mean')
        return (1 - self.alpha) * nll + self.alpha * (loss / n)


# --- Dual-Path Neural Network (DPNN) Model Architecture ---
class DPNN(nn.Module):
    def __init__(self, num_classes=4):
        super(DPNN, self).__init__()

        # Shallow Pathway: Local feature extraction
        self.shallow_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.shallow_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shallow_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.shallow_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Deep Pathway: Global feature extraction
        self.deep_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.deep_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.deep_conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3)
        self.deep_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fusion Layer
        self.fusion_conv = nn.Conv2d(128 * 2, 256, kernel_size=3, stride=1, padding=1)
        self.fusion_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # Adjust size based on input dimensions
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Shallow Pathway
        shallow = F.relu(self.shallow_conv1(x))
        shallow = self.shallow_pool1(shallow)
        shallow = F.relu(self.shallow_conv2(shallow))
        shallow = self.shallow_pool2(shallow)

        # Deep Pathway
        deep = F.relu(self.deep_conv1(x))
        deep = self.deep_pool1(deep)
        deep = F.relu(self.deep_conv2(deep))
        deep = self.deep_pool2(deep)

        # Fusion Layer: Concatenating shallow and deep features
        combined = torch.cat((shallow, deep), dim=1)
        fused = F.relu(self.fusion_conv(combined))
        fused = self.fusion_pool(fused)

        # Flatten and Fully Connected Layers
        fused = fused.view(fused.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(fused))
        x = self.fc2(x)

        return x


# --- Dataset Loading & Transformation ---
def load_data():
    base_dir = "/content/drive/MyDrive/data"

    def load_images_from_folder(folder):
        images = []
        for filename in tqdm(os.listdir(folder)):
            img_path = os.path.join(folder, filename)
            if img_path.endswith(".png"):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                images.append(img)
        return np.array(images)

    train_cyst_dir = os.path.join(base_dir, "Train/cyst")
    train_stone_dir = os.path.join(base_dir, "Train/stone")
    train_tumor_dir = os.path.join(base_dir, "Train/tumor")
    train_normal_dir = os.path.join(base_dir, "Train/normal")

    test_cyst_dir = os.path.join(base_dir, "Validation/cyst")
    test_stone_dir = os.path.join(base_dir, "Validation/stone")
    test_tumor_dir = os.path.join(base_dir, "Validation/tumor")
    test_normal_dir = os.path.join(base_dir, "Validation/normal")

    cyst_train = load_images_from_folder(train_cyst_dir)
    stone_train = load_images_from_folder(train_stone_dir)
    tumor_train = load_images_from_folder(train_tumor_dir)
    normal_train = load_images_from_folder(train_normal_dir)

    cyst_test = load_images_from_folder(test_cyst_dir)
    stone_test = load_images_from_folder(test_stone_dir)
    tumor_test = load_images_from_folder(test_tumor_dir)
    normal_test = load_images_from_folder(test_normal_dir)

    # Labels: 0 - Cyst, 1 - Stone, 2 - Tumor, 3 - Normal
    cyst_train_label = np.zeros(len(cyst_train))
    stone_train_label = np.ones(len(stone_train))
    tumor_train_label = 2 * np.ones(len(tumor_train))
    normal_train_label = 3 * np.ones(len(normal_train))

    cyst_test_label = np.zeros(len(cyst_test))
    stone_test_label = np.ones(len(stone_test))
    tumor_test_label = 2 * np.ones(len(tumor_test))
    normal_test_label = 3 * np.ones(len(normal_test))

    X_train = np.concatenate((cyst_train, stone_train, tumor_train, normal_train), axis=0)
    Y_train = np.concatenate((cyst_train_label, stone_train_label, tumor_train_label, normal_train_label), axis=0)

    X_test = np.concatenate((cyst_test, stone_test, tumor_test, normal_test), axis=0)
    Y_test = np.concatenate((cyst_test_label, stone_test_label, tumor_test_label, normal_test_label), axis=0)

    s = np.arange(X_train.shape[0])
    np.random.shuffle(s)
    X_train, Y_train = X_train[s], Y_train[s]

    s = np.arange(X_test.shape[0])
    np.random.shuffle(s)
    X_test, Y_test = X_test[s], Y_test[s]

    Y_train = np.eye(4)[Y_train.astype(int)]
    Y_test = np.eye(4)[Y_test.astype(int)]

    X_train = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    X_test = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(dim=1)).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.argmax(dim=1)).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    return train_losses, val_losses, train_accuracies, val_accuracies


# --- Evaluation Function ---
def evaluate_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.argmax(dim=1).cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# --- Main Execution ---
def main():
    train_loader, val_loader = load_data()

    # List of models to compare
    models_dict = {
        "CNN": CNNModel(num_classes=4),
        "mVGG19": models.vgg19(pretrained=True),
        "InceptionV3": models.inception_v3(pretrained=True),
        "ResNet152V2": models.resnet152(pretrained=True),
        "EfficientNetB7": models.efficientnet_b7(pretrained=True),
        "AlexNet": models.alexnet(pretrained=True),
        "EANet": EANet(),  # Define this model if available
        "DPNN": DPNN(num_classes=4)
    }

    for model_name, model in models_dict.items():
        print(f"Training {model_name}...")
        criterion = LabelSmoothingCrossEntropy(alpha=0.1, num_classes=4)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer)

        print(f"Evaluating {model_name}...")
        evaluate_model(model, val_loader)


if __name__ == "__main__":
    main()
