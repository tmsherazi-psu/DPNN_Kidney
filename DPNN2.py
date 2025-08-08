# Import required libraries
import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# Apply CLAHE for contrast enhancement (as per manuscript preprocessing steps)
def apply_clahe(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_clahe = cv2.merge((l, a, b))
    return cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)


# Dataset loader with CLAHE enhancement option
def Dataset_loader(DIR, RESIZE, use_clahe=True):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR, IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)
            img = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)
            if use_clahe:
                img = apply_clahe(img)  # Apply CLAHE enhancement
            IMG.append(np.array(img))
    return IMG


# Load dataset
cyst_train = np.array(Dataset_loader('/content/drive/MyDrive/data/Train/cyst', 224))
stone_train = np.array(Dataset_loader('/content/drive/MyDrive/data/Train/stone', 224))
tumor_train = np.array(Dataset_loader('/content/drive/MyDrive/data/Train/tumor', 224))
normal_train = np.array(Dataset_loader('/content/drive/MyDrive/data/Train/normal', 224))

cyst_test = np.array(Dataset_loader('/content/drive/MyDrive/data/Validation/cyst', 224))
stone_test = np.array(Dataset_loader('/content/drive/MyDrive/data/Validation/stone', 224))
tumor_test = np.array(Dataset_loader('/content/drive/MyDrive/data/Validation/tumor', 224))
normal_test = np.array(Dataset_loader('/content/drive/MyDrive/data/Validation/normal', 224))

# Create labels (4 classes: Cyst, Stone, Tumor, Normal)
cyst_train_label = np.zeros(len(cyst_train))
stone_train_label = np.ones(len(stone_train))
tumor_train_label = np.full(len(tumor_train), 2)
normal_train_label = np.full(len(normal_train), 3)

cyst_test_label = np.zeros(len(cyst_test))
stone_test_label = np.ones(len(stone_test))
tumor_test_label = np.full(len(tumor_test), 2)
normal_test_label = np.full(len(normal_test), 3)

# Merge and shuffle datasets
X_train = np.concatenate((cyst_train, stone_train, tumor_train, normal_train), axis=0)
Y_train = np.concatenate((cyst_train_label, stone_train_label, tumor_train_label, normal_train_label), axis=0)
X_test = np.concatenate((cyst_test, stone_test, tumor_test, normal_test), axis=0)
Y_test = np.concatenate((cyst_test_label, stone_test_label, tumor_test_label, normal_test_label), axis=0)

# Shuffle the data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train, Y_train = X_train[s], Y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test, Y_test = X_test[s], Y_test[s]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
X_test = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
Y_train = torch.tensor(Y_train, dtype=torch.long)
Y_test = torch.tensor(Y_test, dtype=torch.long)

# Wrap into DataLoader for efficient batching
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Label Smoothing CrossEntropy Loss (as per manuscript)
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, num_classes=4):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        targets = targets.view(-1)
        log_preds = F.log_softmax(inputs, dim=-1)
        nll_loss = F.nll_loss(log_preds, targets, reduction='none')
        smooth_loss = -log_preds.mean(dim=-1)
        loss = (1 - self.alpha) * nll_loss + self.alpha * smooth_loss
        return loss.mean()


# Dual-Path Neural Network (DPNN) Architecture
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


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=10):
    device = next(model.parameters()).device
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if scheduler:
            scheduler.step()

    return train_losses, val_losses, train_accuracies, val_accuracies


# Evaluation function
def evaluate_model(model, test_loader):
    y_true, y_pred = [], []

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cyst', 'Stone', 'Tumor', 'Normal'],
                yticklabels=['Cyst', 'Stone', 'Tumor', 'Normal'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Plotting function
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title("Accuracy Curve")
    plt.legend()
    plt.show()


# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPNN(num_classes=4).to(device)
    criterion = LabelSmoothingCrossEntropy(alpha=0.1, num_classes=4)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.9999)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler=scheduler, epochs=50
    )

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    evaluate_model(model, test_loader)

    torch.save(model.state_dict(), 'dpnn_model.pth')


if __name__ == "__main__":
    main()
