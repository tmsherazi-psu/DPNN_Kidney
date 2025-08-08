import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# --- Dual-Path Neural Network (DPNN) Architecture ---
class DPNN(nn.Module):
    def __init__(self, num_classes=4):  # 4 classes: Cyst, Stone, Tumor, Normal
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


# --- Label Smoothing CrossEntropy Loss ---
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


# --- Agent Class for Training and Evaluation ---
class Agent_DPNN:
    def __init__(self, model, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(config['device'])
        self.config = config
        self.epoch = 0

    def train(self, data_loader, loss_function):
        self.model.train()
        for epoch in range(self.config['n_epoch']):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = loss_function(outputs, labels)

                # Backward pass
                loss.backward()

                # Optimize
                self.optimizer.step()

                running_loss += loss.item()

                if i % self.config['save_interval'] == 0:  # Save model at intervals
                    print(f"Epoch [{epoch + 1}/{self.config['n_epoch']}], Step [{i + 1}], Loss: {loss.item():.4f}")

            # Step the scheduler
            self.scheduler.step()
            print(f'Epoch [{epoch + 1}/{self.config["n_epoch"]}], Average Loss: {running_loss / len(data_loader):.4f}')

    def evaluate(self, data_loader, loss_function):
        self.model.eval()
        total_dice_score = 0.0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute Dice score or any other evaluation metrics
                dice_score = self.compute_dice_score(outputs, labels)
                total_dice_score += dice_score

        average_dice_score = total_dice_score / len(data_loader)
        return average_dice_score

    def compute_dice_score(self, outputs, labels):
        smooth = 1e-5
        outputs = torch.sigmoid(outputs)
        intersection = (outputs * labels).sum()
        dice = (2. * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)
        return dice.item()


# --- Configuration for Training ---
config = {
    'device': 'cuda:0',
    'lr': 1e-3,  # Learning rate set to 0.001
    'lr_gamma': 0.9999,
    'n_epoch': 50,  # 50 epochs as per the manuscript
    'batch_size': 32,
    'save_interval': 10,
    'evaluate_interval': 10,
    'optimizer': 'Adam',  # Adam optimizer
    'scheduler': 'CyclicLR',  # CyclicLR scheduler
    'loss_function': 'Smoothed Cross-Entropy Loss',  # Label smoothing cross-entropy loss
}


# --- Initialize Dataset and Dataloader ---
# Replace with actual dataset path
dataset = YourCustomDataset("image_path", "label_path")  # Replace with actual dataset
device = torch.device(config['device'])
model = DPNN(num_classes=4).to(device)  # 4 classes: Cyst, Stone, Tumor, Normal

# Optimizer and Scheduler
optimizer = Adam(model.parameters(), lr=config['lr'])
scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, step_size_up=2000, mode='triangular2')

# Loss function
loss_function = LabelSmoothingCrossEntropy(alpha=0.1, num_classes=4)

# Initialize the agent for DPNN
agent = Agent_DPNN(model, optimizer, scheduler, config)

# Set up data loader (adjust with actual DataLoader and dataset)
data_loader = DataLoader(dataset, shuffle=True, batch_size=config['batch_size'])

# Train the model
agent.train(data_loader, loss_function)
