import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Define dataset paths
train_path = r"skin_disease_dataset\train_set"
test_path = r"skin_disease_dataset\test_set"

# Ensure dataset paths exist
for path in [train_path, test_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(train_path, transform=data_transforms['train']),
    'val': datasets.ImageFolder(test_path, transform=data_transforms['val'])
}

# Dataloaders with optimizations
dataloaders = {
    phase: torch.utils.data.DataLoader(
        image_datasets[phase], batch_size=64, shuffle=(phase == 'train'),
        num_workers=4, pin_memory=True
    )
    for phase in ['train', 'val']
}

# Dataset size and class info
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

# Load EfficientNetB2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b2(weights="EfficientNet_B2_Weights.IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# Loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Mixed precision training for speedup
scaler = torch.cuda.amp.GradScaler()


# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n" + "-" * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'), torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        scheduler.step()

    return model, history


# Train the model
model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=10)

# Plot training results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Model evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    y_pred, y_true = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    accuracy = correct / total * 100
    return accuracy, y_true, y_pred


# Validate and print accuracy
val_accuracy, y_true, y_pred = evaluate_model(model, dataloaders['val'])
print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), 'optimized_skin_disease_model.pth')
print("Model saved as optimized_skin_disease_model.pth")

# Confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
