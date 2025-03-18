# Install PyTorch if it's not already installed
!pip install torch torchvision torchaudio

%matplotlib inline

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
from PIL import Image  # Added for prediction on single image

# Define the paths to your dataset
train_path = r"C:\Users\chill\Desktop\Minor Project\archive\skin_disease_dataset\train_set"
test_path = r"C:\Users\chill\Desktop\Minor Project\archive\skin_disease_dataset\test_set"

# Check if the paths exist
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Train directory not found: {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test directory not found: {test_path}")

# Data augmentation and normalization for training
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

# Load the datasets
image_datasets = {
    'train': datasets.ImageFolder(train_path, data_transforms['train']),
    'val': datasets.ImageFolder(test_path, data_transforms['val'])
}
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=64, shuffle=False)
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

# Load the EfficientNetB2 model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b2(weights="EfficientNet_B2_Weights.IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).cpu().numpy()

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'train':
                scheduler.step()

    return model, history

# Train the model
model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=100)

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Set model to evaluation mode
model.eval()

# Initialize counters for accuracy calculation
correct = 0
total = 0

# Turn off gradients for validation/testing
with torch.no_grad():
    for inputs, labels in dataloaders['val']:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        # Count correct predictions
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Calculate accuracy
accuracy = correct / total * 100
print(f"Validation Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), 'skin_disease_efficientnetB2_model.pth')
print("Model saved as skin_disease_efficientnetB2_model.pth")

# Evaluate on test set and create confusion matrix
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in dataloaders['val']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:")
print(report)

# -------------------------------
# Added Prediction Function for a Provided Input Image
# -------------------------------
def predict_from_image(input_image, model, transform, class_names):
    """
    Predict the class of an input PIL image using the trained model.
    
    Args:
        input_image (PIL.Image.Image): The input image.
        model (torch.nn.Module): Trained model.
        transform (torchvision.transforms): Transformations to apply to the image.
        class_names (list): List of class names.
    
    Returns:
        str: Predicted class label.
    """
    model.eval()
    image = input_image.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    return predicted_class

# Example usage of the prediction function using an input image
# For demonstration, we'll open an image from disk.
# In practice, you can provide any PIL Image (for example, obtained from an input widget or camera).
test_image = Image.open("path/to/your/single/image.jpg")  # Replace with your image file or input method
predicted_class = predict_from_image(test_image, model, data_transforms['val'], class_names)
print(f"Predicted class for the provided image: {predicted_class}")
