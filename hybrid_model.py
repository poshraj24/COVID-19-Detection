from Training_Evaluation import *
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

#Combining CNN and Resnet 

#########DATA LOADING#####################
# Initialize the dataset
dataset = CovidDataset(COVID_FOLDER, NON_COVID_FOLDER, transform=transform)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Initialize the dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

#########MODEL DEFINITION#####################

# Load the pre-trained ResNet18 model
resnet = models.resnet18(pretrained=True)

# Remove the fully connected layer (fc) from ResNet
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fc layer
# This gives us a feature extractor with ResNet

# Freeze ResNet layers to keep the pre-trained weights unchanged
for param in resnet.parameters():
    param.requires_grad = False

# Define a new class to combine ResNet and your CNN model
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        
        # Pre-trained ResNet for feature extraction
        self.resnet = resnet
        
        # CNN layers after ResNet
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1)  # Adjust input channels from ResNet
        self.bn1 = nn.BatchNorm2d(128) #batch normalization
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Swish activation (using SiLU which is equivalent to Swish)
        self.swish = nn.SiLU()

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce to [batch_size, 64, 1, 1]

        # Fully connected layers
        self.fc1 = nn.Linear(64, 64)  # 64 input features from the global average pooling
        self.dropout50 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)  # Output layer for binary classification
        
    def forward(self, x):
        # Extract features with ResNet (ResNet output will be [batch_size, 512, 1, 1])
        x = self.resnet(x)
        
            # CNN layers after ResNet
        x = self.conv1(x)               # Apply first convolutional layer
        x = self.bn1(x)                 # Apply first batch normalization
        x = self.swish(x)               # Apply activation (Swish)

        x = self.conv2(x)               # Apply second convolutional layer
        x = self.bn2(x)                 # Apply second batch normalization
        x = self.swish(x)               # Apply activation (Swish)
        
        # Global Average Pooling (output: [batch_size, 64, 1, 1])
        x = self.global_avg_pool(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # [batch_size, 64]
        
        # Fully connected layers
        x = self.swish(self.fc1(x))     # First fully connected layer with Swish activation
        x = self.dropout50(x)           # Apply dropout
        x = torch.sigmoid(self.fc2(x))  # Output layer with sigmoid for binary classification
        return x
        
        

# Initialize the hybrid model
model = HybridModel()

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#############OPTIMIZATION######################
# Set L1 regularization parameter
l1_lambda = 1e-4
# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001,betas=(0.8, 0.999), weight_decay=1e-5 )  # Train all layers (including custom CNN layers)

# Define learning rate scheduler (StepLR: reduces LR by gamma every step_size epochs)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Halve the LR every 5 epochs

#########TRAINING######################
# Store losses and accuracies for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop (same as before)
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # Reshape labels
        
        optimizer.zero_grad()  # Zero the gradients
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)

        # Compute L1 regularization loss (sum of absolute values of model parameters)
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        
        # Combine original loss with L1 regularization
        total_loss = loss + l1_lambda * l1_loss
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        if i % 10 == 9:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0
        
        # Update training metrics
        running_loss += total_loss.item()
        predicted = (outputs > 0.5).float()
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = 100 * correct_train / total_train
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_accuracy)

    # Validation step
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            
            # Compute validation loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Predictions for accuracy, precision, recall, and F1-score
            predicted = (outputs > 0.5).float()
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss / len(test_loader))
    val_accuracies.append(val_accuracy)

    # Calculate precision, recall, F1-score, and confusion matrix
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    print(f'Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracy:.2f}%')
    print(f'Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
    # Confusion matrix visualization
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-COVID', 'COVID'], yticklabels=['Non-COVID', 'COVID'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    # Step the learning rate scheduler
    scheduler.step()

print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'covid_classification_hybrid.pth')

# Plot the loss and accuracy graph
plt.figure(figsize=(12, 5))

# Loss graph
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()

# Accuracy graph
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epochs')
plt.legend()

plt.tight_layout()
plt.show()

#########EVALUATION######################


# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
        outputs = model(images)
        predicted = (outputs > 0.5).float()  # Sigmoid output > 0.5 is class 1 (COVID)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total}%')