from Training_Evaluation import *
##Transfer learning implementation
# Initialize the dataset
dataset = CovidDataset(COVID_FOLDER, NON_COVID_FOLDER, transform=transform)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Initialize the dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

# Load Pretrained ResNet18 Model
model = models.resnet18(pretrained=True)

# Freeze the parameters of ResNet to avoid backpropagation through them
for param in model.parameters():
    param.requires_grad = False

# Modify the fully connected layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # Binary classification, output 1 unit

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Only update the final layer

# Training the model
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the appropriate device
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # Reshape labels

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = torch.sigmoid(model(images))  # Sigmoid for binary output

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:    # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'covid_classification_transfer_learning.pth')

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

print(f'Accuracy of the network on the test images: {100 * correct / total}%') #Acc=82.29166666666667%