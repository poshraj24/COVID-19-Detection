from Extract_Transform import *
from Feature_Extract_CNN import *

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Training the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the appropriate device (GPU if available)
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        outputs= torch.squeeze(outputs)

        # Check if batch sizes match
        if outputs.size(0) != labels.size(0):
            print(f"Batch size mismatch: outputs {outputs.size(0)}, labels {labels.size(0)}")
            continue  
        
        loss = criterion(outputs, labels.float())    # BCELoss expects float labels

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'covid_classification.pth')

# Evaluate the model
model.eval()
correct=0
total=0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total}%')