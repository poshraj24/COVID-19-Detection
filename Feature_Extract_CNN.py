from Extract_Transform import *
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout25 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 14 * 14, 64)  # Adjust input size based on final output from conv layers
        self.dropout50 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Conv Layer 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout25(x)
        
        # Conv Layer 2
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout25(x)
        
        # Conv Layer 3
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout25(x)
        
        # Conv Layer 4
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = self.dropout25(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout50(x)
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification output
        
        return x
    
# # #Initialize the model
model=CNN()

# # #Define the loss function and optimizer
criterion=nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)