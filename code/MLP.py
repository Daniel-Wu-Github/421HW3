import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size=3, hidden_size=3, output_size=3):
        super(MLP, self).__init__()
        ### YOUR CODE HERE
        # Define the first linear layer: input features -> hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Define the second linear layer: hidden layer -> output features
        self.fc2 = nn.Linear(hidden_size, output_size)
        ### END YOUR CODE

    def forward(self, x):
        ### YOUR CODE HERE
        # Pass input through the first linear layer
        x = self.fc1(x)
        # Apply the ReLU activation function
        x = F.relu(x)
        # Pass the result through the second linear layer to get the final output
        x = self.fc2(x)
        ### END YOUR CODE
        return x
