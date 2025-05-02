import torch
import torch.nn as nn
import torch.nn.functional as F

import pennylane as qml
from pennylane import numpy as np

# convolutional neural network for good performance
# PyTorch tutorials: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class NetCNN(nn.Module):
    def __init__(self, n_px, n_classes):
        super().__init__()
        # out_size = [w-k+2p]/s+1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * n_px//4 * n_px//4, 16)
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# fully connected neural network for direct comparison with quantum model
# PyTorch tutorials: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class NeuralNetwork(nn.Module):
    def __init__(self, n_px, n_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_px*n_px, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, n_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class QNN(nn.Module):
    def __init__(self, n_qubits=6, n_classes=4, n_layers=8):
        super(QNN, self).__init__()
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Quantum circuit using AmplitudeEmbedding and StronglyEntanglingLayers
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        weight_shapes = {"weights": (n_layers, self.n_qubits, 3)}
        qnode = qml.QNode(circuit, self.dev, interface="torch", diff_method="backprop")
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.fc = nn.Linear(self.n_qubits, self.n_classes)

    def forward(self, x):
        # Flatten from (batch_size, 1, n_px, n_px) to (batch_size, n_px*n_px)
        x = x.view(x.size(0), -1)  # Flatten to shape (B, 64)
        
        # Normalize input to unit vector for amplitude embedding
        x = F.normalize(x, p=2, dim=1)
        x = self.qlayer(x)  # Pass through quantum layer
        x = self.fc(x)      # Final linear layer for logits
        return x

def PGD(model, feats, labels, epsilon=0.1, alpha=None, num_iter=10):
    if alpha is None:
        alpha = epsilon/num_iter
    
    model.eval()  # Ensure model is in eval mode
    feats = feats.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    # Start with zero perturbation
    delta = torch.zeros_like(feats, requires_grad=True).to(device)
    
    for t in range(num_iter):
        outputs = model(feats + delta)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        # Update delta and clamp it to epsilon
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    
    return delta.detach()








        