import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.optim import Adam

from tqdm import tqdm

import matplotlib.pyplot as plt

def get_dataset(digits=[5, 8], n_px=16, train_size=1000, test_size=200):
    # Load raw data (uint8 [0-255])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True)

    # Helper: filter, sample, normalize, resize
    def prepare(data, targets, digits, size, n_px):
        # Filter by class
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for d in digits:
            mask |= (targets == d)
        data_f = data[mask]
        targ_f = targets[mask]

        # Sample 'size' examples
        idx = torch.randperm(len(data_f))[:size]
        imgs = data_f[idx].unsqueeze(1).float() / 255.0  # [samples,1,28,28]
        labs = targ_f[idx]

        # Resize to n_px x n_px
        imgs_resized = F.interpolate(imgs, size=(n_px, n_px), mode='bilinear', align_corners=False)
        # Map labels to 0..len(digits)-1
        labs_mapped = torch.tensor([digits.index(int(l)) for l in labs])
        return imgs_resized, labs_mapped

    x_train, y_train = prepare(mnist_train.data, mnist_train.targets, digits, train_size, n_px)
    x_test, y_test = prepare(mnist_test.data, mnist_test.targets, digits, test_size, n_px)
    
    return (x_train, y_train), (x_test, y_test)

def visualise_data(x, y, pred=None):
    n_img = len(x)
    fig, axes = plt.subplots(1, n_img, figsize=(2*n_img, 2))
    for i in range(n_img):
        axes[i].imshow(x[i], cmap="gray")
        if pred is None:
            axes[i].set_title("Label: {}".format(y[i]))
        else:
            axes[i].set_title("Label: {}, Pred: {}".format(y[i], pred[i]))
    plt.tight_layout(w_pad=2)

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()  # Set the model to training mode
    total_loss, correct, total = 0.0, 0, 0

    loop = tqdm(dataloader, desc="Train", leave=False)
    for X, y in loop:
        X, y = X.to(device), y.to(device)
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item() * X.size(0)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train_and_plot_accuracy(model, train_loader, test_loader, epochs, lr, device):
    """
    Train the model and plot accuracy vs epochs for both training and testing datasets.
    
    Args:
    - model: PyTorch model to be trained.
    - train_loader: DataLoader for the training data.
    - test_loader: DataLoader for the test data.
    - epochs: Number of epochs to train the model.
    - lr: Learning rate for the optimizer.
    - device: Device to run the model on (CPU or CPU).
    
    Returns:
    - None (plots the accuracy graph).
    """

    # Initialize loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # Using CrossEntropyLoss by default
    optimizer = Adam(model.parameters(), lr=lr)  # Using Adam optimizer with provided learning rate

    # Initialize ReduceLROnPlateau scheduler with fixed parameters
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Lists to store accuracies
    train_accuracies = []
    test_accuracies = []

    # Start time tracking
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        # Training phase
        train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer, device)
        
        # Testing phase
        test_loss, test_acc = test_loop(test_loader, model, loss_fn, device)

        # Store the accuracies
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # Print epoch details (optional)
        # current_lr = optimizer.param_groups[0]['lr']
        current_lr = scheduler.get_last_lr()[0]
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | "
                  f"Learning Rate: {current_lr:.6f}")

        # Step the scheduler based on the validation loss (test_loss in this case)
        scheduler.step(test_loss)

    # End time tracking
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")

    # Determine dynamic lower y-limit
    min_acc = min(min(train_accuracies), min(test_accuracies))
    lower_limit = max(0, min_acc - 0.05)  # Optional margin below the lowest point

    # Plotting Accuracy vs Epochs
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', color='blue', marker='o')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='orange', marker='x')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(type(model).__name__)
    plt.ylim(lower_limit, 1)  # Dynamic lower, fixed upper
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Evaluate model on adversarial examples
def evaluate_under_attack(model, attack_model, base_feats, labels, epsilons, alpha, num_iter):
    acc_list = []

    for eps in epsilons:
        perturb = PGD(model, base_feats, labels, epsilon=eps, alpha=alpha, num_iter=num_iter)
        x_adv = (base_feats + perturb).clamp(0, 1)  # Clamp to valid pixel range
        adv_loader = DataLoader(TensorDataset(x_adv, labels), batch_size=batch_size)
        _, acc = test_loop(adv_loader, model, loss_fn, device)
        acc_list.append(acc)

    return acc_list
