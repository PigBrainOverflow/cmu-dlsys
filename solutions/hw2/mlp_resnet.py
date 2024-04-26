import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    seq = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    return nn.Sequential(nn.Residual(seq), nn.ReLU())
    ### END YOUR SOLUTION


# def MLPResNet(
#     dim,
#     hidden_dim=100,
#     num_blocks=3,
#     num_classes=10,
#     norm=nn.BatchNorm1d,
#     drop_prob=0.1,
# ):
#     ### BEGIN YOUR SOLUTION
#     resblocks = tuple(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks))
#     seq = nn.Sequential(
#         nn.Linear(dim, hidden_dim),
#         nn.ReLU(),
#         *resblocks,
#         nn.Linear(hidden_dim, num_classes)
#     )
#     print(seq)
#     return seq
#     ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    layers = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    ]
    for _ in range(num_blocks):
        layers.append(
            ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob)
        )
    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)
    ### END YOUR SOLUTION

def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # Set the model to training or evaluation mode
    if opt:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_error = 0
    total_samples = 0

    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = nn.SoftmaxLoss(outputs, targets)  # Define loss_function according to your needs

        # Backward pass and optimization
        if opt:
            opt.reset_grad()  # Reset gradients
            loss.backward()  # Compute gradients
            opt.step()       # Update parameters

        # Accumulate metrics
        total_loss += loss.item()
        total_error += (outputs.argmax(dim=1) != targets).sum().item()  # Assuming classification task
        total_samples += inputs.size(0)

    # Calculate average loss and error rate
    avg_loss = total_loss / total_samples
    avg_error_rate = total_error / total_samples

    return avg_error_rate, avg_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz",
        f"{data_dir}/t10k-labels-idx1-ubyte.gz",
    )
    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
        test_err, test_loss = epoch(test_dataloader, model)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
