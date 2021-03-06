import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from customDataset import CatsAndDogsDataset

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 0.0001
batch_size = 32
num_epochs = 1

# Load Data
dataset = CatsAndDogsDataset(
    csv_file="cats_dogs.csv", root_dir='cats_dogs_resized', transform=transforms.ToTensor())


train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])

train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_set,
                         batch_size=batch_size, shuffle=True)


# Load pretained model and modify it
model = torchvision.models.googlenet(pretrained=True)
model.to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# if load_model:
#     load_checkpoint(torch.load("my_checkpoint.pth.tar"))

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    mean_loss = sum(losses) / len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:.5f}')
# Check accuracy on training & test to see how good our model is


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
