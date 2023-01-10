import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import AlexNet
from torchvision import transforms
from torchvision.datasets import ImageFolder

from img_resize import to_tensor, crop_image, to_square, resize_image


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)")


def main():
    net = AlexNet(num_classes=2)
    criterion = nn.CrossEntropyLoss()

    preprocess = lambda img: to_tensor(
        crop_image(
            to_square(
                resize_image(img, size=(256, 256))
            ),
            size=(224, 224)
        )
    )

    training_data = ImageFolder('images', transform=preprocess)
    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)

    test_data = ImageFolder('images', transform=preprocess)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    for X, y in test_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

if __name__ == '__main__':
    main()
