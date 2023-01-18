import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet152
from torchvision import transforms
from torchvision.datasets import ImageFolder


def print_wrapper(*args):
    print(*args, flush=True)


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print_wrapper(loss)


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
    print_wrapper(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)")


def main():
    # Parameters
    print_wrapper("Starting...")
    net = resnet152(weights=None, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        net = nn.DataParallel(net, device_ids=list(
            range(torch.cuda.device_count())))
        torch.backends.cudnn.benchmark = True

    print_wrapper(f"Using device: {device}")

    preprocess = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    training_data = ImageFolder('data', transform=preprocess)
    print_wrapper(training_data[0][0].shape)
    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)

    test_data = ImageFolder('test_data', transform=preprocess)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    print_wrapper(f"Training data: {len(training_data)}")
    print_wrapper(f"Test data: {len(test_data)}")

    net.to(device)

    # Train the model
    for epoch in range(1, 10):
        train(net, train_loader, criterion, optimizer, device)
        test(net, test_loader, criterion, device)
        print_wrapper(f"Epoch {epoch} done")
    print_wrapper("Training done")

    # Save the model
    torch.save(net.state_dict(), 'output/model.pt')
    print_wrapper("Model saved")


if __name__ == '__main__':
    main()
