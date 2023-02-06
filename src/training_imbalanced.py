# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import AlexNet
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchsampler import ImbalancedDatasetSampler

from torch.utils.tensorboard import SummaryWriter


def print_wrapper(*args):
    print(*args, flush=True)


# %%
def train_with_summary(model, train_loader, criterion, optimizer, device, writer, test_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print_wrapper(loss)
        writer.add_scalar('training loss', loss.item(), batch_idx)
        if batch_idx % 200 == 0:
            print_wrapper("test for {}".format(batch_idx))
            test_with_summary(model, test_loader, criterion,
                              device, writer, batch_idx)


# %%
def test_with_summary(model, test_loader, criterion, device, writer, batch_idx):
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
    writer.add_scalar('accuracy', 100. * correct /
                      len(test_loader.dataset), batch_idx)


# %%
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


# %%
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


# %% Parameters
net = AlexNet(num_classes=2)
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_wrapper(f"Using device: {device}")

# Training on multiple GPUs
# if torch.cuda.device_count() > 1:
#     print_wrapper("Using {} GPUs".format(torch.cuda.device_count()))
#     net = nn.DataParallel(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# %%
training_data = ImageFolder('images', transform=preprocess)
print_wrapper(training_data[0][0].shape)
train_loader = DataLoader(
    training_data, sampler=ImbalancedDatasetSampler(training_data), batch_size=64)

test_data = ImageFolder('test_data', transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64)

print_wrapper(f"Training data: {len(training_data)}")
print_wrapper(f"Test data: {len(test_data)}")

net.to(device)

# %% Train the model
writer = SummaryWriter()
try:
    train_with_summary(net, train_loader, criterion,
                       optimizer, device, writer, test_loader)
except KeyboardInterrupt:
    print_wrapper("Training interrupted")
    pass
except Exception as e:
    print_wrapper("Error occured: {}".format(e))
    pass
writer.close()
# %%
# test_data2 = ImageFolder('test_data', transform=transforms.ToTensor())
# test_loader2 = DataLoader(test_data2, batch_size=64)
test(net, test_loader, criterion, device)
print_wrapper("Training done")

# %% Save the model
# net = net.module if hasattr(net, 'module') else net
torch.save(net.state_dict(), 'model.pth')
print_wrapper("Model saved")
