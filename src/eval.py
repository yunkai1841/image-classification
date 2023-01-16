"""
Evaluate the trained model on the test set.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.models import AlexNet
from torchvision import transforms
from torchvision.datasets import ImageFolder

from img_resize import to_tensor, crop_image, to_square, resize_image
from training import test, preprocess


def main():
    # Parameters
    net = AlexNet(num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    net.load_state_dict(torch.load('output/model.pt', map_location=device))
    net.to(device)

    test_data = ImageFolder('test_data', transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    print(f"Test set size: {len(test_data)}")

    test(net, test_loader, criterion, device)


if __name__ == '__main__':
    main()