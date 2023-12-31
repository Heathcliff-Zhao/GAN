from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader():
    # 数据加载和预处理
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True, drop_last=True)
    return train_loader
