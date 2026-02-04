import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_mnist_data(batch_size=32):
    # 使用数据增强和预处理：转换为 Tensor，进行标准化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为 Tensor
        transforms.Normalize((0.5,), (0.5,))  # 单通道标准化：均值0.5，标准差0.5
    ])

    # 加载训练数据集
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 加载测试数据集
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader
