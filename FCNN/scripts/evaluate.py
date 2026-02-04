import torch
import torch.nn as nn
from models.fcnn import SimpleFCNN
from scripts.load_data import load_mnist_data

def evaluate_model(config):
    # 加载模型
    model = SimpleFCNN(config['input_size'], config['output_size'], config['hidden_size'])
    model.load_state_dict(torch.load(config['model_save_path']))
    model.eval()  # 设置为评估模式

    # 加载 MNIST 测试数据
    _, testloader = load_mnist_data(batch_size=config['batch_size'])

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.view(-1, 28*28)  # 展平图像数据
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the 10000 test images: {100 * correct / total}%')
