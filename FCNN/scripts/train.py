import torch
import torch.nn as nn
import torch.optim as optim
from models.fcnn import SimpleFCNN  # 导入 FCNN 模型
from scripts.load_data import load_mnist_data

def train_model(config):
    # 加载 MNIST 数据
    trainloader, testloader = load_mnist_data(batch_size=config['batch_size'])

    # 初始化 FCNN 模型
    model = SimpleFCNN(config['input_size'], config['output_size'], config['hidden_size'])

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    # 训练模型
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.view(-1, 28*28)  # 展平图像数据

            optimizer.zero_grad()  # 清空之前的梯度

            # 正向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()
            if i % 200 == 199:  # 每200个batch输出一次损失
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}")
                running_loss = 0.0

    print("Finished Training")
    torch.save(model.state_dict(), config['model_save_path'])
