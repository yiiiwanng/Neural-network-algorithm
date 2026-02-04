# scripts/predict.py
import torch
from models.fcnn import SimpleFCNN
from scripts.load_data import load_mnist_data
import pandas as pd


def make_predictions(config):
    _, testloader = load_mnist_data(batch_size=config['batch_size'])

    # 加载模型
    model = SimpleFCNN(config['input_size'], config['output_size'], config['hidden_size'])
    model.load_state_dict(torch.load(config['model_save_path']))
    model.eval()

    predictions = []
    with torch.no_grad():
        for data in testloader:
            inputs, _ = data
            inputs = inputs.view(-1, 28 * 28)  # 展平图像
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())

    # 保存预测结果
    df = pd.DataFrame(predictions, columns=['Predicted'])
    df.to_csv(config['prediction_output_path'], index=False)
    print(f'Predictions saved to {config["prediction_output_path"]}')
