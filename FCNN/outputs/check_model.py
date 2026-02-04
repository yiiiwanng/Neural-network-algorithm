import torch
import os


def check_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: The file '{model_path}' does not exist.")
        return

    try:
        # 尝试加载模型
        model = torch.load(model_path)
        print("Model loaded successfully.")

        # 检查模型的权重（state_dict）
        print("\nModel's state_dict (weights):")
        for param_tensor in model.items():
            print(f"{param_tensor[0]}: {param_tensor[1].size()}")

    except Exception as e:
        print(f"Error: Failed to load the model. {str(e)}")


if __name__ == "__main__":
    # 设置 model.pth 文件的路径
    model_path = '../outputs/model.pth'

    # 调用函数检查模型
    check_model(model_path)
