from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.predict import make_predictions
import config

def main():
    # 训练模型
    print("Training model...")
    train_model(config.config)  # 使用 config 配置开始训练

    # 评估模型
    print("\nEvaluating model...")
    evaluate_model(config.config)  # 评估模型

    # 进行预测
    print("\nMaking predictions...")
    make_predictions(config.config)  # 使用训练好的模型进行预测


if __name__ == "__main__":
    main()
