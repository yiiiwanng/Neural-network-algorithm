config = {
    'train_data_path': './data',  # MNIST 数据集的保存路径
    'batch_size': 32,             # 批次大小
    'epochs': 10,                 # 训练轮数
    'learning_rate': 0.001,
    'hidden_size': 128,           # 隐藏层大小，可以调整
    'input_size': 784,            # MNIST 每张图片是 28x28 (784 个特征)
    'output_size': 10,            # MNIST 有 10 个数字类别
    'model_save_path': './outputs/model.pth',
    'prediction_output_path': './outputs/predictions.csv'
}