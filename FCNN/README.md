#FCNN Project

## 项目介绍
这是一个全连接神经网络（FCNN）的实现示例，使用 PyTorch 框架和MNIST公开数据集，进行训练和评估。

## 如何使用
1. 克隆项目：
   ```bash
   git clone https://github.com/yiiiwanng/FCNN.git
   cd fcnn-project
   ```
2.创建并激活虚拟环境：
```bash
pip -m venv venv 

# Windows
.venv/Scripts/activate
# Linux/macOS
source .venv/bin/activate
```
3.安装依赖：
```bash
pip install -r requirements.txt
```
3.训练模型：
```bash
python scripts/train.py
```
4.评估模型：
```bash
python scripts/evaluate.py
```
5.进行预测：
```bash
python scripts/predict.py
```