import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torch.nn.functional as F

# 设置matplotlib的配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取MNIST数据集的函数
def read_images(filename):
    # 读取图像文件的代码
    with open(filename, 'rb') as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, 1, num_rows, num_cols)  # 保持 (num_images, 1, 28, 28)
        return images

def read_labels(filename):
    # 读取标签文件的代码
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# 加载和预处理MNIST数据集
train_images_path = "C:\\Users\\lfq\\PycharmProjects\\MNIST\\train-images.idx3-ubyte"
train_labels_path = "C:\\Users\\lfq\\PycharmProjects\\MNIST\\train-labels.idx1-ubyte"

x_train = read_images(train_images_path)
y_train = read_labels(train_labels_path)

# 数据预处理：归一化
x_train_normalize = x_train.astype('float32') / 255.0

# 转换为PyTorch张量
x_train_tensor = torch.tensor(x_train_normalize, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# 数据增强
transform = transforms.Compose([
    transforms.RandomRotation(15),  # 随机旋转角度
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=10),  # 仿射变换
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.GaussianBlur(3),  # 高斯模糊
    transforms.RandomErasing(),  # 随机擦除部分像素
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 创建PyTorch DataLoader
train_data = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=200, shuffle=True)

# 定义卷积神经网络（CNN）模型
class EnhancedCNNModel(nn.Module):
    def __init__(self):
        # 初始化模型层
        super(EnhancedCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 新增卷积层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # 修改全连接层大小
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # 前向传播
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化CNN模型并转移到GPU
model = EnhancedCNNModel().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 使用较小的学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)  # 动态调整学习率

# 早停法类
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        # 初始化早停法参数
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        # 早停法逻辑
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss >= self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# 训练模型
epochs = 50
train_loss = []
train_accuracy = []
early_stopping = EarlyStopping(patience=5, delta=0.001)

for epoch in range(epochs):
    # 训练过程
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # 将数据转移到GPU
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = 100 * correct / total
    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)

    scheduler.step(epoch_loss)  # 动态调整学习率

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    early_stopping(epoch_loss)
    if early_stopping.early_stop:
        print("早停法触发，停止训练")
        break

# 可视化训练过程中的准确率和损失
plt.plot(range(len(train_accuracy)), train_accuracy, label='训练准确率')
# 绘制准确率图
plt.title('训练准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()
plt.show()

plt.plot(range(len(train_loss)), train_loss, label='训练损失')
# 绘制损失图
plt.title('训练损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()
plt.show()

# 保存模型
torch.save(model.state_dict(), 'mnist_cnn_model.pth')