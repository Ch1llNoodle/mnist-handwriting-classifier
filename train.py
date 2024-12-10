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
train_images_path = "D:\\GitHub\\mnist-handwriting-classifier\\train-images.idx3-ubyte"
train_labels_path = "D:\\GitHub\\mnist-handwriting-classifier\\train-labels.idx1-ubyte"

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
    transforms.GaussianBlur(3),  # 高斯模糊
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
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率衰减

# 早停法类
class EarlyStopping:
    def __init__(self, patience=5, delta=0, smoothing=False):
        self.patience = patience
        self.delta = delta
        self.smoothing = smoothing
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.avg_loss = None  # 滑动平均值（仅在启用平滑时使用）

    def __call__(self, val_loss):
        # 平滑损失计算
        if self.smoothing:
            self.avg_loss = (
                0.8 * self.avg_loss + 0.2 * val_loss if self.avg_loss is not None else val_loss
            )
            current_loss = self.avg_loss
        else:
            current_loss = val_loss

        # 判断是否有改进
        if current_loss < self.best_loss - self.delta:
            print(f"验证损失从 {self.best_loss:.4f} 降低到 {current_loss:.4f}，计数器重置。")
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"验证损失未改进。计数器：{self.counter}/{self.patience}")

            # 检查是否需要触发早停
            if self.counter >= self.patience:
                self.early_stop = True

# 训练模型
epochs = 50
train_loss = []
train_accuracy = []
# 初始化早停逻辑
early_stopping = EarlyStopping(patience=5, delta=0.001, smoothing=True)

for epoch in range(epochs):
    model.train()
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

    scheduler.step()  # 仅更新学习率
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # 调用早停逻辑
    early_stopping(epoch_loss)
    if early_stopping.early_stop:
        print(f"触发早停，在第 {epoch + 1} 轮停止训练。")
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