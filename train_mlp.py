import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


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

# 数据预处理
x_train_normalize = x_train.astype('float32') / 255.0
y_train_onehot = np.eye(10)[y_train]

x_train_tensor = torch.tensor(x_train_normalize, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_onehot, dtype=torch.float32)

train_data = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=200, shuffle=True)

# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平张量为 (batch_size, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型并迁移到 GPU
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
train_loss = []
train_accuracy = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 数据迁移到 GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == torch.max(labels, 1)[1]).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = 100 * correct / total
    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

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
torch.save(model.state_dict(), 'mnist_mlp_model.pth')