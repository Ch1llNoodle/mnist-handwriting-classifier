import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2
import numpy as np

# 设置matplotlib的配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局参数
PADDING = 4
TARGET_SIZE = (28, 28)

# 加载自定义测试集
def load_custom_data(test_base_path):
    images, labels = [], []
    for folder_name in os.listdir(test_base_path):
        folder_path = os.path.join(test_base_path, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.bmp'):
                    file_path = os.path.join(folder_path, file_name)
                    img = Image.open(file_path).convert('L').resize(TARGET_SIZE)
                    images.append(np.array(img))
                    labels.append(int(folder_name))
    return (np.array(images), np.array(labels)) if images and labels else (None, None)

# 数字图像处理：裁剪、调整大小、对一化
def preprocess_custom_data(x_custom, device):
    preprocessed_images = []
    kernel = np.ones((2, 2), np.uint8)  # 用于膏肥操作

    for img in x_custom:
        img = 255 - img  # 反转颜色（白底变黑底）
        thickened_img = cv2.dilate(img, kernel, iterations=1)
        # 检测边界并裁剪
        contours, _ = cv2.findContours(thickened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            side_length = max(w, h) + 2 * PADDING
            x_center, y_center = x + w // 2, y + h // 2
            x_start, y_start = max(0, x_center - side_length // 2), max(0, y_center - side_length // 2)
            x_end, y_end = x_start + side_length, y_start + side_length
            cropped_img = thickened_img[y_start:y_end, x_start:x_end]
            resized_img = cv2.resize(cropped_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            normalized_img = resized_img.astype('float32') / 255.0
            preprocessed_images.append(normalized_img)

    if not preprocessed_images:
        return torch.empty(0, 1, *TARGET_SIZE, device=device)
    return torch.tensor(np.array(preprocessed_images).reshape(len(preprocessed_images), 1, *TARGET_SIZE), device=device)

# 定义多层感知器（MLP）模型
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)  # 输出直接是 10 类

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 最后一层没有激活函数，因为后续会计算 cross-entropy loss
        return x

# 加载模型
def load_model(model_path):
    model = MLPModel().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()  # 设置为评价模式
    return model

# 计算每个数字的识别准确率并返回正确预测的统计数据
def calculate_accuracy(model, x_test, y_test):
    correct_per_digit = {i: {'correct': 0, 'total': 0} for i in range(10)}
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(y_test)):
            digit = y_test[i].item()
            correct_per_digit[digit]['total'] += 1
            if predicted[i] == digit:
                correct_per_digit[digit]['correct'] += 1

    # 返回每个数字的正确和总数以及每个数字的识别准确率
    accuracy_per_digit_dict = {
        digit: (correct_per_digit[digit]['correct'] / correct_per_digit[digit]['total']) * 100
        if correct_per_digit[digit]['total'] > 0 else 0
        for digit in range(10)
    }

    return correct_per_digit, accuracy_per_digit_dict

test_base_path = r"C:\Users\lfq\Desktop\来都来了\模式识别原理及应用\手写数字"
x_custom, y_custom = load_custom_data(test_base_path)

if x_custom is None or y_custom is None:
    print("未能加载测试集，请检查路径或文件格式。")
    exit()

# 打印测试集的长度
print(f"测试集数据长度: {len(x_custom)}")
print(f"测试集标签长度: {len(y_custom)}")

# 预处理自定义测试集数据
x_custom_tensor = preprocess_custom_data(x_custom, device)
y_custom_tensor = torch.tensor(y_custom, dtype=torch.long, device=device)

# 加载模型
model_path = 'mnist_mlp_model.pth'
model = load_model(model_path)

# 调用函数并获取结果
correct_per_digit, accuracy_per_digit_dict = calculate_accuracy(model, x_custom_tensor, y_custom_tensor)

# 将模型输出与标签对比
y_pred = []
y_true = []
with torch.no_grad():
    outputs = model(x_custom_tensor)
    _, predicted = torch.max(outputs, 1)
    y_pred = predicted.cpu().numpy()
    y_true = y_custom_tensor.cpu().numpy()

# 计算并展示混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.title("混淆矩阵")
plt.xlabel("预测值")
plt.ylabel("真实值")
plt.show()

# 获取模型预测结果
model.eval()  # 确保模型处于评估模式
with torch.no_grad():
    outputs = model(x_custom_tensor)
    _, predicted = torch.max(outputs, 1)

# 记录每个标签第一次识别错误的图像
error_images = {}
error_labels = {}  # 记录真实标签和预测标签

for i in range(len(y_custom)):
    true_label = y_custom_tensor[i].item()
    pred_label = predicted[i].item()

    # 如果识别错误，且这个数字的错误样本还没有被记录
    if true_label != pred_label and true_label not in error_images:
        error_images[true_label] = x_custom_tensor[i].cpu().numpy().squeeze()  # 提取图像
        error_labels[true_label] = pred_label  # 记录预测标签

# 显示每个标签第一张识别错误的图像
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle('每个标签第一张识别错误的图像', fontsize=16)

for i in range(10):  # 遍历 0 到 9 的数字标签
    ax = axes[i // 5, i % 5]

    if i in error_images:
        ax.imshow(error_images[i], cmap='gray')
        ax.set_title(f'真实值: {i}, 预测值: {error_labels[i]}')
    else:
        ax.text(0.5, 0.5, 'No Error', fontsize=12, ha='center', va='center')

    ax.axis('off')  # 隐藏坐标轴

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

# 输出每个数字的识别正确率
for digit, accuracy in accuracy_per_digit_dict.items():
    print(f"数字 {digit} 的识别正确率: {accuracy:.2f}%")

# 计算手写数字的整体识别精度
total_correct = sum(correct_per_digit[digit]['correct'] for digit in range(10))
total_images = sum(correct_per_digit[digit]['total'] for digit in range(10))
overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0

print(f"手写数字的整体识别精度: {overall_accuracy:.2f}%")