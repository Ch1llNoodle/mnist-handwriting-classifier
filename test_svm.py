import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# 设置matplotlib的配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 全局参数
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
                    images.append(np.array(img).flatten())
                    labels.append(int(folder_name))
    return (np.array(images), np.array(labels)) if images and labels else (None, None)

# 加载模型和归一化器
svm_model = joblib.load('svm_mnist_model.pkl')
scaler = joblib.load('scaler.pkl')

# 加载测试数据
test_base_path = r"C:\Users\lfq\Desktop\来都来了\模式识别原理及应用\手写数字"
x_test, y_test = load_custom_data(test_base_path)

if x_test is None or y_test is None:
    print("未能加载测试集，请检查路径或文件格式。")
    exit()

# 数据预处理：归一化
x_test = scaler.transform(x_test.astype('float32') / 255.0)

# 模型预测
y_pred = svm_model.predict(x_test)

# 评估结果
accuracy = accuracy_score(y_test, y_pred)
print(f"整体识别准确率: {accuracy * 100:.2f}%")
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.title("混淆矩阵")
plt.xlabel("预测值")
plt.ylabel("真实值")
plt.show()