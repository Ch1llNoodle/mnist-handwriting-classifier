import numpy as np
import struct
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # 用于保存和加载模型
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # 导入 tqdm 库

# 设置matplotlib的配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取MNIST数据集的函数
def read_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num_images, num_rows * num_cols)  # 展平成向量

def read_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# 加载MNIST数据集
train_images_path = "D:\\GitHub\\mnist-handwriting-classifier\\train-images.idx3-ubyte"
train_labels_path = "D:\\GitHub\\mnist-handwriting-classifier\\train-labels.idx1-ubyte"

x_train = read_images(train_images_path)
y_train = read_labels(train_labels_path)

# 数据预处理：归一化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.astype('float32') / 255.0)

# 训练 SVM 模型
print("开始训练 SVM...")
svm_model = SVC(kernel='rbf', C=10, gamma=0.05)  # 使用 RBF 核函数

# 使用 tqdm 显示进度条
for _ in tqdm(range(1), desc="训练 SVM"):
    svm_model.fit(x_train, y_train)

print("SVM 训练完成！")

# 保存模型和归一化器
joblib.dump(svm_model, 'svm_mnist_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("SVM 模型已保存。")