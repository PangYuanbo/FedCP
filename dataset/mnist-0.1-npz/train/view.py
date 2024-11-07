import numpy as np
import matplotlib.pyplot as plt

# 加载.npz文件，设置allow_pickle=True
data = np.load('train0_.npz', allow_pickle=True)

# 打印图像数据的类型和形状
images = data['data']
print("Type of images:", type(images))
print("Shape of images:", np.shape(images))
print("Content of images:", images)

# 假设图像数据存储在'data'键中，并且包含'x'和'y'键
x_data = images['x']
y_data = images['y']

# 可视化前20个图像
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_data[i][0], cmap='gray')
    plt.axis('off')
    plt.title(f'Label: {y_data[i]}')

plt.tight_layout()
plt.show()