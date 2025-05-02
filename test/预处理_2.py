import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取红外图像（假设是灰度图）
img_path = 'C:/Users/liushihao/Desktop/IR_DRONE_120_271_141.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 应用直方图均衡化
equalized = cv2.equalizeHist(img)

# 显示原图和增强后的图像
plt.subplot(1, 2, 1)
plt.title('原始图像')
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('直方图均衡化')
plt.imshow(equalized, cmap='gray')

plt.show()
