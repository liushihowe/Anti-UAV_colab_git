import cv2
import numpy as np

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    """
    Resize and pad an image while preserving the aspect ratio to fit a target size (new_shape).
    The target size is typically a square, such as (640, 640).
    """
    shape = img.shape[:2]  # current shape [height, width]
    
    # 计算缩放比例
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # compute the resize ratio
    if not scaleup:
        ratio = min(ratio, 1.0)  # 不允许放大

    # 计算新的尺寸
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))  # (width, height)

    # 填充的宽度和高度
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # padding widths
    dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # padding to multiples of stride

    # 缩放图像
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 在四周填充，填充的颜色是灰色
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img_padded

def expand_to_rgb(img):
    """
    扩展灰度图（单通道）为三通道（RGB）。
    """
    if len(img.shape) == 2:  # 如果是灰度图
        img = np.stack([img] * 3, axis=-1)  # 将单通道图像复制到三通道
    return img

# 示例使用
img_path = 'C:/Users/liushihao/Desktop/IR_DRONE_120_271_141.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像

# 调用 letterbox 函数，调整图像大小并填充
img_resized = letterbox(img, new_shape=(640, 640), color=(114, 114, 114))

# 将单通道灰度图扩展为三通道
img_rgb = expand_to_rgb(img_resized)

# 打印图像形状
print('Resized and Padded Image Shape:', img_rgb.shape)

# 显示图像
cv2.imshow('Resized and Padded Image', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图像
cv2.imwrite('resized_padded_image.jpg', img_rgb)
print('Resized and Padded Image Shape:', img_rgb.shape)

