import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# 这个的话 基本上是通过统计像素

# 直接测试实战的

def find_circle(mask):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_gray[mask_gray < 50] = 0
    mask_gray[mask_gray >= 50] = 255
    # 进行反色处理  因为我找到的里面是要白的
    mask_gray = 255 - mask_gray
    # 识别轮廓   只检测外轮廓   然后获取轮廓总面积 然后对轮廓内的黑色部分进行计算
    result = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    # 转换成数组
    np.empty(mask_gray.shape, dtype=np.float32)
    # plt.imshow(result)
    # plt.show()

    now_gray = 255 - mask_gray
    # 把线的面积找出来
    # 把整个轮廓的面积找出来
    pixels = cv2.countNonZero(now_gray)
    return pixels


if __name__ == '__main__':
    start = time.perf_counter()
    mask = cv2.imread('../test/circle.png')
    whole_area = find_circle(mask)
    end = time.perf_counter()
    print("运行耗时", end - start)
    print("轮廓内面积为", whole_area)
