import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# 直接测试实战的

def find_circle(path):
    mask = cv2.imread(path)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_gray[mask_gray < 50] = 0
    mask_gray[mask_gray >= 50] = 255
    # 识别轮廓
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    # 转换成数组
    np.empty(mask_gray.shape, dtype=np.float32)
    plt.imshow(result)
    plt.show()

    for contour in contours:
        # 计算到轮廓的距离
        perimeter = cv2.arcLength(contour, True)
        print("周长是", perimeter)
    # cv2.imshow('Maximum inscribed circle', result)
    # cv2.waitKey(0)
    plt.imshow(result)
    plt.show()


if __name__ == '__main__':
    start = time.perf_counter()
    find_circle('../pic/1.jpg')
    end = time.perf_counter()
    print("运行耗时", end - start)
