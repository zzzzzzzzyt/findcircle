import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


#  通过像素来求周长的

def find_circle(path):
    mask = cv2.imread(path)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_gray[mask_gray < 50] = 0
    mask_gray[mask_gray >= 50] = 255
    # 进行反色处理
    mask_gray = 255 - mask_gray
    # 识别轮廓
    # CV_CHAIN_APPROX_NONE是把轮廓上所有的点都保存下来，但其实我们并不需要，我们只用保存拐点的即可
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask, contours, -1, (0, 255, 255), 1)

    # 转换成数组
    np.empty(mask_gray.shape, dtype=np.float32)
    plt.imshow(mask)
    plt.show()

    for contour in contours:
        # 计算到轮廓的距离
        print("周长是", len(contour))
        print("半径是", len(contour)/(2*math.pi))
    # cv2.imshow('Maximum inscribed circle', result)
    # cv2.waitKey(0)
    plt.imshow(result)
    plt.show()


if __name__ == '__main__':
    start = time.perf_counter()
    find_circle('../test/test4.png')
    end = time.perf_counter()
    print("运行耗时", end - start)
