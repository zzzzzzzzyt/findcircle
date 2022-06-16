import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# 进一步处理 先进行孔洞的膨胀  再进行腐蚀去掉一部分干扰的小孔

def find_circle(path):
    mask = cv2.imread(path)

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_gray[mask_gray < 50] = 0
    mask_gray[mask_gray >= 50] = 255
    plt.imshow(mask_gray)
    plt.show()

    # # 形态学内核 参数作用
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    # # 形态学闭运算 参数作用
    # mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel)
    # plt.imshow(mask_gray)
    # plt.show()

    # 识别轮廓
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    # 转换成数组
    np.empty(mask_gray.shape, dtype=np.float32)
    plt.imshow(result)
    plt.show()

    cv2.drawContours(mask, contours, -1, (0, 255, 255), 1)
    plt.imshow(mask)
    plt.show()


if __name__ == '__main__':
    start = time.perf_counter()
    find_circle('../pic/1.jpg')
    end = time.perf_counter()
    print("运行耗时", end - start)
