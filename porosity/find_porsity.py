import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# 这个的话 是通过内嵌的函数来进行计算  我认为难免会产生一定的问题

# 直接测试实战的

def find_circle(path):
    mask = cv2.imread(path)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_gray[mask_gray < 50] = 0
    mask_gray[mask_gray >= 50] = 255
    # 进行反色处理  因为我找到的里面是要白的
    mask_gray = 255 - mask_gray
    # 识别轮廓
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    # 转换成数组
    np.empty(mask_gray.shape, dtype=np.float32)
    plt.imshow(result)
    plt.show()
    area = 0
    max_area = 0
    ct = 0
    for contour in contours:
        # 计算到轮廓的距离
        # x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(result, (x, y), (x + w, y + h), (153, 153, 0), 5)
        temp_area = cv2.contourArea(contour)
        area += temp_area
        if temp_area>max_area:
            max_area = temp_area
            ct= [ contour]
        print("面积是", temp_area)
    # cv2.imshow('Maximum inscribed circle', result)
    # cv2.waitKey(0)
    # 找最大的
    cv2.drawContours(result, ct, -1, (0, 255, 255), 1)
    plt.imshow(result)
    plt.show()
    ratio = (area-max_area)/max_area
    return max_area,ratio


if __name__ == '__main__':
    start = time.perf_counter()
    whole_area,ratio = find_circle('../realPic/5.jpg')
    end = time.perf_counter()
    print("运行耗时", end - start)
    print("轮廓内面积为", whole_area)
    print("孔隙率为", ratio)
