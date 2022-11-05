import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# 这个的话 是通过内嵌的函数来进行计算  我认为难免会产生一定的问题

# 直接测试实战的

def find_circle(mask):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_gray[mask_gray < 50] = 0
    mask_gray[mask_gray >= 50] = 255
    # 进行反色处理  因为我找到的里面是要白的
    mask_gray = 255 - mask_gray
    # 识别轮廓   只检测外轮廓   然后获取轮廓总面积 然后对轮廓内的黑色部分进行计算
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    # 转换成数组
    np.empty(mask_gray.shape, dtype=np.float32)
    plt.imshow(result)
    plt.show()
    ct = 0
    x, y = mask_gray.shape
    pixels_black = 0
    pixels = 0
    for contour in contours:
        # 找到轮廓内的黑色面积
        # 不用contourArea函数 直接用查找像素点的个数会不会好一点  这边再给出一种用边缘内像素个数来计算连通域面积的方法
        # 对每一个连通域使用一个掩码模板计算非0像素(即连通区域像素个数)
        singe_mask = np.zeros((x, y))
        now_gray = 255 - mask_gray
        # 把线的面积找出来
        pixels_black = pixels_black + cv2.countNonZero(now_gray)
        # 把整个轮廓的面积找出来
        fill_image = cv2.fillConvexPoly(singe_mask, contour, 255)
        pixels = pixels + cv2.countNonZero(fill_image)

    plt.imshow(result)
    plt.show()
    ratio = 1 - pixels_black / pixels
    return pixels, ratio


if __name__ == '__main__':
    start = time.perf_counter()
    mask = cv2.imread('../realPic/7.jpg')
    whole_area, ratio = find_circle(mask)
    end = time.perf_counter()
    print("运行耗时", end - start)
    print("轮廓内面积为", whole_area)
    print("孔隙率为", ratio)
