import math
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import cos, sin


# 可以尝试多线程  为了快  之后看看

def max_circle(f):
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    plt.imshow(img)
    plt.show()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plot_x = np.linspace(0, 2 * math.pi, 100)
    plt.figure()
    plt.imshow(img_gray)
    # 用多线程来解决速度缓慢的问题
    for c in contours:
        left_x = min(c[:, 0, 0])
        right_x = max(c[:, 0, 0])
        down_y = max(c[:, 0, 1])
        up_y = min(c[:, 0, 1])
        upper_r = min(right_x - left_x, down_y - up_y) / 2
        # 定义相切二分精度
        precision = math.sqrt((right_x - left_x) ** 2 + (down_y - up_y) ** 2) / (2 ** 13)
        # 构造包含轮廓的矩形的所有像素点
        n_x = 2 ** 8
        n_y = 2 ** 8
        pixel_x = np.linspace(left_x, right_x, n_x)
        pixel_y = np.linspace(up_y, down_y, n_y)
        # [pixel_x, pixel_y] = ndgrid(pixel_x, pixel_y);
        # pixel_x = reshape(pixel_x, numel(pixel_x), 1);
        # pixel_y = reshape(pixel_y, numel(pixel_y), 1);
        xx, yy = np.meshgrid(pixel_x, pixel_y)
        # % 筛选出轮廓内所有像素点
        in_list = []
        for i in range(pixel_x.shape[0]):
            for j in range(pixel_x.shape[0]):
                if cv2.pointPolygonTest(c, (xx[i][j], yy[i][j]), False) > 0:
                    in_list.append((xx[i][j], yy[i][j]))
        in_point = np.array(in_list)
        # pixel_x = in_point[:, 0]
        # pixel_y = in_point[:, 1]
        # 随机搜索百分之一像素提高内切圆半径下限
        n = len(in_point)
        rand_index = random.sample(range(n), n // 100)
        rand_index.sort()
        radius = 0
        big_r = upper_r
        center = None
        for rand_id in rand_index:
            tr = iterated_optimal_in_circle_radius_get(c, in_point[rand_id][0], in_point[rand_id][1], radius, big_r,
                                                       precision)
            if tr > radius:
                radius = tr
                center = (in_point[rand_id][0], in_point[rand_id][1])  # 只有半径变大才允许位置变更，否则保持之前位置不变
        # 循环搜索剩余像素对应内切圆半径
        loops_index = [i for i in range(n) if i not in rand_index]
        for loop_id in loops_index:
            tr = iterated_optimal_in_circle_radius_get(c, in_point[loop_id][0], in_point[loop_id][1], radius, big_r,
                                                       precision)
            if tr > radius:
                radius = tr
                center = (in_point[loop_id][0], in_point[loop_id][1])  # 只有半径变大才允许位置变更，否则保持之前位置不变
        circle_x = center[0] + radius * cos(plot_x)
        circle_y = center[1] + radius * sin(plot_x)
        print("最终半径为", radius)
        plt.plot(circle_x, circle_y)

    plt.show()


# 持续的获得圆的半径的函数
def iterated_optimal_in_circle_radius_get(contours, pixel_x, pixel_y, small_r, big_r, precision):
    radius = small_r
    circle_distribute = np.linspace(0, 2 * math.pi, 360)  # 确定圆散点剖分数360, 720  就是圆半径的分散
    circle_x = pixel_x + radius * cos(circle_distribute)
    circle_y = pixel_y + radius * sin(circle_distribute)
    for i in range(len(circle_y)):
        if cv2.pointPolygonTest(contours, (circle_x[i], circle_y[i]), False) < 0:  # 如果圆散集有在轮廓之外的点
            return 0
    while big_r - small_r >= precision:  # 二分法寻找最大半径
        half_r = (small_r + big_r) / 2
        circle_x = pixel_x + half_r * cos(circle_distribute)
        circle_y = pixel_y + half_r * sin(circle_distribute)
        if_out = False
        for i in range(len(circle_y)):
            if cv2.pointPolygonTest(contours, (circle_x[i], circle_y[i]), False) < 0:  # 如果圆散集有在轮廓之外的点
                big_r = half_r
                if_out = True
        if not if_out:
            small_r = half_r
    radius = small_r
    return radius


if __name__ == '__main__':
    start = time.perf_counter()
    max_circle('pic/four.png')
    end = time.perf_counter()
    print("运行耗时", end - start)
