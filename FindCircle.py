import random
import time

import cv2
import math
import numpy as np
from numpy.ma import cos, sin
import matplotlib.pyplot as plt


def max_circle(f):

    img = cv2.imread(f, cv2.IMREAD_COLOR)
    plt.imshow(img)
    plt.show()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contous, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contous:
        left_x = min(c[:, 0, 0])
        right_x = max(c[:, 0, 0])
        down_y = max(c[:, 0, 1])
        up_y = min(c[:, 0, 1])
        upper_r = min(right_x - left_x, down_y - up_y) / 2
        # 定义相切二分精度
        precision = math.sqrt((right_x - left_x) ** 2 + (down_y - up_y) ** 2) / (2 ** 13)
        # 构造包含轮廓的矩形的所有像素点
        Nx = 2 ** 8
        Ny = 2 ** 8
        pixel_X = np.linspace(left_x, right_x, Nx)
        pixel_Y = np.linspace(up_y, down_y, Ny)
        # [pixel_X, pixel_Y] = ndgrid(pixel_X, pixel_Y);
        # pixel_X = reshape(pixel_X, numel(pixel_X), 1);
        # pixel_Y = reshape(pixel_Y, numel(pixel_Y), 1);
        xx, yy = np.meshgrid(pixel_X, pixel_Y)
        # % 筛选出轮廓内所有像素点
    in_list = []
    for c in contous:
        for i in range(pixel_X.shape[0]):
            for j in range(pixel_X.shape[0]):
                if cv2.pointPolygonTest(c, (xx[i][j], yy[i][j]), False) > 0:
                    in_list.append((xx[i][j], yy[i][j]))
    in_point = np.array(in_list)
    # pixel_X = in_point[:, 0]
    # pixel_Y = in_point[:, 1]
    # 随机搜索百分之一像素提高内切圆半径下限
    N = len(in_point)
    rand_index = random.sample(range(N), N // 100)
    rand_index.sort()
    radius = 0
    big_r = upper_r
    center = None
    for id in rand_index:
        tr = iterated_optimal_incircle_radius_get(c, in_point[id][0], in_point[id][1], radius, big_r, precision)
        if tr > radius:
            radius = tr
            center = (in_point[id][0], in_point[id][1])  # 只有半径变大才允许位置变更，否则保持之前位置不变
    # 循环搜索剩余像素对应内切圆半径
    loops_index = [i for i in range(N) if i not in rand_index]
    for id in loops_index:
        tr = iterated_optimal_incircle_radius_get(c, in_point[id][0], in_point[id][1], radius, big_r, precision)
        if tr > radius:
            radius = tr
            center = (in_point[id][0], in_point[id][1])  # 只有半径变大才允许位置变更，否则保持之前位置不变
    # 效果测试
    plot_x = np.linspace(0, 2 * math.pi, 100)
    circle_X = center[0] + radius * cos(plot_x)
    circle_Y = center[1] + radius * sin(plot_x)
    print(radius * 2)
    plt.figure()
    plt.imshow(img_gray)
    plt.plot(circle_X, circle_Y)
    plt.show()


# 持续的获得圆的半径的函数
def iterated_optimal_incircle_radius_get(contous, pixelx, pixely, small_r, big_r, precision):
    radius = small_r
    L = np.linspace(0, 2 * math.pi, 360)  # 确定圆散点剖分数360, 720
    circle_X = pixelx + radius * cos(L)
    circle_Y = pixely + radius * sin(L)
    for i in range(len(circle_Y)):
        if cv2.pointPolygonTest(contous, (circle_X[i], circle_Y[i]), False) < 0:  # 如果圆散集有在轮廓之外的点
            return 0
    while big_r - small_r >= precision:  # 二分法寻找最大半径
        half_r = (small_r + big_r) / 2
        circle_X = pixelx + half_r * cos(L)
        circle_Y = pixely + half_r * sin(L)
        if_out = False
        for i in range(len(circle_Y)):
            if cv2.pointPolygonTest(contous, (circle_X[i], circle_Y[i]), False) < 0:  # 如果圆散集有在轮廓之外的点
                big_r = half_r
                if_out = True
        if not if_out:
            small_r = half_r
    radius = small_r
    return radius


if __name__ == '__main__':
    print(time.thread_time())
    max_circle('irregular.png')
    print(time.thread_time())
