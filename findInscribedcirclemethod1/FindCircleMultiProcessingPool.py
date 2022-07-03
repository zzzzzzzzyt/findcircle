import math
# 导进多进程的包
import multiprocessing
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import cos, sin


# 多进程版本
def max_circle(photo_path):
    # cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道 cv2.IMREAD_GRAYSCALE：读入灰度图片
    # cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道
    img = cv2.imread(photo_path, cv2.IMREAD_COLOR)
    # plt.imshow(img)
    # 用来显示
    # plt.show()

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray[img_gray < 130] = 0
    img_gray[img_gray >= 130] = 255

    # 进行反色处理
    # img_gray = 255 - img_gray

    plt.imshow(img_gray)
    plt.show()
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plot_x = np.linspace(0, 2 * math.pi, 100)
    plt.figure()
    plt.imshow(img_gray)
    # 创建进程组 目的是因为主进程是需要等待所有线程结束 才进行执行 我们并不能确定有多少个进程在执行 所以要创建相应的数组 存放进程

    # 用多进程来解决速度缓慢的问题  同时要考虑到进程同步的过程  我们要等所有进程结束 我们才能显示相应的圆
    # 队列还是设置了一下限制
    # 要创建下 相应的进程锁 等锁解完 才能继续运行   因为我是四核所以可以同时运行四个进程
    process_queue = multiprocessing.Manager().Queue()
    # 利用进程池进行计算  进程池中的进程的数量跟我的电脑cpu核数一致  变通 根据使用的电脑处理器核数进行处理
    current_cpu_count = multiprocessing.cpu_count()
    process_pool = multiprocessing.Pool(processes=current_cpu_count)

    for c in contours:
        # 进行异步执行 怎么根本没进去
        process_pool.apply_async(draw_circle, args=(c, plot_x, process_queue))

    process_pool.close()  # 进程池关闭后不再接收新的请求
    process_pool.join()  # 等待所有进程池的进程执行完毕

    # 将队列中的参数全部取出
    while not process_queue.empty():
        temp = process_queue.get()
        plt.plot(temp[0], temp[1])
    plt.show()


#  提取出来的目的 就是用来当线程里面执行的参数
def draw_circle(c, plot_x, queue):
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
    xx, yy = np.meshgrid(pixel_x, pixel_y)
    # % 筛选出轮廓内所有像素点
    in_list = []
    for i in range(pixel_x.shape[0]):
        for j in range(pixel_x.shape[0]):
            if cv2.pointPolygonTest(c, (xx[i][j], yy[i][j]), False) > 0:
                in_list.append((xx[i][j], yy[i][j]))
    in_point = np.array(in_list)
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
    queue.put([circle_x, circle_y])


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
        circle_y = pixel_y + half_r * sin(circle_distribute)
        circle_x = pixel_x + half_r * cos(circle_distribute)
        if_out = False
        for i in range(len(circle_y)):
            if cv2.pointPolygonTest(contours, (circle_x[i], circle_y[i]), False) < 0:  # 如果圆散集有在轮廓之外的点
                big_r = half_r
                if_out = True
        if not if_out:
            small_r = half_r
    radius = small_r
    # print("迭代半径为：", radius)
    return radius


if __name__ == '__main__':
    start = time.perf_counter()
    # global_array = multiprocessing.Manager().Array()
    max_circle('../pic/bigTrue.png')
    end = time.perf_counter()
    print("运行耗时", end - start)
