# 导进多进程的包
import multiprocessing
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# 新方法多进程版本


def find_circle(path):
    mask = cv2.imread(path)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_gray[mask_gray < 130] = 0
    mask_gray[mask_gray >= 130] = 255
    # 识别轮廓
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    raw_dist = np.empty(mask_gray.shape, dtype=np.float32)

    plt.imshow(result)
    plt.show()
    plt.figure()

    process_queue = multiprocessing.Manager().Queue()
    current_cpu_count = multiprocessing.cpu_count()
    process_pool = multiprocessing.Pool(processes=current_cpu_count)

    for contour in contours:
        # 画圈
        process_pool.apply_async(draw, args=(contour, mask_gray, raw_dist, process_queue))

    process_pool.close()  # 进程池关闭后不再接收新的请求
    process_pool.join()  # 等待所有进程池的进程执行完毕

    # cv2.imshow('Maximum inscribed circle', result)
    # cv2.waitKey(0)
    while not process_queue.empty():
        circle_element = process_queue.get()
        cv2.circle(result, circle_element[0], circle_element[1], (0, 255, 0), 2, 1, 0)

    plt.imshow(result)
    plt.show()


def draw(contour, mask_gray, raw_dist, process_queue):
    for i in range(mask_gray.shape[0]):
        for j in range(mask_gray.shape[1]):
            raw_dist[i, j] = cv2.pointPolygonTest(contour, (j, i), True)  # 检测点是否在轮廓内  然后获取这个点离轮廓的最近距离
    # 获取最大值即内接圆半径，中心点坐标   找出最大值和最小值还有他们的坐标
    _, max_val, _, max_dist_pt = cv2.minMaxLoc(raw_dist)
    max_val = abs(max_val)
    # 画出最大内接圆  cvtColor颜色转换空间
    radius = np.int(max_val)
    process_queue.put([max_dist_pt, radius])


if __name__ == '__main__':
    start = time.perf_counter()
    find_circle('../pic/bigTrue.png')
    end = time.perf_counter()
    print("运行耗时", end - start)
