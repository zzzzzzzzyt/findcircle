import cv2
import numpy as np
import time


def find_circle(path):
    mask = cv2.imread(path)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 识别轮廓
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        # 计算到轮廓的距离
        raw_dist = np.empty(mask_gray.shape, dtype=np.float32)
        for i in range(mask_gray.shape[0]):
            for j in range(mask_gray.shape[1]):
                raw_dist[i, j] = cv2.pointPolygonTest(contour, (j, i), True)  # 检测点是否在轮廓内  然后获取这个点离轮廓的最近距离
        # 获取最大值即内接圆半径，中心点坐标   找出最大值和最小值还有他们的坐标
        _, max_val, _, max_dist_pt = cv2.minMaxLoc(raw_dist)
        max_val = abs(max_val)
        # 画出最大内接圆  cvtColor颜色转换空间
        radius = np.int(max_val)
        cv2.circle(result, max_dist_pt, radius, (0, 255, 0), 2, 1, 0)
    cv2.imshow('Maximum inscribed circle', result)
    # cv2.waitKey(0)


if __name__ == '__main__':
    start = time.perf_counter()
    find_circle('pic/four.png')
    end = time.perf_counter()
    print("运行耗时", end - start)
