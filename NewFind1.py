import cv2
import numpy as np
import time

if __name__ == '__main__':
    t0 = time.time()
    mask_gray = cv2.imread('pic/diamond.png', 0)
    rows, cols = mask_gray.shape
    # Get the contours
    print(f'原图尺寸{mask_gray.shape}')
    # 识别轮廓
    _, mask_grayqqqq = cv2.threshold(mask_gray, 50, 255, cv2.THRESH_BINARY)
    contours2, _ = cv2.findContours(mask_grayqqqq, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mm = cv2.resize(mask_grayqqqq, (int(cols / 8 / 4), int(rows / 8 / 4)))
    contours, _ = cv2.findContours(mm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
        if 100 < cv2.contourArea(i):
            # print(cv2.contourArea(i))
            cont = i

    for i in contours2:
        if 5000 < cv2.contourArea(i):
            # print(cv2.contourArea(i))
            cont2 = i

    # 计算到轮廓的距离
    x, y, w, h = cv2.boundingRect(cont)
    raw_dist = np.zeros_like(mm, dtype=np.float32)

    raw_dist2 = np.zeros_like(mask_grayqqqq, dtype=np.float32)

    for i in range(int(x + w / 4), int(x + w / 4 * 3)):
        for j in range(int(y + h / 4), int(y + h / 4 * 3)):
            raw_dist[i, j] = cv2.pointPolygonTest(cont, (j, i), True)

    # 获取最大值即内接圆半径，中心点坐标
    minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
    print(f'缩放后的圆心{maxDistPt}')

    for i in range(maxDistPt[1] * 8 * 4 - 25, maxDistPt[1] * 8 * 4 + 25):
        for j in range(maxDistPt[0] * 8 * 4 - 25, maxDistPt[0] * 8 * 4 + 25):
            raw_dist2[i, j] = cv2.pointPolygonTest(cont2, (j, i), True)

    minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist2)
    maxVal = abs(maxVal)
    # 画出最大内接圆
    result = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    print(f'原图的圆心{maxDistPt}')

    center_of_circle = maxDistPt
    cv2.circle(result, maxDistPt, int(maxVal), (0, 255, 0), 2, 1, 0)
    # cv2.imwrite('./sawww.bmp',result)
    result = cv2.resize(result, (1024, 1024))

    print(f'耗时{time.time() - t0}')
    cv2.imshow('Maximum inscribed circle', result)
    cv2.waitKey(0)
