import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# 蒙特卡洛方法 计算不规则图像面积

def find_circle(mask):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_gray[mask_gray < 50] = 0
    mask_gray[mask_gray >= 50] = 255
    # 进行反色处理  因为我找到的里面是要白的
    print(mask_gray)
    # 转换成数组
    count = 0
    total = 8000000
    wide = mask.shape[0]
    high = mask.shape[1]
    full_area = wide*high
    for i in range(total):
        x = np.random.randint(high)
        y = np.random.randint(wide)
        if mask_gray[y][x] == 255:
            count += 1
    return full_area*(count/total)


if __name__ == '__main__':
    start = time.perf_counter()
    mask = cv2.imread('../test/test5.png')
    whole_area = find_circle(mask)
    end = time.perf_counter()
    print("运行耗时", end - start)
    print("轮廓内面积为", whole_area)
