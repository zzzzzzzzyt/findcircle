"""
@Time ： 2022/7/3 14:35
@Auth ： 祝英台炸油条
@File ：read_pic.py
"""

# 检测下用什么方法可以使读入的图片损失没有那么严重
import time
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start = time.perf_counter()
    pic = cv2.imread('../pic/11.png')
    print(pic.shape)

    plt.imshow(pic)
    plt.show()
    end = time.perf_counter()
    print("运行耗时", end - start)