import time

import cv2
import matplotlib.pyplot as plt



def find_circle(path):
    mask = cv2.imread(path)
    plt.imshow(mask)
    plt.show()
    # 图片的放大操作
    # mask = cv2.resize(mask, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    plt.imshow(mask)
    plt.show()
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_gray[mask_gray < 50] = 0
    mask_gray[mask_gray >= 50] = 255
    # 进行反色处理 因为我们要获取的是白色被轮廓包裹的区域
    # mask_gray = 255-mask_gray
    plt.imshow(mask_gray)
    plt.show()
    # 识别轮廓
    # 进行放大处理

    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    # 转换成数组
    # np.empty(mask_gray.shape, dtype=np.float32)
    plt.imshow(result)
    plt.show()
    cv2.drawContours(mask, contours, -1, (0, 255, 255), 1)
    cv2.imwrite('../realPic/5_contour.jpg',mask)
    plt.imshow(mask)
    plt.show()


if __name__ == '__main__':
    start = time.perf_counter()
    find_circle('../realPic/6.jpg')
    end = time.perf_counter()
    print("运行耗时", end - start)
