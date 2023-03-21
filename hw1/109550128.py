import numpy as np
import cv2
import math

def swap(image, sample):    
    for i in range(120):
        for j in range(200):
            image[i][j] = sample[i][j + 400]
            image[i][j + 400] = sample[i][j]
    return image

def gray_256(image):
    for i in range(240, 360):
        for j in range(200):
            sum = 0
            for k in range(3):
                sum = sum + image[i][j][k]
            sum = sum / 3
            image[i][j][0] = sum
            image[i][j][1] = sum
            image[i][j][2] = sum
    return image

def gray_4(image):
    for i in range(240, 360):
        for j in range(400, 600):
            sum = 0
            for k in range(3):
                sum = sum + image[i][j][k]
            sum = sum / 3
            sum = math.floor(sum / 64) * 64
            image[i][j][0] = sum 
            image[i][j][1] = sum
            image[i][j][2] = sum
    return image

def filter_red(image):
    for i in range(120, 240):
        for j in range(200):
            if image[i][j][2] <= 150 or image[i][j][2] * 0.6 <= image[i][j][1] or image[i][j][2] * 0.6 <= image[i][j][0]:
                sum = 0
                for k in range(3):
                    sum = sum + image[i][j][k]
                sum = sum / 3
                image[i][j][0] = sum
                image[i][j][1] = sum
                image[i][j][2] = sum
    return image

def filter_yellow(image):
    for i in range(120, 240):
        for j in range(400, 600):
            cond  = 0
            delta = 0
            delta += image[i][j][1]
            cond  += image[i][j][1]
            delta -= image[i][j][2]
            cond  += image[i][j][2]
            if abs(delta) >= 50 or cond * 0.3 <= image[i][j][0]:
                sum = 0
                for k in range(3):
                    sum = sum + image[i][j][k]
                sum = sum / 3
                image[i][j][0] = sum
                image[i][j][1] = sum
                image[i][j][2] = sum
    return image

def channel_operation(image):
    for i in range(240, 360):
        for j in range(200, 400):
            if image[i][j][1] > 127:
                image[i][j][1] = 255
            else:
                image[i][j][1] = image[i][j][1] * 2
    return image

def bilinear(image, sample):
    for i in range(120):
        for j in range(200):
            for k in range(3):
                sum = 0
                sum = sum + sample[math.floor(i / 2)][math.floor(j / 2) + 200][k] + sample[math.ceil(i / 2)][math.floor(j / 2) + 200][k]
                sum = sum + sample[math.floor(i / 2)][math.ceil(j / 2)  + 200][k] + sample[math.ceil(i / 2)][math.ceil(j / 2)  + 200][k]
                sum = sum / 4
                image[i][j + 200][k] = sum
    return image

def cubic(p0, p1, p2, p3, x):
    func = 0.0
    func = func + (-0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3) * x ** 3
    func = func + (p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3) * x ** 2
    func = func + (-0.5 * p0 + 0.5 * p2) * x + p1
    if func < 0:
        return 0
    elif func > 255:
        return 255
    return func

def bicubic(image, sample):
    for i in range(120):
        for j in range(200):
            sample_x = (i // 2) + 120
            sample_y = (j // 2) + 200
            val = []
            for dx in range(4):
                color = []
                idx_row = sample_x + dx - 1
                for k in range(3):
                    p0 = sample[idx_row][sample_y - 1][k]
                    p1 = sample[idx_row][sample_y][k]
                    p2 = sample[idx_row][sample_y + 1][k]
                    p3 = sample[idx_row][sample_y + 2][k]
                    color.append(cubic(p0, p1, p2, p3, (j / 2.0) - (j // 2)))
                val.append(color)
            for k in range(3):
                p0 = val[0][k]
                p1 = val[1][k]
                p2 = val[2][k]
                p3 = val[3][k]
                image[i + 120][j + 200][k] = cubic(p0, p1, p2, p3, (j / 2.0) - (j // 2))
    return image

if __name__ == '__main__':
    image  = cv2.imread('test.jpg')
    sample = cv2.imread('test.jpg')

    image = swap(image, sample)
    image = gray_256(image)
    image = gray_4(image)
    image = filter_red(image)
    image = filter_yellow(image)
    image = channel_operation(image)
    image = bilinear(image, sample)
    image = bicubic(image, sample)


    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('109550128.jpg', image)