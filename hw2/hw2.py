import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def compute_histogram(image, row, col):
    histogram = np.zeros(256)
    for i in range(row):
        for j in range(col):
            histogram[image[i][j][0]] += 1
    return histogram

def transform_func(histogram, row, col):
    sum = 0
    transform = np.zeros(256)
    for i in range(256):
        sum += histogram[i]
        transform[i] = round(sum * 255 / (row * col)) 
    return transform

def Gaussian(r, alpha):
    return math.e ** (-1 * (r ** 2) / (2 * alpha * alpha))


if __name__ == '__main__':
    image  = cv2.imread('Q1.jpg')
    row, col, channel = image.shape
    histogram = compute_histogram(image, row, col)
    transform = transform_func(histogram, row, col)

    for i in range(row):
        for j in range(col):
            for c in range(channel):
                image[i][j][c] = transform[image[i][j][c]]

    cv2.imwrite('Q1_ans.jpg', image)

    image = cv2.imread('Q1.jpg')
    image2 = cv2.imread('Q2.jpg')
    row2, col2, channel2 = image2.shape
    histogram2 = compute_histogram(image2, row2, col2)
    transform2 = transform_func(histogram2, row2, col2)


    for i in range(row):
        for j in range(col):
            step = transform[image[i][j][0]]
            decide = 0
            for k in range(256):
                if transform2[k] - step == 0:
                    decide = k
                    break
                elif transform2[k] > step:
                    if transform2[k] - step < step - transform2[k - 1]:
                        decide = k
                    break
                else:
                    decide = k
            for c in range(channel):
                image[i][j][c]= decide

    histogram3 = compute_histogram(image, row, col)
    transform3 = transform_func(histogram3, row, col)

    plt.plot(transform, label = 'Q1')
    plt.plot(transform2, label = 'Q2')
    plt.plot(transform3, label = 'Q1_new')
    plt.show()
    
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    cv2.imwrite('Q2_ans.jpg', image)  

    image = cv2.imread('Q3.jpg')
    row, col, channel = image.shape
    kernel = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            distance = math.sqrt((i - (5 // 2)) ** 2 + (j - (5 // 2)) ** 2)
            value = Gaussian(distance, 25)
            kernel[i, j] = value

    kernel /= np.sum(kernel)
    one_channel = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            one_channel[i][j] = image[i][j][0]
    one_channel = np.pad(one_channel, ((2, 2), (2, 2)), mode='constant')
    for i in range(row):
        for j in range(col):
            tmp = np.sum(one_channel[i : i + 5, j : j + 5] * kernel)
            image[i, j] = [tmp, tmp, tmp]

    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    cv2.imwrite('Q3_ans.jpg', image)  
