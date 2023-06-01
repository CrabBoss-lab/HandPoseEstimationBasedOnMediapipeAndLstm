# -*- codeing = utf-8 -*-
# @Time :2023/4/3 14:23
# @Author :yujunyu
# @Site :
# @File :test_use_cap.py
# @software: PyCharm

import cv2


def built_in_cap():
    import cv2

    cap = cv2.VideoCapture(0)  # VideoCapture()中参数是0，表示打开笔记本的内置摄像头
    cv2.namedWindow('camera')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('camera', frame)
            cv2.waitKey(1)


def use_cap():
    cap = cv2.VideoCapture(1)  # VideoCapture()中参数是1，表示打开外接usb摄像头
    cv2.namedWindow('camera')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('camera', frame)
            cv2.waitKey(1)



if __name__ == '__main__':
    use_cap()

