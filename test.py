#!/usr/bin/env python
# coding=utf-8

"""
测试双摄像头
测试屏幕输出提示信息
1、太远，请靠近
2、太近，请远离
3、移动过大，请保持稳定
a、疑似照片
b、疑似电子设备
c、活体
"""

import cv2
import threading

capturing = {
    'is': True,
    'msg': 0
}


def draw_str(dst, target, s, front=(255, 255, 255), back=(0, 0, 0), size=1.0):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, size, back, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, size, front, lineType=cv2.LINE_AA)


def camera(ca_id):
    """
    开启摄像头
    :param ca_id:
    :return:
    """
    cap = cv2.VideoCapture(ca_id)
    count = 0
    while capturing['is']:
        key = cv2.waitKey(1)
        if key == 27:  # Esc for exit
            capturing['is'] = False
            break
        elif key == 113:
            count += 1
            print capturing['msg']
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        draw_str(frame, (20, 40), str(count), size=2.0)
        cv2.imshow('Cap\t' + str(ca_id), frame)


def main():
    video_srcs = [0, 1]
    tsk = []
    for video_src in video_srcs:
        t = threading.Thread(target=camera, args=(video_src,))
        t.start()
        tsk.append(t)
    for t in tsk:
        t.join()
    print 'start capture.'


if __name__ == '__main__':
    main()
