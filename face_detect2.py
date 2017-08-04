#!/usr/bin/env python
# coding=utf-8
"""
Face detect

Keys:
    ESC    - exit

测试了
检测整张图片光流场的效果，并只显示人脸范围的光流场情况
这时光流场特别混乱，严重受到人脸范围外的图形影响
"""

import cv2
import numpy as np
import math
import time
from common import clock


def scale((x0, y0), (x1, y1), scaling=1.0, limit=None):
    """scaling. limit is a array [w, h]"""
    scaling = scaling - 1
    xd = int(abs(x1 - x0) * scaling / 2)
    yd = int(abs(y1 - y0) * scaling / 2)

    rx0 = x0 - xd
    ry0 = y0 - yd
    rx1 = x1 + xd
    ry1 = y1 + yd

    if limit is not None:
        if rx0 < 0:
            rx0 = 0
        if ry0 < 0:
            ry0 = 0
        if rx1 > limit[0]:
            rx1 = limit[0]
        if ry1 > limit[1]:
            ry1 = limit[1]

    return (rx0, ry0), (rx1, ry1)


def draw_rectangles(img, rectangles, color=(0, 255, 0), scaling=1.0):
    """draw rectangle. """
    for x0, y0, x1, y1 in rectangles:
        xd, yd = scale((x0, y0), (x1, y1), scaling)
        cv2.rectangle(img, xd, yd, color, 2)


def draw_rectangle(img, rectangle, color=(0, 255, 0), scaling=1.0):
    """draw rectangle. """
    x0, y0, x1, y1 = rectangle
    xd, yd = scale((x0, y0), (x1, y1), scaling)
    cv2.rectangle(img, xd, yd, color, 2)


def detect(img, cascade, option=None):
    """detect base on cascade type"""
    if option is None:
        option = {
            "scaleFactor": 1.4,
            "minNeighbors": 4,
            "minSize": (30, 30),
            "flags": cv2.CASCADE_SCALE_IMAGE
        }
    # detects = cascade.detectMultiScale(img, scaleFactor=option["scaleFactor"], minNeighbors=option["minNeighbors"],
    #                                    minSize=option["minSize"], flags=option["flags"])
    detects = cascade.detectMultiScale(img, **option)
    if len(detects) == 0:
        return []
    detects[:, 2:] += detects[:, :2]
    return detects


def draw_flow(img, area, flow, step=16, color=(0, 0, 255)):
    """draw as line with flow"""
    h, w = img.shape[:2]
    rx0, ry0, rx1, ry1 = area

    sx = 0
    if rx0 % step != 0:
        sx += step
    sx += rx0 / step * step

    sy = 0
    if ry0 % step != 0:
        sy += step
    sy += ry0 / step * step

    y, x = np.mgrid[sy:ry1:step, sx:rx1:step].reshape(2, -1).astype(int)

    if y.shape[:1] != x.shape[:1]:
        print y.shape, x.shape
        return
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img, lines, 0, color)  # draw lines
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)  # redraw the start point
    return lines


def opt_flow(p_gray, n_gray, option=None):
    def_option = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    if option is not None:
        def_option.update(option)
    flow = cv2.calcOpticalFlowFarneback(p_gray, n_gray, None, **def_option)
    return flow


def anti_spoofing(lines):  # for angle
    """anti spoofing"""
    # return  True
    if offset_check(lines, 8, 0.0017):
        return False
    vs = []
    (lu, ru, rd, ld) = (0, 0, 0, 0)
    for (x0, y0), (x1, y1) in lines:
        if x1 - x0 < 0 and y1 - y0 > 0:
            lu += 1
            vs.append(math.atan(abs((y1 - y0) / (x1 - x0))) / math.pi * 180)
        elif x1 - x0 > 0 and y1 - y0 > 0:
            ru += 1
            vs.append(180 - math.atan(abs((y1 - y0) / (x1 - x0))) / math.pi * 180)
        elif x1 - x0 > 0 and y1 - y0 < 0:
            rd += 1
            vs.append(180 + math.atan(abs((y1 - y0) / (x1 - x0))) / math.pi * 180)
        elif x1 - x0 < 0 and y1 - y0 < 0:
            ld += 1
            vs.append(360 - math.atan(abs((y1 - y0) / (x1 - x0))) / math.pi * 180)
    if len(vs) == 0:
        return False
    else:
        fc = var_angle(vs)
        # if 13000 > fc > 9000:
        #     print fc
        return 13000 > fc > 9000


def anti_spoofing2(lines):  # for offset
    """
    with module of vector
    根据偏移中不一致性的 方差进行判断，可排除静止照片
    """
    if offset_check(lines):
        # if not offset_check(lines, 2.0, 3.0, 0.02):
        return False
    # mds = []
    # for (x0, y0), (x1, y1) in lines:
    #     md = math.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
    #     mds.append(md)
    # md_avg = sum(mds) / len(mds)
    # sum_pow = 0
    # for md in mds:
    #     sum_pow += math.pow(md - md_avg, 2)
    # md_var = 1.0 * sum_pow / len(mds)
    # # if md_var > 0.1:
    # #     print md_var
    md_var = var_offset(lines)
    return 0.2 > md_var > 0.1  # 人眼眨动范围大概在 0.2-0.1


def position_check_eye(lines, eye_areas):
    """
    check if the offset include in eys area.
    偏移 30%在识别出的眼睛的范围内即认为是眼睛在眨动（眼睛的识别框有可能只有一个，所以占比会比较低些）
    """
    lines = offset_check_get(lines, 1, 8)
    if len(lines) == 0:
        return False
    inc = []
    for rx0, ry0, rx1, ry1 in eye_areas:
        (rx0, ry0), (rx1, ry1) = scale((rx0, ry0), (rx1, ry1), 1.1)
        for line in lines:
            (x0, y0), (x1, y1) = line
            if rx0 < x0 < rx1 and rx0 < x1 < rx1 and ry0 < y0 < ry1 and ry0 < y1 < ry1:
                inc.append(line)
    fc = var_offset(inc)
    print 1.0 * len(inc) / len(lines)
    # print "--------"
    print fc
    # print inc
    print '--------'
    return 1.0 * len(inc) / len(lines) > 0.01 and fc > 0.06


def var_offset(lines):
    """
    向量 模 的 方差
    :param lines:
    :return:
    """
    if len(lines) == 0:
        return 0
    mds = []
    for (x0, y0), (x1, y1) in lines:
        md = math.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
        mds.append(md)
    md_avg = sum(mds) / len(mds)
    sum_pow = 0
    for md in mds:
        sum_pow += math.pow(md - md_avg, 2)
    md_var = 1.0 * sum_pow / len(mds)
    # if md_var > 0.1:
    #     print md_var
    return md_var


def var_angle(array):
    """variance"""
    if len(array) == 0:
        return
    sum = 0
    for num in array:
        sum += num
    avg = sum / len(array)

    sum2 = 0

    for num in array:
        jj = num - avg
        if jj > 180:
            jj = 360 - jj
        sum2 += math.pow(jj, 2)
    rs = sum2 / len(array)
    # df = open('d:\\a.txt', 'w')
    # for a in array:
    #     df.write(str(a) + '\t')
    #     print(a)
    # df.close()
    return rs


def offset_check(vectors, min_offset=3.0, percent=0.8):
    """
    offset check. if the vector module between min_offset and max_offset, and below percent return true
    排除照片的大幅度移动
    偏移幅度大于3.0 且占所有的偏移点大于80%则判定为照片的大幅度移动
    这里由于人脸为立体的且背景基本保持不变，这样当把识别出的人脸范围扩大后判断偏移可排除部分有背景的照片的移动
    """
    count = len(offset_check_get(vectors, min_offset))
    # print 1.0 * count
    # print len(vectors)
    # print 1.0 * count / len(vectors)
    if 1.0 * count / len(vectors) > percent:
        print '疑似照片' + str(time.time())
    return 1.0 * count / len(vectors) > percent


def offset_check_get(vectors, min_offset=0.0, max_offset=10.0):
    vectors_pass = []
    for vector in vectors:
        (x0, y0), (x1, y1) = vector
        length = math.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
        # if 2.0 < length < 3.0:
        #     print length
        if max_offset > length > min_offset:
            vectors_pass.append(vector)
            # print vector
    return vectors_pass


def draw_str(dst, target, s, front=(255, 255, 255), back=(0, 0, 0)):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, back, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, front, lineType=cv2.LINE_AA)


def main():
    print __doc__
    cap = cv2.VideoCapture(0)
    f_cascade = cv2.CascadeClassifier("C:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml")
    e_cascade = cv2.CascadeClassifier("C:\\opencv\\opencv\\build\etc\\haarcascades\\haarcascade_eye.xml")
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_count = 0  # 显示帧计时
    while True:
        if cv2.waitKey(1) == 27:  # Esc for exit
            break
        t = clock()
        if frame_count > 0:
            frame_count -= 1
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape
        rectangles = detect(gray, f_cascade)
        # draw_rectangles(img, rectangles, color=(255, 225, 0))  # 人脸范围
        # draw_rectangles(img, rectangles, scaling=1.3)  # 扩大后的范围

        for rectangle in rectangles:
            rx0, ry0, rx1, ry1 = rectangle
            draw_rectangle(img, rectangle, color=(255, 225, 0))  # 人脸范围
            flow = opt_flow(prev_gray, gray)  # get opt flow
            # lines = draw_flow(img[y0:y1, x0:x1], flow, False)
            lines = draw_flow(img, rectangle, flow)  # 显示光流点
            # if frame_count <= 0 and anti_spoofing2(lines) and position_check_eye(lines, rectangles_eye):
            #     frame_count = 30
            # if frame_count > 0:
            #     print 'yesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyesyes'
            #     draw_str(img, (20, 50), "Pass", (0, 69, 255), (255, 255, 255))  # print success
            #     # draw_str(img, (rx0 - 15, ry0 - 15), "Pass", (0, 69, 255), (255, 255, 255))  # print success
        prev_gray = gray
        dt = clock() - t
        draw_str(img, (20, 20), 'time: %.1f ms' % (dt * 1000))
        cv2.imshow("Face detect", img)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
