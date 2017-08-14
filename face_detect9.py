#!/usr/bin/env python
# coding=utf-8
"""
Face detect

Keys:
    ESC    - exit
"""

from __future__ import print_function
import cv2
import numpy as np
import math
import time
from common import clock


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


def draw_rectangle(img, rectangle, color=(0, 255, 0), scaling=1.0):
    """draw rectangle. """
    x0, y0, x1, y1 = rectangle
    xd, yd = scale((x0, y0), (x1, y1), scaling)
    cv2.rectangle(img, xd, yd, color, 2)


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


def draw_str(dst, target, s, front=(255, 255, 255), back=(0, 0, 0)):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, back, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, front, lineType=cv2.LINE_AA)


def draw_rectangles(img, rectangles, color=(0, 255, 0), scaling=1.0):
    """draw rectangle. """
    for x0, y0, x1, y1 in rectangles:
        xd, yd = scale((x0, y0), (x1, y1), scaling)
        cv2.rectangle(img, xd, yd, color, 2)


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


def draw_flow(detected, flow, show=True, step=16, color=(0, 0, 255)):
    """draw as line with flow"""
    h, w = detected.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = detected
    if show:
        cv2.polylines(vis, lines, 0, color)  # draw lines
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)  # redraw the start point
    return lines


def main():
    """
    采用 伪眼部光流检测
    在脸部位置相对静止及不眨眼的情况下（眨眼时，眼睛部位的关键点会出现减少情况），
    眼球中虹膜或瞳孔的关键点的位移与脸部其他位置的关键不同，由此判断
    在眼部取变化最大的四个点，在眼部外取变化最大的四个点，分别取平均值，然后作比较，相差在指定标准外的则为活体
    :return:
    """
    print(__doc__)
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('D:/data.avi', fourcc, 30, (640, 480))
    f_cascade = cv2.CascadeClassifier("C:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml")
    e_cascade = cv2.CascadeClassifier("C:\\opencv\\opencv\\build\etc\\haarcascades\\haarcascade_eye.xml")
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

    tracks = []
    frame_index = 0
    detect_interval = 3
    track_len = 10
    msg_show = 0  # 通过信息显示帧数
    has_face = False

    # 存储每一帧的光流
    eye_flow_lines_t = []

    while True:
        if cv2.waitKey(1) == 27:  # Esc for exit
            break
        t = clock()
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rectangles = detect(gray, f_cascade)

        mask = np.zeros_like(gray)  # 设置关键点遮罩

        if len(rectangles) == 1:  # 限制一张人脸
            if not (has_face and True):
                tracks = []
            has_face = True
            for rectangle in rectangles:
                rx0, ry0, rx1, ry1 = rectangle
                # if not (140 < rx1 - rx0 < 160 and 140 < ry1 - ry0 < 160):  # 限定人脸识别框的大小
                #     continue
                draw_rectangle(img, rectangle, color=(0, 225, 0))  # 人脸范围
                # rectangles_eye = detect(gray[ry0:ry1, rx0:rx1], e_cascade)  # 获取眼睛范围
                # draw_rectangles(img[ry0:ry1, rx0:rx1], rectangles_eye, color=(255, 0, 225))

                # 眼部光流场功能
                eye_flow_lines = []
                # for erx0, ery0, erx1, ery1 in rectangles_eye:
                #     eye_flow = opt_flow(prev_gray[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1],
                #                         gray[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1])  # get opt flow
                #     eye_flow_lines.append(draw_flow(img[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1],
                #                                     eye_flow, step=4))  # 显示光流点

                # 假眼部位置，假设脸纵向上部1/4位置到2/4位置及 横向左部1/6到5/6位置为眼部，以抵消眼部识别不能每次都有效地问题

                face_h = ry1 - ry0
                face_w = rx1 - rx0
                face_hs = face_h / 4
                face_he = face_h / 2
                face_ws = face_w / 6
                face_we = face_w / 6 * 5
                eye_flow = opt_flow(prev_gray[ry0:ry1, rx0:rx1][face_hs:face_he, face_ws:face_we],
                                    gray[ry0:ry1, rx0:rx1][face_hs:face_he, face_ws:face_we])
                eye_flow_lines.append(
                    draw_flow(img[ry0:ry1, rx0:rx1][face_hs:face_he, face_ws:face_we], eye_flow, step=4))

                eye_sorted = []  # 排序后的长度集合(眼睛)
                eye_sorted2 = []
                for lines in eye_flow_lines:
                    mds = []
                    for (x1, y1), (x2, y2) in lines:
                        md = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        mds.append(md)
                        eye_sorted2.append(md)
                    eye_sorted.append(sorted(mds, reverse=True))
                    eye_flow_lines_t.append(eye_sorted2)  # 存储每一帧的光流位移信息
                # 绘制关键点轨迹
                # 会删除位移较大的关键点
                # 不影响其他的鉴别
                if len(tracks) > 0:
                    img0, img1 = prev_gray, gray
                    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                    good = d < 0.5
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        if not (rx0 < x < rx1 and ry0 < y < ry1):
                            continue
                        tr.append((x, y))
                        if len(tr) > track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                    tracks = new_tracks
                    cv2.polylines(img, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
                    draw_str(img, (20, 20), 'track count: %d' % len(tracks))

                # 限定人脸为兴趣区域
                cv2.fillPoly(mask, np.array([[[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]]]), (255, 255, 255))
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)  # 排除上一次的关键点

                if frame_index % detect_interval == 0:
                    # print('**************** start ***************')
                    l_sorted = []
                    l_sorted_eye = []  # 眼睛区域的关键点
                    l_sorted_out = []  # 眼睛外部的关键点

                    l_tmp = []
                    l_tmp_eye = []
                    l_tmp_out = []
                    for tr in tracks:
                        (x0, y0) = tr[0]
                        (x1, y1) = tr[-1]

                        if rx0 + face_ws < x1 < rx0 + face_we and ry0 + face_hs < y1 < ry1 + face_he:
                            l_tmp_eye.append(round(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2), 2))
                        else:
                            l_tmp_out.append(round(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2), 2))

                        # 计算各个关键点的角度和位移，以便观察
                        l = round(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2), 2)
                        l_tmp.append(l)
                        # if l > 0:
                        # print(round(math.atan(abs((y1 - y0) / (x1 - x0))) / math.pi * 180, 2), end=':')
                        # print(l, end='\t')
                    # print('\n+++++++++++++++')
                    l_sorted = sorted(l_tmp, reverse=True)
                    l_sorted_eye = sorted(l_tmp_eye, reverse=True)
                    l_sorted_out = sorted(l_tmp_out, reverse=True)
                    if len(l_sorted_eye) > 3 and len(l_sorted_out) > 3:
                        eye_avg = sum(l_sorted_eye[:4]) / 4
                        out_avg = sum(l_sorted_out[:4]) / 4
                        print('eye: ', l_sorted_eye[:4], eye_avg)
                        print('out: ', l_sorted_out[:4], out_avg)
                        if out_avg < 1 and eye_avg - out_avg > 2:
                            print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
                            print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
                            print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
                            print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
                            msg_show = 30

                    # # ========打印前十个=========================
                    # if True:
                    #     for i, md2 in enumerate(eye_sorted):
                    #         count = 0
                    #         print('眼睛', str(i + 1), end=':\t')
                    #         for md in md2:
                    #             count += 1
                    #             if count > 150:
                    #                 break
                    #             print(round(md, 2), end=',')
                    #         print()
                    #     print('###################')
                    #
                    # # 活体检测
                    # np_eye = np.array(sorted(eye_sorted2, reverse=True)[:30])
                    # np_eye = np_eye[np_eye > 0]
                    # np_l = np.array(l_sorted[:10])
                    #
                    # print('length: ', np_eye.size, '+++++', np_l.size)
                    # if np_eye.size != 0 and np_l.size != 0:
                    #     flow_per = np_eye[np_eye > 2].size * 1.0 / np_eye.size
                    #     ln_per = np_l[np_l > 2].size * 1.0 / np_l.size
                    #     print('percent: ', flow_per, '---', ln_per)
                    #     print(0.8, ' > ', flow_per, ' > ', 0.05, ' and ln_pre < ', 0.2)
                    #     if 0.8 > flow_per > 0.05 and ln_per < 0.2:
                    #         msg_show = 30
                    #         print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
                    #         print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
                    #         print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
                    #         print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')

                    # print('**************** end ***************')
                    # 判断关键点
                    p = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            tracks.append([(x, y)])

            frame_index += 1
        else:
            has_face = False

        prev_gray = gray
        dt = clock() - t
        # draw_str(img, (20, 20), 'time: %.1f ms' % (dt * 1000))
        if msg_show > 0:
            draw_str(img, (450, 20), 'YES', front=(0, 0, 255))
            msg_show -= 1
        cv2.imshow("Face detect", img)
        out.write(img)
        # cv2.imshow('mask', mask)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 添加满足添加的点的数量的判断 如 满足条件的点数量必须大于10
    main()
