#!/usr/bin/env python
# coding=utf-8
"""
Face detect

Keys:
    ESC    - exit
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
    """
    选取八个区域进行规律查找
    :return:
    """
    print __doc__
    cap = cv2.VideoCapture(0)
    f_cascade = cv2.CascadeClassifier("C:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml")
    e_cascade = cv2.CascadeClassifier("C:\\opencv\\opencv\\build\etc\\haarcascades\\haarcascade_eye.xml")
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    track_len = 10
    detect_interval = 1
    tracks = []
    frame_idx = 0

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

    # test_areas = []  # 固定四个角
    # test_areas.append((0, 0, 128, 128))  # lu
    # test_areas.append((640 - 128, 0, 640, 128))  # ru
    # test_areas.append((640 - 128, 480 - 128, 640, 480))  # rd
    # test_areas.append((0, 480 - 128, 128, 480))  # ld

    while True:
        if cv2.waitKey(1) == 27:  # Esc for exit
            break
        t = clock()
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape
        rectangles = detect(gray, f_cascade)
        # draw_rectangles(img, rectangles, color=(255, 225, 0))  # 人脸范围
        # draw_rectangles(img, rectangles, scaling=1.3)  # 扩大后的范围

        if len(rectangles) == 1:  # 限制一张人脸
            for rectangle in rectangles:
                rx0, ry0, rx1, ry1 = rectangle
                # if rx1 - rx0 != 151 and ry1 - ry0 != 151:  # 限定人脸识别框的大小
                #     continue
                draw_rectangle(img, rectangle, color=(255, 225, 0))  # 人脸范围
                # draw_rectangle(img, rectangle, scaling=1.4)  # 扩大后的范围
                rectangles_eye = detect(gray[ry0:ry1, rx0:rx1], e_cascade)



                # if len(rectangles_eye) < 1:  # 限制必须两只眼睛都识别出来
                #     continue

                test_areas_face = []
                size = 32
                face_h = ry1 - ry0
                face_w = rx1 - rx0
                test_areas_face.append((0, 0, size, size))  # lu
                test_areas_face.append((face_w - size, 0, face_w, size))  # ru
                test_areas_face.append((face_w - size, face_h - size, face_w, face_h))  # rd
                test_areas_face.append((0, face_h - size, size, face_h))  # ld
                test_areas_face.append((face_w / 4, 0, face_w / 4 * 3, size))  # mu
                test_areas_face.append((face_w / 4, face_h - size, face_w / 4 * 3, face_h))  # md
                test_areas_face.append((0, face_h / 4, size, face_h / 4 * 3))  # ml
                test_areas_face.append((face_w - size, face_h / 4, face_w, face_h / 4 * 3))  # mr

                draw_rectangles(img[ry0:ry1, rx0:rx1], rectangles_eye, color=(255, 0, 225))

                draw_rectangles(img[ry0:ry1, rx0:rx1], test_areas_face, color=(25, 23, 225))

                eye_flow_lines = []  # 眼睛光流位移
                eye_flow_angs = []  # 眼睛光流角度
                for erx0, ery0, erx1, ery1 in rectangles_eye:
                    if len(tracks) > 0:
                        img0, img1 = prev_gray[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1], gray[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1]
                        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                        good = d < 1
                        new_tracks = []
                        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                            if not good_flag:
                                continue
                            tr.append((x, y))
                            if len(tr) > track_len:
                                del tr[0]
                            new_tracks.append(tr)
                            cv2.circle(img[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1], (x, y), 2, (0, 255, 0), -1)
                        tracks = new_tracks
                        cv2.polylines(img[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1], [np.int32(tr) for tr in tracks], False, (0, 255, 0))
                        draw_str(img, (200, 20), 'track count: %d' % len(tracks))

                    if frame_idx % detect_interval == 0:
                        mask = np.zeros_like(gray[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1])
                        mask[:] = 255
                        for x, y in [np.int32(tr[-1]) for tr in tracks]:
                            cv2.circle(mask, (x, y), 2, 0, -1)
                        p = cv2.goodFeaturesToTrack(gray[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1], mask=mask, **feature_params)
                        if p is not None:
                            for x, y in np.float32(p).reshape(-1, 2):
                                tracks.append([(x, y)])




                test_flow_lines = []  # 测试区域光流位移
                test_flow_angs = []  # 测试区域光流角度
                for erx0, ery0, erx1, ery1 in test_areas_face:
                    test_flow = opt_flow(prev_gray[ery0:ery1, erx0:erx1], gray[ery0:ery1, erx0:erx1])  # get opt flow
                    lines = draw_flow(img[ry0:ry1, rx0:rx1][ery0:ery1, erx0:erx1], test_flow, step=4)  # 显示光流点
                    test_flow_lines.append(lines)

                    # mag, ang = cv2.cartToPolar(eye_flow[..., 0], eye_flow[..., 1])
                    # test_flow_angs.append(ang)

                eye_sorted = []  # 排序后的长度集合(眼睛)
                for lines in eye_flow_lines:
                    mds = []
                    for (x1, y1), (x2, y2) in lines:
                        md = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        mds.append(md)
                    eye_sorted.append(sorted(mds, reverse=True))

                test_sorted = []  # 排序后的长度集合(test)
                for lines in test_flow_lines:
                    mds = []
                    for (x1, y1), (x2, y2) in lines:
                        md = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        mds.append(md)
                    test_sorted.append(sorted(mds, reverse=True))

                # ========打印前十个=========================
                if True:
                    for i, md2 in enumerate(eye_sorted):
                        count = 0
                        print '眼睛' + str(i + 1) + ':\t',
                        for md in md2:
                            count += 1
                            if count > 10:
                                break
                            print str(round(md, 2)) + ',',
                        print ''
                    print ''

                    for i, md2 in enumerate(test_sorted):
                        count = 0
                        print '测试' + str(i + 1) + ':\t',
                        for md in md2:
                            count += 1
                            if count > 10:
                                break
                            print str(round(md, 2)) + ',',
                        print ''
                    print ''

                if False:
                    # ============= 打印前十个平均值 ============
                    for i, md2 in enumerate(eye_sorted):
                        count = 0
                        print '眼睛' + str(i + 1) + ':\t',
                        sum_avg = []
                        for md in md2:
                            count += 1
                            if count > 10:
                                break
                            sum_avg.append(md)
                        print round(1.0 * sum(sum_avg) / len(sum_avg), 2)

                    for i, md2 in enumerate(test_sorted):
                        count = 0
                        print '测试' + str(i + 1) + ':\t',
                        sum_avg = []
                        for md in md2:
                            count += 1
                            if count > 10:
                                break
                            sum_avg.append(md)
                        print round(1.0 * sum(sum_avg) / len(sum_avg), 2)
                    print ''

        prev_gray = gray
        dt = clock() - t
        draw_str(img, (20, 20), 'time: %.1f ms' % (dt * 1000))
        cv2.imshow("Face detect", img)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
