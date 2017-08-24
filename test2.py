#!/usr/bin/env python
# coding=utf-8

import threading
import time


def f1():
    i = 10
    while i > 0:
        time.sleep(1)
        print i
        i -= 1

if __name__ == '__main__':
    t = threading.Thread(target=f1)
    t.start()
    t2 = threading.Thread(target=f1)
    t2.start()
    print 'finished'
