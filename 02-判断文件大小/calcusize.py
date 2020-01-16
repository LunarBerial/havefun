# coding: utf-8

import os, shutil


def calcusize(size, unit):
    size = size/1024
    if unit == 'k':
        return size
    size = size/1024
    if unit == 'M':
        return size
    size = size/1024
    if unit == 'G':
        return size

    return 0


def getfilesize(filename = '', unit = 'k'):
    if not os.path.exists(filename):
        return None
    size = os.path.getsize(filename)
    size = calcusize(size, unit)
    size = round(size,2)
    return size


if __name__ == "__main__":
    size = getfilesize('1.mp3', 'k')

    print(size)

