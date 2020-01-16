# -*- coding: utf8 -*-
import cv2
# 读取头像和国旗图案

def merge(input):
    img_head = cv2.imread(input)
    print(type(img_head))
    img_flag = cv2.imread('flag.png')
    # 获取头像和国旗图案宽度
    w_head, h_head = img_head.shape[:2]
    w_flag, h_flag = img_flag.shape[:2]
    # 计算图案缩放比例
    scale = w_head / w_flag / 4
    # 缩放图案
    img_flag = cv2.resize(img_flag, (0, 0), fx=scale, fy=scale)
    # 获取缩放后新宽度
    w_flag, h_flag = img_flag.shape[:2]
    # 按3个通道合并图片
    # Tips: numpy中直接“=”或切片是对数组的浅拷贝。引用copy() 函数后，才是创建新对象。
    new_head_0 = img_head.copy()
    new_head_1 = img_head.copy()
    new_head_2 = img_head.copy()
    new_head_3 = img_head.copy()
    for c in range(0, 3):
        new_head_0[w_head - w_flag:, h_head - h_flag:, c] = img_flag[:, :, c]
        new_head_1[:w_flag, :h_flag, c] = img_flag[:, :, c]
        new_head_2[:w_flag, h_head - h_flag:, c] = img_flag[:, :, c]
        new_head_3[w_head - w_flag:, :h_flag, c] = img_flag[:, :, c]
    # 保存最终结果
    cv2.imwrite('0'+input, new_head_0)
    cv2.imwrite('1' + input, new_head_1)
    cv2.imwrite('2' + input, new_head_2)
    cv2.imwrite('3' + input, new_head_3)


def mergewithnobackground():
    from PIL import Image
    mark = Image.open('flag.png')


if __name__=="__main__":
    import os
    # for i in os.listdir('head'):
    #     print(i)
    #     merge('head/'+i)
    merge('head.jpg')
    cv2.resize()