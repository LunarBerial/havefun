#coding:utf-8
'''
功能描述：将图像拆解，并用字母拼出原图像（保留原色彩）
'''
import os, random
import cv2
import numpy as np

# 深色部分用密度高的字母，浅色部分用密度低的。
L_list = ['2.jpg', '0.jpg', '9.jpg'] # '7.jpg', '17.jpg', '21.jpg',
M_list = ['22.jpg', '24.jpg'] # '14.jpg', '23.jpg','3.jpg', '15.jpg',
S_list = ['4.jpg', '29.png' ] # '5.jpg',  '8.jpg', '12.jpg', '28.jpg', '25.jpg',
XS_list = ['20.jpg', '11.jpg'] #'16.jpg', '19.jpg', '26.jpg', '27.jpg',
blank_list = ['1.jpg']


'''
函数功能：将目标色块赋予字母内容，及原图的色彩
'''
def find_cell(data, i, a_list, newdata, r, c):
    temp = cv2.imread('resource/' + a_list[i])
    temp_0 = temp[:, :, 0]
    temp_1= temp[:, :, 1]
    temp_2 = temp[:, :, 2]
    temp_0[temp_0 < 200] = data[20 * r, 20 * c, 0] # 此处取左上角的颜色做基准
    temp_1[temp_1 < 200] = data[20 * r, 20 * c, 1] # 浅拷贝，temp值随之改变
    temp_2[temp_2 < 200] = data[20 * r, 20 * c, 2]
    # print(r, c)
    if r == 17 and c == 20:
        print(r)
    newdata[20 * r: 20 * (r + 1), 20 * c: 20 * (c + 1), :] = temp

    return newdata


def doge_jpg():
    data = cv2.imread('bang.jpg')
    data = cv2.resize(data, (0, 0), fx = 0.5, fy=0.5) # 字母的分辨率为20*20，如果图片本身太小或太大，需要resize。< 1缩小， >1 放大
    print(data.shape)
    grey_data = data[:,:,0] * 0.2989 + data[:, :, 0] * 0.5870 + data[:,:, 2] * 0.114 # 取灰度，用于衡量深浅
    # print(gray_data)
    # cv2.imwrite('grey.jpg', grey_data)
    newdata = np.ones(data.shape) * 245 #新建底色为白的原尺寸数组
    for r in range(int(grey_data.shape[0]/20)):
        for c in range(int(grey_data.shape[1]/20)):
            s = np.sum(grey_data[20* r: 20 *(r + 1), 20 * c: 20 *(c+1)])/ (20 * 20) # 计算灰度均值
            # 分割为5个部分随机填入相应密度的字母
            if s < 60:
                i = random.randint(0, len(L_list) -1 )
                newdata = find_cell(data, i, L_list, newdata, r, c)
            elif s < 120:
                i = random.randint(0, len(M_list) - 1)
                newdata = find_cell(data, i, M_list, newdata, r, c)
            elif s < 180:
                i = random.randint(0, len(S_list) - 1)
                newdata = find_cell(data, i, S_list, newdata, r, c)
            elif s < 240:
                i = random.randint(0, len(XS_list) - 1)
                newdata = find_cell(data, i, XS_list, newdata, r, c)
            else:
                newdata[20 * r: 20 * (r + 1), 20 * c: 20 * (c + 1), :] = cv2.imread('resource/' + blank_list[0])
    # newdata = cv2.resize(newdata, (0, 0), fx = 1/2, fy = 1/2) # 如前面resize过，此处要重新缩放。
    cv2.imwrite('bang_s_a.jpg', newdata)


# 从序列图片中分割每个字母
def split_alpha():
    data = cv2.imread('ALPHA.png')
    blank = np.ones((17, 2, 3)) * 245 # 原序列高度为17，设置空白宽度为3
    s = 0
    e = 0
    idx = 0
    rubost = 0
    for i in range(data.shape[1] - 2):
        a = blank == data[:,i: i + 2, :]
        if False in a:
            e = i
        elif rubost > 1:
            cv2.imwrite('resource/'+str(idx) + '.jpg', data[:, s: e, :])
            idx += 1
            s = i
            rubost = 0
        else:
            rubost += 1
            e = i


# 将字母图片统一resize为20 * 20，不够的部分用白色填充
def resizejpg():
    for j in os.listdir('resource'):
        data = cv2.imread('resource/'+j)
        newdata = np.ones((20,20,3)) * 245
        newdata[:17, : data.shape[1], :] = data[:17,:,:]
        cv2.imwrite('resource/'+j, newdata)
        print(data.shape)

# 根据黑色占比，计算字母密度
def countweight():
    p_dict = {}
    for j in os.listdir('resource'):
        data = cv2.imread('resource/'+j)
        data[data > 1] = 1
        p_dict[j] = np.sum(data[:,:, 0])
    print(sorted(p_dict, key = lambda x: p_dict[x]))
    print(p_dict)
if __name__ == "__main__":
    # split_alpha()
    # resizejpg()
    # countweight()
    doge_jpg()