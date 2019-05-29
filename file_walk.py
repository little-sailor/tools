
def getImage(path, end):
    import os
    #all = os.walk(path)
    filelist = []
    for filename in os.listdir(path):
        if filename.endswith(end):
            filelist.append(filename)
    return filelist


def getROI(inf_file):
    import re
    key = u'车脸位置'
    roi = []
    with open(inf_file, 'r', encoding='GB2312') as f:
        for line in f:
            if key in line:
                print(line)
                roi = [int(s) for s in re.findall(r'\d+', line)]
                break
    if len(roi) != 4:
        return 0, 0, 0, 0
    else:
        return roi[0], roi[1], roi[2], roi[3]


def cropImage(src_path, dst_path, filelist):
    import cv2 as cv
    import numpy as np
    for file in filelist:
        img = cv.imread(src_path + file)
        inf = file.replace('_1.jpg', '.inf')
        x1, y1, x2, y2 = getROI(src_path + inf)
        if x1 == 0 and y1 == 0 and x2 ==0 and y2 == 0:
            continue
        x = ((x1) >> 5) << 5
        y = ((y1) >> 5) << 5
        w = ((x2 - x1) >> 4) << 4
        h = ((y2 - y1) >> 4) << 4

        print(x, y, x + w, y + h)
        imgcrop = img[y:y+h:, x:x+w:, :]
        cv.imwrite(dst_path+file,imgcrop)

if __name__ == '__main__':
    path = r'E:\900W\2019052319'
    src_path = r'E:\900W\2019052319\\'
    suffix = r'.jpg'
    dst_path = path + r'\\crop\\'
    flist = getImage(src_path, suffix)

    cropImage(src_path, dst_path, flist)

