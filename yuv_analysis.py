from matplotlib.pyplot import figure, imshow, show
from matplotlib.image import imsave
import cv2 as cv
import numpy as np


def load_yuv(file, w, h, type):
    import struct

    with open(file, "rb") as f:
        data = f.read()
        data = struct.unpack('<'+str(int(len(data)))+'B', data)
    print(min(data), max(data))
    if type == '444p':
        y = np.array(data[:w*h:]).reshape(h, w).astype(np.uint8)
        u = np.array(data[w*h:w*h*2:]).reshape(h, w).astype(np.uint8)
        v = np.array(data[2*w*h::]).reshape(h, w).astype(np.uint8)
        yuv = np.stack([y,u,v], axis=-1)
    elif type == 'nv12':
        y = np.array(data[:w*h:]).reshape(h, w).astype(np.uint8)
        uv = np.array(data[w*h::]).reshape(int(h/2), w).astype(np.uint8)
        u = uv[::, ::2]
        v = uv[::, 1::2]
        u = np.kron(u, np.ones([2, 2])).astype(np.uint8)
        v = np.kron(v, np.ones([2, 2])).astype(np.uint8)
        yuv = np.stack([y,u,v], axis=-1)
    elif type == 'yv12':
        y = np.array(data[:w*h:]).reshape(h, w).astype(np.uint8)
        v = np.array(data[w*h:int(w*h+w*h/4):]).reshape(int(h/2), int(w/2)).astype(np.uint8)
        u = np.array(data[int(w*h+w*h/4)::]).reshape(int(h/2), int(w/2)).astype(np.uint8)
        v = np.kron(v, np.ones([2, 2])).astype(np.uint8)
        u = np.kron(u, np.ones([2, 2])).astype(np.uint8)
        yuv = np.stack([y,v,u], axis=-1)	
    return yuv


def yuv2rgb(yuv):
    rgb = cv.cvtColor(yuv, cv.COLOR_YUV2RGB)
    return rgb


def save_yuv(filename, data):
    with open(filename, 'wb') as f:
        f.write(data[:, :, 0].ravel())
        f.write(data[:, :, 1].ravel())
        f.write(data[:, :, 2].ravel())


if __name__ == '__main__':
    width = 2592
    height = 1944
    path = r'C:\Users\lujy\Desktop\\'
    file = '0'
    suffix = '.yuv'

    yuv = load_yuv(path + file + suffix, width ,height, 'nv12')

    cv.namedWindow('y', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.imshow('y', yuv2rgb(yuv))

    cv.waitKey(0)
