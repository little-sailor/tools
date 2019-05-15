from matplotlib.pyplot import figure, imshow, show, subplot, imsave
import numpy as np
import struct
import cv2 as cv


def load_raw_image(file, w, h, bit_width):
    with open(file, "rb") as f:
        data = f.read()
        data = struct.unpack('<'+str(int(len(data)/ 2))+'H', data)
    raw = np.array(data).reshape(h, w)
    print(max(raw.ravel()))
    raw = raw / (1<<bitwitdh) * 256
    raw = raw.astype('uint8')
    #print(max(raw.ravel()))
    return raw


def wb(raw):
    b = raw[::2, ::2]
    gb = raw[::2, 1::2]
    gr = raw[1::2, ::2]
    r = raw[1::2, 1::2]
    '''
    figure(1)
    subplot(2,2,1)
    imshow(b, cmap= 'gray')
    subplot(2,2,2)
    imshow(gb, cmap= 'gray')
    subplot(2,2,3)
    imshow(gr, cmap= 'gray')
    subplot(2,2,4)
    imshow(r, cmap= 'gray')
    '''
    r2g = sum(r.ravel()) / ((sum(gr.ravel()) + sum(gb.ravel())) / 2)
    b2g = sum(b.ravel()) / ((sum(gr.ravel()) + sum(gb.ravel())) / 2)

    r1 = r * 1 / r2g
    b1 = b * 1 / b2g

    wb = raw
    wb[::2, ::2] = b1
    wb[1::2, 1::2] = r1
    return wb


def demosaic(raw):
    rgb = cv.cvtColor(raw, cv.COLOR_BayerBG2RGB)
    return rgb


if __name__ == '__main__' :
    w = 2592
    h = 1944
    bitwitdh = 12
    path = r'C:\Users\lujy.HVTEAM\Desktop\\'
    file = 'HisiRAW_2592x1944_12bits_RGGB_Linear_Route1_20190515140843'
    suffix = '.raw'

    raw = load_raw_image(path + file + suffix, w, h, bitwitdh)
    #raw = wb(raw)
    rgb = demosaic(raw)

    imsave(path+file+'.jpg', rgb)

    figure(2)
    imshow(rgb)
    show()
