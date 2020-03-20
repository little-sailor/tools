# from matplotlib.pyplot import figure, imshow, show, subplot, imsave
import struct

import cv2 as cv
import numpy as np


def load_raw_image(file, w, h, bit_width):
    with open(file, "rb") as f:
        data = f.read()
        data = struct.unpack('<' + str(int(len(data) / 2)) + 'H', data)
    raw = np.array(data).reshape(h, w)
    print(np.max(raw), np.average(raw), np.min(raw))
    raw = raw / (1 << bit_width)
    return raw.astype('float32')


def raw_dump(raw, file):
    import pandas as pd
    df = pd.DataFrame(raw)
    df.to_csv(file)

gx0 = 0
gy0 = 0
gx1 = 0
gy1 = 0

def onMouse(event, x, y, flags, param):
    global gx0, gy0, gx1, gy1
    if event == cv.EVENT_LBUTTONDOWN:
        gx0, gy0 = x, y
        print(x, y)
    elif event == cv.EVENT_LBUTTONUP:
        gx1, gy1 = x, y
        print(x, y)
    # elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_LBUTTONDOWN):
    #     img = param.copy()
    #     cv.rectangle(img, (x0, y0), (x, y), (255, 0, 0))
    #     cv.imshow('raw', img)

def getRegion(name, img):
    cv.namedWindow(name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.imshow(name, img)
    cv.setMouseCallback(name, onMouse, img)
    cv.waitKey(0)
    cv.destroyWindow(name)
    global gx0, gy0, gx1, gy1

    if gx0 == gx1 or gy0 == gy1:
        return 0, 0, img.shape[1], img.shape[0]

    x0 = gx0 >> 1 << 1
    y0 = gy0 >> 1 << 1
    x1 = gx1 >> 1 << 1
    y1 = gy1 >> 1 << 1

    gx0 = gx1 = gy0 = gy1 = 0

    return x0, y0, x1, y1


def wb(raw, format):
    x0, y0, x1, y1 = getRegion('wbRegion', raw)
    wbRaw = raw[y0:y1:, x0:x1:]

    if format == 'rggb':
        r = wbRaw[::2, ::2]
        gr = wbRaw[::2, 1::2]
        gb = wbRaw[1::2, ::2]
        b = wbRaw[1::2, 1::2]

        r0 = raw[::2, ::2]
        gr0 = raw[::2, 1::2]
        gb0 = raw[1::2, ::2]
        b0 = raw[1::2, 1::2]

    elif format == 'bggr':
        b = wbRaw[::2, ::2]
        gb = wbRaw[::2, 1::2]
        gr = wbRaw[1::2, ::2]
        r = wbRaw[1::2, 1::2]

        b0 = raw[::2, ::2]
        gb0 = raw[::2, 1::2]
        gr0 = raw[1::2, ::2]
        r0 = raw[1::2, 1::2]
    elif format == 'grbg':
        gr = wbRaw[::2, ::2]
        r = wbRaw[::2, 1::2]
        b = wbRaw[1::2, ::2]
        gb = wbRaw[1::2, 1::2]

        gr0 = raw[::2, ::2]
        r0 = raw[::2, 1::2]
        b0 = raw[1::2, ::2]
        gb0 = raw[1::2, 1::2]
    elif format == 'gbrg':
        gb = wbRaw[::2, ::2]
        b = wbRaw[::2, 1::2]
        r = wbRaw[1::2, ::2]
        gr = wbRaw[1::2, 1::2]

        gb0 = raw[::2, ::2]
        b0 = raw[::2, 1::2]
        r0 = raw[1::2, ::2]
        gr0 = raw[1::2, 1::2]

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

    print('r2g:', 1 / r2g)
    print('b2g:', 1 / b2g)

    r0 = r0 * 1 / r2g
    b0 = b0 * 1 / b2g

    r0[r0 > 1] = 1
    b0[b0 > 1] = 1

    if format == 'rggb':
        raw[::2, ::2] = r0
        raw[1::2, 1::2] = b0
    elif format == 'bggr':
        raw[::2, ::2] = b0
        raw[1::2, 1::2] = r0
    elif format == 'grbg':
        raw[::2, 1::2] = r0
        raw[1::2, ::2] = b0
    elif format == 'gbrg':
        raw[::2, 1::2] = b0
        raw[1::2, ::2] = r0

    return raw


def bayerNR(raw, ksize):
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    kernel = cv.getGaussianKernel(ksize, sigma)
    raw[::2, ::2] = cv.filter2D(raw[::2, ::2], -1, kernel)
    raw[::2, 1::2] = cv.filter2D(raw[::2, 1::2], -1, kernel)
    raw[1::2, ::2] = cv.filter2D(raw[1::2, ::2], -1, kernel)
    raw[1::2, 1::2] = cv.filter2D(raw[1::2, 1::2], -1, kernel)
    return raw


# def bayerNR(raw, ksize, sigmaX, sigmaY):
#     # raw[::2, ::2] = cv.GaussianBlur(raw[::2, ::2], ksize, sigmaX, sigmaY)
#     # raw[::2, 1::2] = cv.GaussianBlur(raw[::2, 1::2], ksize, sigmaX, sigmaY)
#     # raw[1::2, ::2] = cv.GaussianBlur(raw[1::2, ::2], ksize, sigmaX, sigmaY)
#     # raw[1::2, 1::2] = cv.GaussianBlur(raw[1::2, 1::2], ksize, sigmaX, sigmaY)
#
#     raw[::2, ::2] = cv.bilateralFilter(raw[::2, ::2],  ksize, sigmaX, sigmaY)
#     raw[::2, 1::2] = cv.bilateralFilter(raw[::2, 1::2],  ksize, sigmaX, sigmaY)
#     raw[1::2, ::2] = cv.bilateralFilter(raw[1::2, ::2],  ksize, sigmaX, sigmaY)
#     raw[1::2, 1::2] = cv.bilateralFilter(raw[1::2, 1::2], ksize, sigmaX, sigmaY)
#     return raw

def sharpenUSM(img, w=0.6):
    blur_img = cv.GaussianBlur(img[:, :, 0], (0, 0), 5)
    usm = cv.addWeighted(img[:, :, 0], 1 + w, blur_img, -w, 0)
    img[:, :, 0] = usm
    return img


def sharpenFilter(img, whiteEdge=0.1, blackEdge=-0.1):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    y0 = img[:, :, 0]

    edge = cv.filter2D(y0, cv.CV_16S, kernel)

    y0[y0 == 0] = 1
    edge = edge / y0

    # th0 = 0.8
    edge[edge > whiteEdge] = whiteEdge
    edge[edge < blackEdge] = blackEdge
    # edge[np.abs(edge) < th0] = 0

    enhance = y0 + y0 * edge

    enhance = np.clip(enhance, 0, 255).astype('uint8')

    img[:, :, 0] = enhance
    return img


def demosaic(raw, format):
    bayerFormat = {'rggb': cv.COLOR_BayerRG2RGB,
                   'bggr': cv.COLOR_BayerBG2RGB,
                   'grbg': cv.COLOR_BayerGR2RGB,
                   'gbrg': cv.COLOR_BayerGB2RGB
                   }
    rgb = cv.cvtColor(raw, bayerFormat[format])
    return rgb


def gamma(rgb, g):
    return np.power(rgb, 1 / g)


def gammaRGB(img, bitw, g):
    return np.clip((np.power(img / (1 << 16), 1 / g) * 256), 0, 255).astype('uint8')


def ccm(img):
    depth = img.shape[2]

    matrix = np.array([[1.790, -0.617, -0.173],
                       [-0.364, 1.397, -0.032],
                       [0.057, -0.815, 1.757]])
    rgb = np.zeros((depth, img.shape[0] * img.shape[1]))

    for i in range(depth):
        rgb[i, :] = img[:, :, i].ravel()

    rgb = matrix @ rgb

    rgb = np.clip(rgb, 0, 65535).astype('uint16')

    for i in range(depth):
        img[:, :, i] = rgb[i, :].reshape(img.shape[0: -1])

    return img


def rgb2yuv(img, s=1.2):
    yuv = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
    yuv[:, :, 1] = (yuv[:, :, 1] - 128) / 128 * s
    yuv[:, :, 2] = (yuv[:, :, 2] - 128) / 128 * s

    yuv[:, :, 1] = np.clip(yuv[:, :, 1], -1, 1).astype('float32')
    yuv[:, :, 2] = np.clip(yuv[:, :, 2], -1, 1).astype('float32')

    yuv[:, :, 1] = np.clip(yuv[:, :, 1] * 128 + 128, 0 , 255).astype('uint8')
    yuv[:, :, 2] = np.clip(yuv[:, :, 2] * 128 + 128, 0 , 255).astype('uint8')

    return yuv


def blc(raw, ob):
    new = raw - ob;
    new[new < 0] = 0
    return new


def claheApply(img):
    clahe = cv.createCLAHE(2, (4, 4))
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    return img

if __name__ == '__main__':
    w = 4096
    h = 2160
    bitwitdh = 12
    path = r'E:\Raw\\'
    file = 'HisiRAW_4096x2160_12bits_RGGB_Linear_Route1_20200316183021'
    bayerFormat = 'rggb'
    ob = 259

    # w = 4096
    # h = 2160
    # bitwitdh = 12
    # path = r'E:\Raw\xgs8000\\'
    # file = 'HisiRAW_4096x2160_12bits_RGGB_Linear_Route1_20200318132946_2000'
    # bayerFormat = 'rggb'
    # ob = 259

    # w = 1920
    # h = 1080
    # bitwitdh = 12
    # path = r'e:\Raw\236\\'
    # file = 'asamplesensor_Date_26-03-2018_Time_14-08-16-2700'
    # bayerFormat = 'rggb'
    # ob = 0

    # w = 2640
    # h = 1536
    # bitwitdh = 10
    # path = r'e:\Raw\\'
    # file = 'HisiRAW_2640x1536_10bits_BGGR_Linear_Route0_20200313110300'
    # bayerFormat = 'bggr'
    # ob =100

    # w = 1920
    # h = 1080
    # bitwitdh = 12
    # path = r'e:\Raw\\'
    # file = 'HisiRAW_1920x1080_12bits_RGGB_Linear_Route0_20190816223320'
    # bayerFormat = 'rggb'
    # ob = 240

    suffix = '.raw'

    # float domain
    img = load_raw_image(path + file + suffix, w, h, bitwitdh)

    x0, y0, x1, y1 = getRegion('CROP', img)
    img = img[y0:y1:, x0:x1:]
    # raw_dump(img, path + file + '.csv')

    img = blc(img, ob / (1 << bitwitdh))

    img = bayerNR(img, 1)

    img = gamma(img, 1.2)

    img = wb(img, bayerFormat)

    # fix domain 16bit
    img = np.clip((img * 65536), 0, 65535).astype('uint16')
    img = demosaic(img, bayerFormat)

    # ccm
    img = ccm(img)

    # gamma rgb
    img = gammaRGB(img, bitwitdh, 1.5)

    # rgb 2 yuv
    img = rgb2yuv(img, 1)

    # sharpen
    img = sharpenUSM(img, 0.99)
    # img = sharpenFilter(img, 0.8, -0.4)

    # csc
    claheApply(img)

    # show and save
    imgf = cv.cvtColor(img, cv.COLOR_YCrCb2RGB)

    cv.namedWindow('img', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.imshow('img', imgf)

    cv.waitKey()

    # cv.imwrite(path + file + '.jpg', imgf)
