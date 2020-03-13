#from matplotlib.pyplot import figure, imshow, show, subplot, imsave
import numpy as np
import struct
import cv2 as cv


def load_raw_image(file, w, h, bit_width):
    with open(file, "rb") as f:
        data = f.read()
        data = struct.unpack('<'+str(int(len(data)/ 2))+'H', data)
    raw = np.array(data).reshape(h, w)
    print(max(raw.ravel()))
    raw = raw / (1 << bit_width)
    return raw.astype('float32')


def raw_dump(raw, file):
    import pandas as pd
    df = pd.DataFrame(raw)
    #df.to_csv(file)


def wb(raw, format):
    if format == 'rggb':
        r = raw[::2, ::2]
        gr = raw[::2, 1::2]
        gb = raw[1::2, ::2]
        b = raw[1::2, 1::2]
    elif format == 'bggr':
        b = raw[::2, ::2]
        gb = raw[::2, 1::2]
        gr = raw[1::2, ::2]
        r = raw[1::2, 1::2]
    elif format == 'grbg':
        gr = raw[::2, ::2]
        r = raw[::2, 1::2]
        b = raw[1::2, ::2]
        gb = raw[1::2, 1::2]
    elif format == 'gbrg':
        gb = raw[::2, ::2]
        b = raw[::2, 1::2]
        r = raw[1::2, ::2]
        gr = raw[1::2, 1::2]

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

    r1 = r * 1 / r2g
    b1 = b * 1 / b2g

    r1[r1 > 1] = 1
    b1[b1 > 1] = 1

    wb = raw
    if format == 'rggb':
        raw[::2, ::2] = r1
        raw[1::2, 1::2] = b1
    elif format == 'bggr':
        raw[::2, ::2] = b1
        raw[1::2, 1::2] = r1
    elif format == 'grbg':
        raw[::2, 1::2] = r1
        raw[1::2, ::2] = b1
    elif format == 'gbrg':
        raw[::2, 1::2] = b1
        raw[1::2, ::2] = r1

    return wb


def bayerNR(raw, ksize, sigmaX, sigmaY):
    # raw[::2, ::2] = cv.GaussianBlur(raw[::2, ::2], ksize, sigmaX, sigmaY)
    # raw[::2, 1::2] = cv.GaussianBlur(raw[::2, 1::2], ksize, sigmaX, sigmaY)
    # raw[1::2, ::2] = cv.GaussianBlur(raw[1::2, ::2], ksize, sigmaX, sigmaY)
    # raw[1::2, 1::2] = cv.GaussianBlur(raw[1::2, 1::2], ksize, sigmaX, sigmaY)

    raw[::2, ::2] = cv.bilateralFilter(raw[::2, ::2],  ksize, sigmaX, sigmaY)
    raw[::2, 1::2] = cv.bilateralFilter(raw[::2, 1::2],  ksize, sigmaX, sigmaY)
    raw[1::2, ::2] = cv.bilateralFilter(raw[1::2, ::2],  ksize, sigmaX, sigmaY)
    raw[1::2, 1::2] = cv.bilateralFilter(raw[1::2, 1::2], ksize, sigmaX, sigmaY)
    return raw

def sharpen(img, k=1):
    blur_img = cv.GaussianBlur(img[:,:,0], (0, 0), 5)
    usm = cv.addWeighted(img[:,:,0], 1.5, blur_img, -0.5, 0)
    img[:,:,0] = usm
    return img
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # y = cv.filter2D(img[:,:,0], -1,kernel)
    # s = img
    # s[:,:,0] = y
    # return s


def demosaic(raw, format):
    bayerFormat = {'rggb':cv.COLOR_BayerRG2RGB,
                   'bggr':cv.COLOR_BayerBG2RGB,
                   'grbg':cv.COLOR_BayerGR2RGB,
                   'gbrg':cv.COLOR_BayerGB2RGB
                   }
    rgb = cv.cvtColor(raw, bayerFormat[format])
    return rgb


def gamma(rgb, g):
    return np.power(rgb, 1/g)


def gammaRGB(img, bitw, g):
    return (np.power(img / (1 << 16),  1/g) * 255).astype('uint8')


def rgb2yuv(img):
    return cv.cvtColor(img, cv.COLOR_RGB2YUV)


def blc(raw, ob):
    new = raw - ob;
    new[new < 0] = 0
    return new


if __name__ == '__main__' :
    w = 1920
    h = 1080
    bitwitdh = 12
    path = r'e:\Raw\236\\'
    file = 'asamplesensor_Date_26-03-2018_Time_14-08-16-2700'
    bayerFormat = 'rggb'
    ob = 0

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
    # raw_dump(img, path + file + '.csv')

    img = blc(img, ob / (1 << bitwitdh))

    img = bayerNR(img, 2, 75, 75)

    img = gamma(img, 1.6)

    img = wb(img, bayerFormat)



    # fix domain 16bit
    img = (img * 65535).astype('uint16')
    img = demosaic(img, bayerFormat)

    # ccm

    # gamma rgb
    img = gammaRGB(img, bitwitdh, 3.5)

    # rgb 2 yuv
    img = rgb2yuv(img)

    # sharpen
    img = sharpen(img)

    # csc


    # show and save
    imgf = cv.cvtColor(img, cv.COLOR_YUV2RGB)

    cv.namedWindow('img', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.imshow('img', imgf)

    cv.waitKey()

    # cv.imwrite(path + file + '.jpg', imgf)
