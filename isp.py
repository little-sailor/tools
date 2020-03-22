import struct

import cv2 as cv
import numpy as np


class RawInfo():
    def __init__(self, path, file_name, suffix, w, h, bitWidth=12, ob=0, bayerFormat='rggb'):
        self.path = path
        self.suffix = suffix
        self.fileName = file_name
        self.w = w
        self.h = h
        self.bitWidth = bitWidth
        self.ob = ob
        self.bayerFormat = bayerFormat


class ISPParameter():
    def __init__(self, bayerNRStrength, gammaRaw, sharpenStrength, gammaRGB, saturation):
        self.bayerNRStrength = bayerNRStrength
        self.gammaRaw = gammaRaw
        self.sharpenStrength = sharpenStrength
        self.gammaRGB = gammaRGB
        self.saturation = saturation


class Region():
    def __init__(self, x0=0, y0=0, x1=0, y1=0):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1


def on_mouse(event, x, y, flags, region):
    if event == cv.EVENT_LBUTTONDOWN:
        region.x0, region.y0 = x, y
        print(x, y)
    elif event == cv.EVENT_LBUTTONUP:
        region.x1, region.y1 = x, y
        print(x, y)


class ISP():
    def __init__(self, info, param):
        self.raw_info = info
        self.parameters = param

    def _blc(self, raw):
        new = raw - (self.raw_info.ob / (1 << self.raw_info.bitWidth));
        new[new < 0] = 0
        return new

    def _wb(self, raw):
        region = self._getRegion('wbRegion', raw)
        wbRaw = raw[region.y0:region.y1:, region.x0:region.x1:]

        format = self.raw_info.bayerFormat

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

        self.r0 = r0
        self.b0 = b0
        self.gr0 = gr0
        self.gb0 = gb0

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

    def _bayerNR(self, raw):
        ksize = self.parameters.bayerNRStrength
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        kernel = cv.getGaussianKernel(ksize, sigma)
        raw[::2, ::2] = cv.filter2D(raw[::2, ::2], -1, kernel)
        raw[::2, 1::2] = cv.filter2D(raw[::2, 1::2], -1, kernel)
        raw[1::2, ::2] = cv.filter2D(raw[1::2, ::2], -1, kernel)
        raw[1::2, 1::2] = cv.filter2D(raw[1::2, 1::2], -1, kernel)
        return raw

    def _sharpenUSM(self, img):
        w = self.parameters.sharpenStrength
        blur_img = cv.GaussianBlur(img[:, :, 0], (0, 0), 5)
        usm = cv.addWeighted(img[:, :, 0], 1 + w, blur_img, -w, 0)
        img[:, :, 0] = usm
        return img

    def _sharpenFilter(self, img, whiteEdge=0.1, blackEdge=-0.1):
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

    def _demosaic(self, raw):
        bayerFormat = {'rggb': cv.COLOR_BayerRG2RGB,
                       'bggr': cv.COLOR_BayerBG2RGB,
                       'grbg': cv.COLOR_BayerGR2RGB,
                       'gbrg': cv.COLOR_BayerGB2RGB
                       }
        rgb = cv.cvtColor(raw, bayerFormat[self.raw_info.bayerFormat])
        return rgb

    def _gamma(self, rgb):
        g = self.parameters.gammaRaw
        return np.power(rgb, 1 / g)

    def _gammaRGB(self, img):
        g = self.parameters.gammaRGB
        return np.clip((np.power(img / (1 << 16), 1 / g) * 256), 0, 255).astype('uint8')

    def _ccm(self, img):
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

    def _rgb2yuv(self, img):
        s = self.parameters.saturation
        yuv = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)

        yuv = yuv.astype('float32')

        yuv[:, :, 1] = np.clip((yuv[:, :, 1] - 128.0) / 255.0 * s, -0.5, 0.5)
        yuv[:, :, 2] = np.clip((yuv[:, :, 2] - 128.0) / 255.0 * s, -0.5, 0.5)

        yuv[:, :, 1] = np.clip(yuv[:, :, 1] * 255 + 128, 0, 255)
        yuv[:, :, 2] = np.clip(yuv[:, :, 2] * 255 + 128, 0, 255)

        yuv = yuv.astype('uint8')

        return yuv

    def _claheApply(self, img):
        clahe = cv.createCLAHE(2, (4, 4))
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        return img

    def _load_raw_image(self):
        with open(self.raw_info.path + self.raw_info.fileName + self.raw_info.suffix, "rb") as f:
            data = f.read()
            data = struct.unpack('<' + str(int(len(data) / 2)) + 'H', data)
        raw = np.array(data).reshape(self.raw_info.h, self.raw_info.w)
        print(np.max(raw), np.average(raw), np.min(raw))
        raw = raw / (1 << self.raw_info.bitWidth)
        return raw.astype('float32')

    def _getRegion(self, win_name, img):
        region = Region()
        cv.namedWindow(win_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.imshow(win_name, img)
        cv.setMouseCallback(win_name, on_mouse, region)
        cv.waitKey(0)
        cv.destroyWindow(win_name)

        if region.x0 == region.x1 or region.y0 == region.y1:
            return Region(0, 0, img.shape[1], img.shape[0])

        region.x0 = region.x0 >> 1 << 1
        region.y0 = region.y0 >> 1 << 1
        region.x1 = region.x1 >> 1 << 1
        region.y1 = region.y1 >> 1 << 1

        return region

    def process(self):
        img = self._load_raw_image()

        region = self._getRegion('CROP', img)

        img = img[region.y0:region.y1:, region.x0:region.x1:]

        # raw_dump(img, path + file + '.csv')

        img = self._blc(img)

        img = self._bayerNR(img)

        img = self._gamma(img)

        img = self._wb(img)

        # fix domain 16bit
        img = np.clip((img * 65536), 0, 65535).astype('uint16')
        img = self._demosaic(img)

        # ccm
        img = self._ccm(img)

        # gamma rgb
        img = self._gammaRGB(img)

        # rgb 2 yuv
        img = self._rgb2yuv(img)

        # sharpen
        img = self._sharpenUSM(img)
        # img = sharpenFilter(img, 0.8, -0.4)

        # csc
        img = self._claheApply(img)

        self.resultYUV = img

    def show_rgbgrb(self):
        win_r = 'r'
        win_gr = 'gr'
        win_gb = 'gb'
        win_b = 'b'

        cv.namedWindow(win_r, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.imshow(win_r, self.r0)

        cv.namedWindow(win_gr, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.imshow(win_gr, self.gr0)

        cv.namedWindow(win_gb, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.imshow(win_gb, self.gb0)

        cv.namedWindow(win_b, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.imshow(win_b, self.b0)

        cv.waitKey(0)
        cv.destroyWindow(win_r)
        cv.destroyWindow(win_gr)
        cv.destroyWindow(win_gb)
        cv.destroyWindow(win_b)

    def show_final(self):
        win_name = 'result'

        self.resultRGB = cv.cvtColor(self.resultYUV, cv.COLOR_YCrCb2RGB)

        cv.namedWindow(win_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.imshow(win_name, self.resultRGB)
        cv.waitKey(0)
        cv.destroyWindow(win_name)

    def save(self):
        cv.imwrite(self.raw_info.path + self.raw_info.fileName + '.jpg', self.resultRGB)


if __name__ == '__main__':
    raw1 = RawInfo(r'E:\Raw\\', 'HisiRAW_4096x2160_12bits_RGGB_Linear_Route1_20200316183021', r'.raw',
                   4096, 2160, 12,
                   259, 'rggb')
    param1 = ISPParameter(1, 1.2, 0.99, 1.2, 4)

    raw2 = RawInfo(r'E:\Raw\xgs8000\\', 'HisiRAW_4096x2160_12bits_RGGB_Linear_Route1_20200318132946_2000', r'.raw',
                   4096, 2160, 12,
                   259, 'rggb')
    param2 = ISPParameter(1, 1.2, 0.99, 1.2, 1)

    raw3 = RawInfo(r'E:\Raw\\', 'HisiRAW_1920x1080_12bits_RGGB_Linear_Route0_20190816223320', r'.raw',
                   1920, 1080, 12,
                   240, 'rggb')
    param3 = ISPParameter(1, 1.2, 0.99, 1.5, 1)

    isp = ISP(raw1, param1)

    isp.process()

    # isp.show_rgbgrb()

    isp.show_final()

    isp.save()
