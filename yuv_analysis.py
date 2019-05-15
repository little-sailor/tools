from matplotlib.pyplot import figure, imshow, show
from matplotlib.image import imsave
import cv2 as cv

def load_yuv(file, w, h, type):
    import struct
    import numpy as np
    with open(file, "rb") as f:
        data = f.read()
        data = struct.unpack('<'+str(int(len(data)))+'B', data)

    if type == '444p':
        y = np.array(data[:w*h:]).reshape(h, w).astype(np.uint8)
        u = np.array(data[w*h:w*h*2:]).reshape(h, w).astype(np.uint8)
        v = np.array(data[2*w*h::]).reshape(h, w).astype(np.uint8)
        yuv = np.stack([y,u,v], axis=-1)
    elif type == '420sp':
        y = np.array(data[:w*h:]).reshape(h, w).astype(np.uint8)
        uv = np.array(data[w*h::]).reshape(int(h/2), w).astype(np.uint8)
        u = uv[::, ::2]
        v = uv[::, 1::2]
        u = np.kron(u, np.ones([2, 2])).astype(np.uint8)
        v = np.kron(v, np.ones([2, 2])).astype(np.uint8)
        yuv = np.stack([y,u,v], axis=-1)
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
    width = 1280
    height = 720
    path = r'D:\ti\ccsv5\ccs_base\dvt_3.3.0.201209281235\ImageAnalyzer\examples\DaVinciHD\\'
    file = 'davinciHD1280x720_420semiplanar'
    suffix = '.yuv'

    yuv = load_yuv(path + file + suffix, width ,height, '420sp')

    rgb = yuv2rgb(yuv)

    imsave(path+file+'.jpg', rgb)

    imshow(rgb)
    show()

