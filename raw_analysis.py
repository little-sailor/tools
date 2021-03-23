import numpy as np
import struct
import cv2 as cv

def load_u8c3_image(file, w, h, bit_width):
	with open(file, "rb") as f:
		data = f.read()
		data = struct.unpack('<'+str(int(len(data)))+'B', data)
	b = np.array(data[:w*h*3:3]).reshape(h, w)
	g = np.array(data[1:w*h*3:3]).reshape(h, w)
	r = np.array(data[2:w*h*3:3]).reshape(h, w)
	rgb = np.stack([r,g,b], axis=-1)
	rgb = rgb.astype('uint8')

	return rgb
	
	
def load_u8_image(file, w, h):
	with open(file, "rb") as f:
		data = f.read()
		data = struct.unpack('<'+str(int(len(data)))+'B', data)
	raw = np.array(data).reshape(h, w)
	raw = raw.astype('uint8')
	print(min(raw.ravel()), max(raw.ravel()))
	return raw
	
def load_s16_image(file, w, h, bit_width):
	with open(file, "rb") as f:
		data = f.read()
		data = struct.unpack('<'+str(int(len(data)/ 2))+'h', data)
	raw = np.array(data).reshape(h, w)
	raw = abs(raw)
	raw = raw / (1<<(bit_width - 1)) * 256
	raw = raw.astype('uint8')
	print(min(raw.ravel()), max(raw.ravel()))
	return raw

def load_raw_image(file, w, h, bit_width):
	with open(file, "rb") as f:
		data = f.read()
		data = struct.unpack('<'+str(int(len(data)/ 2))+'H', data)
	raw = np.array(data).reshape(h, w)
	print(max(raw.ravel()))
	raw = raw / (1<<bit_width)
	raw = raw * 256
	raw = raw.astype('uint8')
	#print(max(raw.ravel()))
	return raw
	
def wb(raw):
	b = raw[::2, ::2]
	gb = raw[::2, 1::2]
	gr = raw[1::2, ::2]
	r = raw[1::2, 1::2]
	
	figure(1)
	subplot(2,2,1)
	imshow(b, cmap= 'gray')
	subplot(2,2,2)
	imshow(gb, cmap= 'gray')
	subplot(2,2,3)
	imshow(gr, cmap= 'gray')
	subplot(2,2,4)
	imshow(r, cmap= 'gray')
	
	r2g = sum(r.ravel()) / ((sum(gr.ravel()) + sum(gb.ravel())) / 2)
	b2g = sum(b.ravel()) / ((sum(gr.ravel()) + sum(gb.ravel())) / 2)

	r1 = r * 1 / r2g
	b1 = b * 1 / b2g

	wb = raw
	wb[::2, ::2] = b1
	wb[1::2, 1::2] = r1
	return wb


def demosaic(raw):
	rgb = cv.cvtColor(raw, cv.COLOR_BayerGR2RGB)
	return rgb


def unpackraw(raw):
	width = raw.shape[1]
	height = raw.shape[0]
	u = 0
	r = 0
	unpacked = np.zeros(raw.shape)
	for h in range(height):
		u = 0
		r = 0
		while r + 5 < width:

			unpacked[h][u + 0] = raw[h][r + 0] * 4 + (raw[h][r + 4] & 3)
			unpacked[h][u + 1] = raw[h][r + 1] * 4 + (raw[h][r + 4] & 12) / 4
			unpacked[h][u + 2] = raw[h][r + 2] * 4 + (raw[h][r + 4] & 48) / 16
			unpacked[h][u + 3] = raw[h][r + 3] * 4 + (raw[h][r + 4] & 192) / 64

			'''
			unpacked[h][u + 0] = raw[h][r + 0] + (raw[h][r + 4] & 3) * 256
			unpacked[h][u + 1] = raw[h][r + 1] + (raw[h][r + 4] & 12) * 256 / 4
			unpacked[h][u + 2] = raw[h][r + 2] + (raw[h][r + 4] & 48) * 256 / 16
			unpacked[h][u + 3] = raw[h][r + 3] + (raw[h][r + 4] & 192) * 256 / 64
			'''
			'''
			unpacked[h][u + 0] = raw[h][r + 0]
			unpacked[h][u + 1] = raw[h][r + 1]
			unpacked[h][u + 2] = raw[h][r + 2]
			unpacked[h][u + 3] = raw[h][r + 3]
			'''

			u = u + 4
			r = r + 5
	print(u, r)
	unpacked = unpacked / 1024 * 255
	unpacked = unpacked.astype('uint8')
	return unpacked


if __name__ == '__main__' :
	w = 2448
	h = 1096
	bitwitdh = 10
	path = r'C:\Users\lujy\Desktop\\'
	file = 'sif'
	suffix = '.raw'

	raw = load_u8_image(path + file + suffix, w, h)
	unpacked_raw = unpackraw(raw)

	cv.namedWindow("raw", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
	cv.imshow("raw", unpacked_raw)
	cv.waitKey(0)
