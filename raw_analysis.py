from matplotlib.pyplot import figure, imshow, show, subplot, imsave
import numpy as np
import struct
import cv2 as cv

def load_u8_image(file, w, h, bit_width):
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
	raw = raw / (1<<16) * 256
	raw = raw.astype('uint8')
	print(min(raw.ravel()), max(raw.ravel()))
	return raw

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
	w = 944
	h = 1024
	bitwitdh = 12
	path = r'Y:\nfs\\'
	file = 'src'
	suffix = '.yuv'

	#raw = load_raw_image(path + file + suffix, w, h, bitwitdh)
	#raw = wb(raw)
	#rgb = demosaic(raw)

	#imsave(path+file+'.jpg', rgb)
	#imshow(rgb)
	
	raw = load_u8_image(path + 'src' + suffix, w, h, bitwitdh)
	figure(1)
	imshow(raw, cmap='gray')
	
	raw = load_u8_image(path + 'dst' + suffix, w, h, bitwitdh)
	figure(2)
	imshow(raw, cmap='gray')
	
	raw = load_s16_image(path + 'srcH' + suffix, w, h, bitwitdh)
	figure(3)
	imshow(raw, cmap='gray')
	
	raw = load_s16_image(path + 'dstH' + suffix, w, h, bitwitdh)
	figure(4)
	imshow(raw, cmap='gray')
	
	raw = load_s16_image(path + 'srcV' + suffix, w, h, bitwitdh)
	figure(5)
	imshow(raw, cmap='gray')
	
	raw = load_s16_image(path + 'dstV' + suffix, w, h, bitwitdh)
	figure(6)
	imshow(raw, cmap='gray')
	
	show()
