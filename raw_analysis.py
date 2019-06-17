from matplotlib.pyplot import figure, imshow, show, subplot, imsave
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
	raw = raw / (1<<bit_width) * 256
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
	w = 1920
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
	
	raw = load_u8_image(path + 'src' + suffix, w, h, 8)
	figure(1)
	imshow(raw, cmap='gray')
	rgb = np.stack([raw,raw,raw], axis=-1)
	imsave(path+'src'+'.jpg', rgb)
	
	raw = load_s16_image(path + 'gh' + suffix, w, h, 16)
	figure(3)
	imshow(raw, cmap='gray')
	
	raw = load_s16_image(path + 'gv' + suffix, w, h, 16)
	figure(4)
	imshow(raw, cmap='gray')
	'''
	raw = load_s16_image(path + 'srcV' + suffix, w, h, 16)
	figure(5)
	imshow(raw, cmap='gray')
	
	raw = load_s16_image(path + 'dstV' + suffix, w, h, 16)
	figure(6)
	imshow(raw, cmap='gray')
	
	raw = load_u8_image(path + 'edge' + suffix, w, h, 8)
	figure(7)
	imshow(raw, cmap='gray')
	
	raw = load_u8c3_image(path + 'rgb' + suffix, w, h, 8)
	figure(8)
	imshow(raw)
	'''
	show()
