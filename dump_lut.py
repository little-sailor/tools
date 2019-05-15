
def load_lut(file):
	import struct
	import numpy as np
	data = []
	with open(file, "rb") as f:
		data = f.read()
		data = struct.unpack('<'+str(int(len(data)/2))+'H', data)

	return data



def transform(s):
	import re
	l = len(s)
	s = ", ".join([str(i) for i in s])
	patten = re.compile(r'(([0-9]{1,4}, ){16,16})')  # {16,16} 不能有空格
	t = re.sub(patten, r'\1\n', s)
	ts = '#ifndef _LUT_H_\n' \
		 '#define LUT_H_\n' \
		 'int g_qlut['+str(l)+'] = {\n'
	te = '\n};\n' \
		 '#endif'
	t = ts + t + te
	return t


if __name__ == '__main__':
	from matplotlib.pyplot import plot, show
	path = r'\\'
	file = ''
	suffix = '.bin'

	lut = load_lut(path + file + suffix)
	l = int(len(lut) / 11)
	lut1 = lut[:l:]
	lut2 = lut[l+1 : 2*l :]
	lut3 = lut[2*l+1: 3*l :]
	lut4 = lut[3*l+1: 4*l:]
	lut5 = lut[4*l+1: 5*l:]
	lut6 = lut[5*l+1: 6*l:]
	lut7 = lut[6*l+1: 7*l:]
	lut8 = lut[7*l+1: 8*l:]
	lut9 = lut[8*l+1: 9*l:]
	lut10 = lut[9*l+1: 10*l:]
	lut11 = lut[10*l+1: 11*l:]
	plot(lut1)
	plot(lut2)
	plot(lut3)
	plot(lut4)
	plot(lut5)
	plot(lut6)
	plot(lut7)
	plot(lut8)
	plot(lut9)
	plot(lut10)
	plot(lut11)
	#plot(lut)
	show()	
	lut = transform(lut11)
	with open(r'E:\lut.h', 'w') as f:
		f.write(lut)
		
