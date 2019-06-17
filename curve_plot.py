

def load_curvev2(file):
	import re
	lum = []
	lumI = []
	key = u'IspDev = [0]'
	with open(file, 'r', encoding='UTF-8') as f:
		for l in f:
			if key in l:
				lum.append(int(re.findall(r'\d+', re.search(r'AveLum = \[\d+\]', l).group(0))[0]))
				lumI.append(int(re.findall(r'\d+', re.search(r'AveLumI = \[\d+\]', l).group(0))[0]))
	return lum, lumI


def load_curve(file):
	x = []
	y = []
	with open(file) as f:
		for l in f:
			x.append(int(l.split(' ')[0]))
			y.append(int(l.split(' ')[1]))
	return x, y
	
	
if __name__ == '__main__' :
	from matplotlib.pyplot import figure, show, plot
	path = r'E:\log\坛洛日志\\'
	file = '172.18.18.242_061310'
	suffix = '.log'

	lum, lumI = load_curvev2(path + file + suffix)

	figure(1)
	plot(lum)
	plot(lumI, 'r')
	show()
