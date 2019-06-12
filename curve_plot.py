
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
	path = r'Y:\my_work\\'
	file = 'o'
	suffix = '.txt'

	x, y = load_curve(path + file + suffix)
	
	figure(1)
	plot(x)
	plot(y[60::])
	show()
