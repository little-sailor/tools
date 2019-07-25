def getdivider(k):
	d = 0;
	for i in range(1, 10, 1):
		if pow(2, i) >= k:
			d = int(pow(2, i - 1))
			break;
	return d;
	
	
if __name__ == '__main__':
	from matplotlib.pyplot import figure, show, plot
	d = 12
	base = 1
	two = 2
	
	q = pow(2, base / d)
	l12 = [round(base * pow(q, n), 4) for n in range(d + 1)]
	
	five = 3/2
	four = 4/3
	l5 = [round(pow(five, n) / getdivider(pow(five, n)), 4) for n in range(1, 6, 1)]
	l5.append(base)
	l5.append(two)
	l5.append(four)
	l5.sort()
		
	l12_5 = [l12[0], l12[2], l12[4], l12[5], l12[7], l12[9], l12[11], l12[12]]
	
	derr = [abs(l5[i]- l12_5[i]) for i in range(len(l5))]

	figure(1)
	plot(l12_5)
	plot(l5, "r*")
	
	figure(2)
	plot(derr)
	
	show()
	
	