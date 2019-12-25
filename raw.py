from matplotlib.pyplot import figure, plot, show
from math import log10

iso = list(range(100,800))
raw = [56 - i for i in range(1,56)]

x = [[i,56] for i in iso]

y = [[800, i] for i in raw]

f = [-100*log10(i[1]/i[0]) for i in x+y]

figure(1)
plot(f)
show()
