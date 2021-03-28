import cv2 as cv
import numpy as np


class AddMinusTest:
    def __init__(self, line, min, max, page):
        self.line = line
        self.min = min
        self.max = max
        self.page = page
        self.add0 = list()
        self.minus0 = list()
        self.add = list()
        self.minus = list()
        self.canvas = 255 * np.ones((600, 400))

    def run(self):
        self._generate()
        self._select()
        self._paint()
        self._show()

    def save(self, name):
        cv.imwrite(name, self.canvas)

    def _generate(self):
        for a in range(self.min, self.max):
            for b in range(self.min, self.max):
                if a + b <= self.max:
                    self.add0.append(str(a) + ' + ' + str(b) + ' =')
                if a - b >= 1:
                #if a - b < 0:
                    self.minus0.append(str(a) + ' - ' + str(b) + ' =')

    def _select(self):
        import random
        random.shuffle(self.add0)
        random.shuffle(self.minus0)
        self.add = self.add0[:self.line:]
        self.minus = self.minus0[:self.line:]

    def _paint(self):
        for i in range(self.line):
            cv.putText(self.canvas, str(i + 1) + '.  ' + self.add[i], (20, 30 + 40 * i), cv.FONT_HERSHEY_COMPLEX, 0.8,
                       (0, 0, 0),
                       2)
            cv.putText(self.canvas, self.minus[i], (240, 30 + 40 * i), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)

    def _show(self):
        cv.namedWindow('practise')
        cv.imshow('practise', self.canvas)
        cv.waitKey(0)


if __name__ == '__main__':
    test = AddMinusTest(15, 1, 10, 1)
    test.run()
    test.save('practise.jpg')
