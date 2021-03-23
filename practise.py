import cv2 as cv
import numpy as np

if __name__ == '__main__':
    add = list()
    minus = list()
    for a in range(1, 10):
        for b in range(1, 10):
            if a + b <= 10:
                add.append(str(a)+' + '+str(b)+' =')
            if a - b >= 1:
                minus.append(str(a)+' - '+str(b)+' =')

    # print(len(add), add)
    # print(len(minus), minus)

    import random
    random.shuffle(add)
    random.shuffle(minus)

    n = 15
    add = add[:n:]
    minus = minus[:n:]
    print(add)
    print(minus)

    #canvas = cv.imread('canvas.jpg')
    canvas = 255 * np.ones((600, 400))
    for i in range(n):
        cv.putText(canvas, str(i + 1) + '.  ' + add[i], (20, 30 + 40 * i), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)
        cv.putText(canvas, minus[i], (240, 30 + 40 * i), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)

    cv.namedWindow('practise')
    cv.imshow('practise', canvas)
    cv.waitKey(0)
    cv.imwrite('practise.jpg', canvas)

