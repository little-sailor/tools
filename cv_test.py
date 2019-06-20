def imgshow(img):
    from matplotlib.pyplot import figure, imshow
    if not hasattr(imgshow, 'cnt'):
        imgshow.cnt = 0
    imgshow.cnt += 1
    figure(imgshow.cnt)
    if len(img.shape) == 3:
        imshow(img[:, :, [2, 1, 0]])
    else:
        imshow(img, cmap='gray')


if __name__ == '__main__':
    import cv2 as cv
    from matplotlib.pyplot import show

    path = r'E:\900W\2019052317\crop\\'
    name = '20190523173813_00000115_1'

    # path = r'C:\Users\lujy.HVTEAM\Desktop\\'
    # name = 'test'

    file = path + name + '.jpg'

    #Load image
    img0 = cv.imread(file)
    img_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

    # edge
    #img_edge = cv.Canny(img_gray, 100, 300)
    img_edge = cv.Sobel(img_gray, cv.CV_8UC1, 1, 0)
    _, img_edge = cv.threshold(img_edge, 128, 255, cv.THRESH_BINARY)
    imgshow(img_edge)

    #dilate
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (50,3))
    img_edge_new = cv.morphologyEx(img_edge, cv.MORPH_CLOSE, kernel)
    imgshow(img_edge_new)

    # find contours
    _, contours, _ = cv.findContours(img_edge_new, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # draw contours
    cv.drawContours(img0, contours, -1, (0, 0, 255), 1)

    #draw rect
    for c in contours:
        rotated_rect = cv.minAreaRect(c)
        box = cv.boxPoints(rotated_rect)

        x, y = rotated_rect[0]
        w, h = rotated_rect[1]
        angle = rotated_rect[2]

        cv.line(img0, (box[0, 0], box[0, 1]), (box[1, 0], box[1, 1]), (0, 255, 0))
        cv.line(img0, (box[1, 0], box[1, 1]), (box[2, 0], box[2, 1]), (0, 255, 0))
        cv.line(img0, (box[2, 0], box[2, 1]), (box[3, 0], box[3, 1]), (0, 255, 0))
        cv.line(img0, (box[3, 0], box[3, 1]), (box[0, 0], box[0, 1]), (0, 255, 0))
        # rect = cv.boundingRect(c)
        # cv.rectangle(img0, (rect[0], rect[1]), (rect[0] + rect[2], rect[1]+ rect[3]), (0,255,0))

    imgshow(img0)
    show()