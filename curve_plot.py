
def load_value(file, key):
    import re
    v = []
    with open(file, 'r', encoding='UTF-8') as f:
        for l in f:
            t = re.findall(r'(?<=' + key + r')\d+', l)
            if t:
                v.append(int(t[0]))
    return v


def load_line_value(file, line_key, key):
    import re
    v = []
    with open(file, 'r', encoding='UTF-8') as f:
        for l in f:
            if line_key in l:
                t = re.findall(r'(?<=' + key + r')\d+', l)
                if t:
                    v.append(int(t[0]))
    return v


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
    path = r'E:\log\\'
    file = '172.18.18.242_061500'
    suffix = '.log'

    lumI = load_line_value(path + file + suffix, r'IspDev = [0]', r'AveLumI = \[')
    lum = load_line_value(path + file + suffix, r'IspDev = [0]', r'AveLum = \[')
    TH = load_line_value(path + file +suffix, r'IspDev = [0]', r'TH = \[')
    figure(1)
    plot(lum)
    plot(lumI, 'r')
    plot(TH, 'g')

    # r = load_value(path + file + suffix, r'"image_color_r":')
    # g = load_value(path + file + suffix, r'"image_color_g":')
    # b = load_value(path + file + suffix, r'"image_color_b":')
    # lum = load_value(path + file + suffix, r'"image_luminance":')

    # figure(2)
    # plot(r, 'r')
    # plot(g, 'g')
    # plot(b, 'b')
    # plot(lum, 'y')
    show()
