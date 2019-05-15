def load_lut(file):
    x = []
    y = []
    with open(file, "rb") as f:
        content = str(f.read(), encoding='utf8').split('\n')
        for s in content:
            if ',' in s:
                x.append(int(s.split(',')[0]))
                y.append(int(s.split(',')[1]))
    return x, y


def interpolation(arr_len, step, x, y):
    y_new = []
    for i in range(arr_len):
        zone = int(i/step)
        y0 = (y[zone] - y[zone + 1]) / (x[zone] - x[zone + 1]) * (i - x[zone + 1]) + y[zone + 1]
        y_new.append(round(y0))

    y_new.append(y[-1])
    return y_new


def format_string(lut):
    import re
    s = ", ".join([str(i) for i in lut])
    patten = re.compile(r'(([0-9]{1,4}, ){16,16})')  # {16,16} 不能有空格
    t = re.sub(patten, r'\1\n', s)
    ts = '#ifndef _LUT_H_\n' \
         '#define _LUT_H_\n' \
         'int g_qlut['+str(len(lut))+'] = {\n'
    te = '\n};\n' \
         '#endif'
    t = ts + t + te
    return t


def save_lut(file, lut_s):
    with open(file, 'w') as f:
        f.write(lut_s)


if __name__ == '__main__':
    from matplotlib.pyplot import plot, subplot, show

    path = r'Y:/nfs//'
    name = 'cab_1511'
    suffix = ''
    arr_len = 1330
    step = 10

    x, y = load_lut(path + name + suffix)

    c0 = interpolation(arr_len, step, x, y)

    plot([x for x in range(arr_len+1)], c0)
    plot(x, y, 'r+')
    show()

    save_lut(path + name + '.h', format_string(c0))