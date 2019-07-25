def calc(h, f, u, s1, s2):
    from math import atan, sqrt
    l = 0.4
    d = f * l / (u * (s2 - s1))
    d0 = sqrt(d*d - h*h)
    theta0 = atan(u *d * (s1 + s2) / (2 * f * (sqrt(d*d - h*h))))
    return d0, theta0


if __name__ == '__main__':
    f = 25e-3
    u = 3.45e-6
    h = 6.5
    w  = 4096
    s1 = 2802
    s2 = 2900

    d, theta = calc(h, f, u, s1 - w / 2, s2 - w / 2)

    print(round(d, 2), round(theta * 180 / 2.14, 2))