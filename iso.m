clc;clear all;

x = 100:800;
y = 1:56;

[X, Y] = meshgrid(x, y);

Z = -100*log10(Y./X);

mesh(X, Y, Z);