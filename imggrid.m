clc;clear all;
p = imread('C:\Users\lujy.HVTEAM\Desktop\xhr.jpg'); %读取图像
[mm,nn,~] = size(p); %获取图像的大小
x = 0:nn/17:nn; %假设水平分成8格
y = 0:mm/15:mm; %假设垂直分成8格
M = meshgrid(x,y); %产生网格
N = meshgrid(y,x); %产生网格
imshow(p);
hold on
plot(x, N, 'r'); %画出水平横线
plot(M, y, 'r'); %画出垂直竖线
