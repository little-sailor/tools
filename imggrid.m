clc;clear all;
p = imread('C:\Users\lujy.HVTEAM\Desktop\xhr.jpg'); %��ȡͼ��
[mm,nn,~] = size(p); %��ȡͼ��Ĵ�С
x = 0:nn/17:nn; %����ˮƽ�ֳ�8��
y = 0:mm/15:mm; %���费ֱ�ֳ�8��
M = meshgrid(x,y); %��������
N = meshgrid(y,x); %��������
imshow(p);
hold on
plot(x, N, 'r'); %����ˮƽ����
plot(M, y, 'r'); %������ֱ����
