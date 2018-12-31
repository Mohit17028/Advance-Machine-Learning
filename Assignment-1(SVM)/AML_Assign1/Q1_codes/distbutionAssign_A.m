clc;
clear;
n=3000; % number of points that you want
center = [0 ,0]; % center coordinates of the circle [x0,y0]
radius = 1; % radius of the circle
angle = 2*pi*rand(n,1);
r = radius*sqrt(rand(n,1));
X1 = r.*cos(angle)+ center(1);
Y1 = r.*sin(angle)+ center(2);
circle=X1.*X1+Y1.*Y1;
index=find(abs(circle)<0.6);
X1(index)=[];
Y1(index)=[];
figure
plot(X1,Y1,'.b')

n=6000; % number of points that you want
center = [0 ,0]; % center coordinates of the circle [x0,y0]
radius = 2; % radius of the circle
angle = 2*pi*rand(n,1);
r = radius*sqrt(rand(n,1));
X2 = r.*cos(angle)+ center(1);
Y2 = r.*sin(angle)+ center(2);
circle=X2.*X2+Y2.*Y2;
index=find(abs(circle)<2.5);
X2(index)=[];
Y2(index)=[];
hold on
plot(X2,Y2,'.r')