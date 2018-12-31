dist1_x=rand(40);
dist1_x=(dist1_x(:)*10)-5;
dist1_y=dist1_x.*dist1_x;
offset=rand(40);
dist1_y=dist1_y+offset(:)*4;

dist2_x=dist1_x-5;
dist2_y=-dist1_y;
dist2_y=dist2_y+40;
figure
plot(dist1_x,dist1_y,'.r');
hold on
plot(dist2_x,dist2_y,'.b');
