%% Lab3 EKF
% pjohnson@g.hmc.edu pking@g.hmc.edu
% This script calculates our jacobians for motion and mesurement models
% The expressions were then copy pasted into our python functions

dat = readtable('2020_2_26__17_21_59_filtered.csv');
dat = table2array(dat);

X = dat(:,1);
Y = dat(:,2);
Z = dat(:,3); 
T = dat(:,4); 
Latitude = dat(:,5); 
Longitude = dat(:,6);
Yaw = dat(:,7); 
Pitch = dat(:,8); 
Roll = dat(:,9); 
AccelX = dat(:,10); 
AccelY = dat(:,11); 
AccelZ = dat(:,12);

figure(1)
plot(X,Y);