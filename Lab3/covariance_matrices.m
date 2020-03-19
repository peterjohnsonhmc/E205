%% Lab3 EKF
% pjohnson@g.hmc.edu pking@g.hmc.edu
% This script calculates variances for our IMU acceleration measurements

data = readtable('2020_2_26__17_21_59_filtered.csv');
%data = readtable('2020_2_26__16_59_7_filtered.csv');


%% 
X = table2array(data(:,1));
Y = table2array(data(:,2));
Z = table2array(data(:,3)); 
T = table2array(data(:,4)); 
Latitude = table2array(data(:,5)); 
Longitude = table2array(data(:,6));
Yaw = table2array(data(:,7)); 
Pitch = table2array(data(:,8)); 
Roll = table2array(data(:,9)); 
AccelX = table2array(data(:,10)); 
AccelY = table2array(data(:,11)); 
AccelZ = table2array(data(:,12));

k=50;
AX = movmean(AccelX,k);
AY = movmean(AccelY,k);
T=706;
figure(1)
plot(AX(1:T));
ylim([-2,2])
%plot(AccelX);

figure(2)
plot(AY(1:T));
ylim([-2,2])

%Upon inspection, first 80 values seem to be within the same amount of
%noise

varx = var(AccelX(1:80));
vary = var(AccelY(1:80));

figure(3)
plot(X)

figure(4)
plot(Y)

figure(5)
plot(Yaw);