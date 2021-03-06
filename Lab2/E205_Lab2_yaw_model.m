%% Lab 2 E205
%pjohnson@g.hmc.edu and pking@g.hmc.edu

%% 1 Read the Data
% In this data, the imu and gps were held stationary
% Allows extraction of variance and covariance
dat = readtable('2020-02-08_08_22_47.csv');

%% 2 Histograms
% Histograms of range data
% Determined bin size to appear the most normal
figure(1)
histogram(dat.Yaw_degrees_);
title('Histogram of constant yaw')
xlabel('Yaw Angle[{\circ}]')
ylabel('Frequency')

%% 3 Sensor Variance
yaw_var = var(dat.Yaw_degrees_);

%% Sensor Model
%Gaussian and uniform distribution
%Assume ground truth (mu) is average of the data, i.e. our data is not
%biased

pd = fitdist(dat.Yaw_degrees_,'Normal');

x = 93:0.001:105; %This is just based on looking at the range of our data
y = pdf(pd,x);
figure(2)
plot(x,y,'LineWidth',2)

hold on
%Overlay a normalized histrogram to compare to normal pdf
histogram(dat.Yaw_degrees_, 'Normalization', 'pdf'); %
title('Normalized Histogram Fitted with Normal Distribution (constant yaw)')
xlabel('Yaw Angle [{\circ}]')
ylabel('Probability')
%Check calculated variances are the same
(pd.sigma^2) - yaw_var 