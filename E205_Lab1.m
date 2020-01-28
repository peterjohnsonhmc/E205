%% Lab 1 E205
%pjohnson@g.hmc.edu and pking@g.hmc.edu

%% 1 Read the Data
N90 = readtable('lab1_azimuth_-90.csv');
P00 = readtable('lab1_azimuth_00.csv');
P90 = readtable('lab1_azimuth_90.csv');
%% 2 Histograms

figure(1)
histogram(N90.Range_m_);
title('Histogram of range for azimuth = -90^{\circ}')
xlabel('Range[m]')
ylabel('Frequency')

figure(2)
histogram(P00.Range_m_);
title('Histogram of range for azimuth = 0^{\circ}')
xlabel('Range[m]')
ylabel('Frequency')

figure(3)
histogram(P90.Range_m_);
title('Histogram of range for azimuth = 90^{\circ}')
xlabel('Range[m]')
ylabel('Frequency')

%% 4 Create a Model
%Use beam finder model
%Gaussian and uniform distribution
%Assume ground truth (mu) is average of the data
z_kts = mean(N90.Range_m_);
pd = fitdist(N90.Range_m_,'Normal');

x = -3:0.1:3;
y = normpdf(x,pd.mu, pd.sigma);
figure(4)
plot(x,y)
%% 5 Transform and Plot the GPS measurmements
%use Azimuth=0 file
%Origin is average of measurements
orig_lat = mean(P00.Latitude);
orig_lon = mean(P00.Longitude);

max_lat = max(P00.Latitude);
min_lat = min(P00.Latitude);

%Transform to X and Y coordinate grid
%Use Equirectangular projection
R_earth = 6.3781*10^6; %[m] (radius of earth)

X = R_earth*(P00.Longitude-orig_lon)*cosd(orig_lat);
Y = R_earth*(P00.Latitude-orig_lat);

figure(5)
scatter(X,Y)
title('XY COordinate representation of GPS');
xlabel('X [m]')
ylabel('Y [m]')
xlim([-200 200])
ylim([-200 200])
%Notice that there is muchmore variance in the X data

%Look at covariance
covXY = cov(X,Y);

%% Implement Baye's Filter
