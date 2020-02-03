%% Lab 1 E205
%pjohnson@g.hmc.edu and pking@g.hmc.edu

%% 1 Read the Data
N90 = readtable('lab1_azimuth_-90.csv');
P00 = readtable('lab1_azimuth_00.csv');
P90 = readtable('lab1_azimuth_90.csv');
%% 2 Histograms

figure(1)
histogram(N90.Range_m_, 'BinWidth', .00375);
title('Histogram of range for azimuth = -90^{\circ}')
xlabel('Range[m]')
ylabel('Frequency')

figure(2)
histogram(P00.Range_m_, 'BinWidth', .00375);
title('Histogram of range for azimuth = 0^{\circ}')
xlabel('Range[m]')
ylabel('Frequency')

figure(3)
histogram(P90.Range_m_, 'BinWidth', .02);
title('Histogram of range for azimuth = 90^{\circ}')
xlabel('Range[m]')
ylabel('Frequency')

% %Scatter plot
% figure(4)
% scatter(P90.Range_m_);
% figure(5)
% scatter(P00.Range_m_);
% figure(6)
% scatter(N90.Range_m_);

%% 4 Create a Model
%Use beam finder model
%Gaussian and uniform distribution
%Assume ground truth (mu) is average of the data, i.e. our data is not
%biased

pd = fitdist(N90.Range_m_,'Normal');

x = 9.25:0.0001:9.31; %This is just based on looking at the range of our data
y = pdf(pd,x);
figure(4)
plot(x,y,'LineWidth',2)

hold on
%Overlay a nomrlaized histrogram to compare to normal pdf
histogram(N90.Range_m_, 'BinWidth', .00375, 'Normalization', 'pdf'); %Normalize to have probabilities
title('Normalized Histogram Fitted with Normal Distribution (azimuth = -90{\circ})')
xlabel('Range[m]')
ylabel('Probability')

% Now need to develop function to get probability
% equation for P(Z|X)
% beam range finder model
sigma_hit = pd.sigma;   %Variance
mu_hit = pd.mu;         %Expected value
zt_star = mu_hit;
zt = N90.Range_m_;
zmax_range = 100; %max range is 100 [m]

%See phit function
%% or try learn intrinsic parameters
%Need to initialize lambda_short to 0.0001;
lambda_short = 0.0001;
Theta = learn_intrinsic_parameters(zt, zt_star, sigma_hit, lambda_short, zmax_range); 

%% 5 Transform and Plot the GPS measurements
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
title('XY Coordinate representation of GPS');
xlabel('X [m]')
ylabel('Y [m]')
xlim([-200 200])
ylim([-200 200])
%From the graph, notice that there is muchmore variance in the X data

%Look at covariance to verify this
covXY = cov(X,Y);

%% Implement Baye's Filter
%Given states
x1 = [-5.5,0.0];
x2 = [5.5,0.0]; %x(2,:)
%And probabilities of each state
px1 = 0.5;
px2 = 0.5;
%And a measurement
ztk = N90.Range_m_(1);
%Calculate Probability of being in each state using Baye's rule
% p(x|z) = p(z|x)*p(x)/p(z)

%Will first need p(z).
%From the notes, we have
%p(z) = sum( p(z|x=i)*p(x=i)

%Define true distances based on beam casting
ztk_star1 = 5.5+11; 
ztk_star2 = 11-5.5;
%Get Conditional measurement probabilities
pzx1 = phit(ztk,sigma_hit, ztk_star1,zmax_range);
pzx2 = phit(ztk,sigma_hit, ztk_star2,zmax_range);
%Get measurement probability as normalizer
pz = sum(pzx1*px1+pzx2*px2);
%Plug into baye's rule
px1z = pzx1*px1/pz;
px2z = pzx2*px2/pz;
%Note, these do not work, division by 0 results in NaN

%% Testing
ThetaTest = [1, 0.0, 0.0, 0.0, sigma_hit, lambda_short];
tq = beam_range_finder_model(ztk,zt_star, zmax_range, ThetaTest ); 
thit = phit(ztk,sigma_hit, zt_star,zmax_range);
tshort = pshort(ztk,lambda_short, zt_star);
tmax = pmax(ztk, zmax_range);
trand = prand(ztk, zmax_range);

