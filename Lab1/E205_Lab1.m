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
sigma = pd.sigma;
mu = pd.mu;
zmax = 100; %max range is 100 [m]


    
function p = prob_z_x(z_meas, z_true)

    %Precompute all of the pdfs
    for i = 1:length(N90.Range_m_)
        bigN(i) = (1/(sqrt(2*pi*sigma^2)))*exp(-0.5*(N90.Range_m_(i)-mu)^2/sigma^2);
    end
    normalizer = sum(bigN);
    normalizer = normalizer^-1;
    bigN =(1/(sqrt(2*pi*sigma^2)))*exp(-0.5*(zmeas-mu).^2/sigma^2);
    pzgivenx = normalizer*bigN;
    
end

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
%Notice that there is muchmore variance in the X data

%Look at covariance
covXY = cov(X,Y);

%% Implement Baye's Filter
%Given states
x1 = [-5.5,0.0];
x2 = [5.5,0.0]; %x(2,:)
%And probabilities of each state
p_x1 = 0.5;
p_x2 = 0.5;
%And a measurement
z = N90.Range_m_(1);
%Calculate Probability of being in each state using Baye's rule
% p(x|z) = p(z|x)*p(x)/p(z)

%Will first need p(z).
%From the notes, we have
%p(z) = sum( p(z|x=i)*p(x=i)

