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


histogram(P00.Range_m_);
title('Histogram of range for azimuth = 0^{\circ}')
xlabel('Range[m]')
ylabel('Frequency')


histogram(P90.Range_m_);
title('Histogram of range for azimuth = 90^{\circ}')
xlabel('Range[m]')
ylabel('Frequency')