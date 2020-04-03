% Blimp EKF Localization
% Peter Johnson
% pjohnson@g.hmc.edu
% E205

% Load the Data
datatable = readtable('blimp_data.csv');
data = table2array(datatable);
% Timestamp, u_v, u_l, u_r, x, y, z, theta
% u is thrusts, xyz are in global coordinates, theta is yaw (CCW) 0 is east
timestamps = data(:,1);
u_v = data(:,2);
u_l = data(:,3);
u_r = data(:,4);
x_gps = data(:,5);
y_gps = data(:,6);
z_gps= data(:,7);
yaw = data(:,8);

%Provided Constants
GAMMA = 0.5;  %[m/s pulses]
ALPHA = 1.0;  %[m/s pulses]
BETA = 0.315; %[rad/ s pulses]

%Global Variable to Access in later functions
global dt 
dt = 0.1;     %[s]

%Convert thrust measurements to velocities
% Vertical Vel
s = GAMMA.*u_v;
%Linear Vel
v = ALPHA.*(u_l+u_r);
% Angular Vel
w = BETA.*(-u_l+u_r);

%{
  This code implements an EKF to track the state of an autonomous blimp.
  Assuming that the noise associated with the state transition function and
  sensors can be modelled as gaussian/normally distributed.
  The state transition function and measurement function can be nonlinear
  since they will be linearized.
  The state is [x,y,z, theta] in the global frame
  control is [s, v, w] the vertical, linear, and angular velcoties
  Measurement is [x_gps, y_gps, z_gps, yaw] (gps and compass)
  From the data, we see the timestep dt is 0.1 s
%}

% Need to determine variances from the data
% Got bounds of data from looking at plot to find constant value sections
VAR_X = var(x_gps(430:length(timestamps)));
VAR_Y = var(y_gps(430:length(timestamps)));
VAR_Z = var(z_gps(430:length(timestamps)));
VAR_YAW = var(yaw(430:length(timestamps)));

VAR_S = var(s(430:length(timestamps)));
VAR_V = var(v(430:length(timestamps)));
VAR_W = var(w(430:length(timestamps)));


global R Q
% Process Noise Covariance Matrix
R = [VAR_S 0 0;
     0 VAR_V 0;
     0 0 VAR_W];

% Measurment Noise Covariance Matrix
Q = [VAR_X 0 0 0;
     0 VAR_Y 0 0;
     0 0 VAR_Z 0;
     0 0 0 VAR_YAW];



% Allocate arrays
 state_ests = ones(4,length(timestamps));
 sigma_ests = ones(4,4,length(timestamps));


% Initialize state and covariance matrix
% We assume that we are starting at 0,0,0,0 -  the first state
% Assume uncertainty associated with initial state is just uncertainty of
% the sensors so use Q matrix
mu_t_prev = [x_gps(1);
             y_gps(1);
             z_gps(1);
             yaw(1)];

sigma_t_prev = Q;


% Main loop to run filter
for t = 1:length(timestamps)
    
    % Get control vector
    u_t = [s(t);
           v(t);
           w(t)];
     
    % Prediction step
    [mu_pred_t, sigma_pred_t] = predictionStep(mu_t_prev, sigma_t_prev, u_t);
    
    %Get measurement Vector
    z_t = [x_gps(t);
           y_gps(t);
           z_gps(t);
           yaw(t)];
    
    % Correction step
    [mu_est_t, sigma_est_t] = correctionStep(mu_pred_t, sigma_pred_t, z_t);
    
    % Log data
    state_ests(:,t) = mu_est_t;
    sigma_ests(:,:,t) = sigma_est_t;
    
    % Update current state est to be previous state est for clarity
    mu_t_prev = mu_est_t;
    sigma_t_prev = sigma_est_t; 
    
end

%Plotting section
figure(1)
scatter3(x_gps, y_gps, z_gps,'.', 'b')
hold on
plot3(state_ests(1,:), state_ests(2,:), state_ests(3,:),'r')
xlabel('X Position [m]')
ylabel('Y Position [m]')
zlabel('Z Position [m]')
xlim([-25 25])
ylim([-50, 0])
zlim([-5 45])
legend('Measurements', 'Estimated Path')

% figure(2)
% scatter(timestamps, x_gps)
% hold on
% scatter(timestamps, state_ests(1,:))
% xlabel('time stamps [s]')
% ylabel('x [m]')
% ylim([-5 5])
% legend('Measured', 'Estimated')
% 
% figure(3)
% scatter(timestamps, y_gps)
% hold on
% scatter(timestamps, state_ests(2,:))
% xlabel('time stamps [s]')
% ylabel('y [m]')
% legend('Measured', 'Estimated')
% 
% figure(4)
% scatter(timestamps, z_gps)
% hold on
% scatter(timestamps, state_ests(3,:))
% xlabel('time stamps [s]')
% ylabel('z [m]')
% legend('Measured', 'Estimated')

% Need theta stds 
theta_stds = sqrt(sigma_ests(4,4,:));
theta_stds = reshape(theta_stds,([1 800]));

STD1 = state_ests(4,:) + 2*theta_stds(1,:);
STD2 = state_ests(4,:) - 2*theta_stds(1,:);
 
figure(5)
scatter(timestamps, yaw, '.', 'b')
hold on
scatter(timestamps, state_ests(4,:),'.','r')
hold on 
plot(timestamps, STD1)
hold on
plot(timestamps, STD2)
ylim([-1.75, 1.75])
xlabel('Time Stamps [s]')
ylabel('Yaw Angle [rad]')
legend('Measured', 'Estimated', '+2 STD', '-2 STD')
 
% figure(6)
% scatter(timestamps, s)
% xlabel('time stamps [s]')
% ylabel('vertical vel [m/s]')
% 
% figure(7)
% scatter(timestamps, v)
% xlabel('time stamps [s]')
% ylabel('linear vel [m/s]')
% 
% figure(8)
% scatter(timestamps, w)
% xlabel('time stamps [s]')
% ylabel('angular vel [m\s]')


% Prediction Step Functions

function [mu_bar_t] = propagateState(mu_t_prev, u_t)
%{
Propogate/predict the state based on chosen motion model
         Use the nonlinear function g(mu_t_prev, u_t)

    Parameters:
    mu_t_prev (array)  -- the previous state estimate
    u_t (array)       -- the current control input (really is odometry)

    Returns:
    mu_bar_t (np.array)   -- the predicted state
 %}
    global dt
   %Destructure inputs
   mu_t_prev_cell = num2cell(mu_t_prev);
   u_t_cell = num2cell(u_t);
   [x, y, z, theta] = mu_t_prev_cell{:};
   [s, v, w] = u_t_cell{:};
   dt = 0.1;

    mu_bar_t = [x + v*cos(theta)*dt;
                y + v*sin(theta)*dt;
                z +            s*dt;
                wrapToPi(theta +        w*dt)];
end

function [G_x] = calcJacobianGx(mu_t_prev, u_t)
% calcJacobianGx Jacobian of g(x_prev_t,u_t) wrt to x
   
   global dt
   mu_t_prev_cell = num2cell(mu_t_prev);
   u_t_cell = num2cell(u_t);
   [x, y, z, theta] = mu_t_prev_cell{:};
   [s, v, w] = u_t_cell{:};
   
    G_x = [1 0 0 -v*sin(theta)*dt;
           0 1 0  v*cos(theta)*dt;
           0 0 1  0;
           0 0 0  1              ];

end

function [G_u] = calcJacobianGu(mu_t_prev, u_t)
% calcJacobianGu Jacobian of g(x_prev_t,u_t) wrt to u
   global dt
    mu_t_prev_cell = num2cell(mu_t_prev);
   u_t_cell = num2cell(u_t);
   [x, y, z, theta] = mu_t_prev_cell{:};
   [s, v, w] = u_t_cell{:};
   

   G_u = [0  cos(theta)*dt 0;
          0  sin(theta)*dt 0;
          dt 0             0;
          0  0            dt];
      
end

function [mu_bar_t, sigma_bar_t] = predictionStep(mu_t_prev, sigma_t_prev, u_t)
% predictionStep EKF prediction step

    global R

    mu_bar_t = propagateState(mu_t_prev, u_t);
    
    G_x = calcJacobianGx(mu_t_prev, u_t);
    G_u = calcJacobianGu(mu_t_prev, u_t);
    
    sigma_bar_t = G_x*sigma_t_prev*G_x.' + G_u*R*G_u.';

end

% Correction Step Functions

function [K_t] = calcKalmanGain(sigma_bar_t, H_t)
% calcKalmanGain 
    global Q
    H_t_T = H_t.';
    K_t = sigma_bar_t*H_t_T*inv(H_t*sigma_bar_t*H_t_T + Q);
    
end

function [mu_est_t, sigma_est_t] = correctionStep(mu_pred_t, sigma_pred_t, z_t)
% correctionStep EKF Correction step
    
     H_t = [1 0 0 0;
            0 1 0 0;
            0 0 1 0;
            0 0 0 1];
      
     K_t = calcKalmanGain(sigma_pred_t, H_t);
     
     % The measurment function z = h(x) is just z = x
     resid = z_t - mu_pred_t;
     resid(4) = wrapToPi(resid(4));
     
     mu_est_t = mu_pred_t + K_t*resid;
     sigma_est_t = (eye(4)-K_t*H_t)*sigma_pred_t;   

end

