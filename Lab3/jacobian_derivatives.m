%% Lab3 EKF
% pjohnson@g.hmc.edu pking@g.hmc.edu
% This script calculates our jacobians for motion and mesurement models
% The expressions were then copy pasted into our python functions
%% Motion Model/Prediction
syms x y theta theta_pre v_x v_y v_theta a_xi a_yi dt
syms x_p y_p theta_p theta_prev_p v_x_p v_y_p v_theta_p
syms X_L Y_L;
state_prev=[v_x_p;
            x_p;
            v_y_p;
            y_p; 
            v_theta_p
            theta_p;
            theta_prev_p];
control = [a_xi;
           a_yi];
           

g=[v_x_p + (-a_yi*sin(theta_p)+a_xi*cos(theta_p))*dt;
    x_p + v_x_p*dt + 1/2*(-a_yi*sin(theta_p)+a_xi*cos(theta_p))*dt^2;
    v_y_p + (a_yi*cos(theta_p)+ a_xi*sin(theta_p))*dt;
    y_p + v_y_p*dt + 1/2*(a_yi*cos(theta_p)+ a_xi*sin(theta_p))*dt^2;
    (theta_p - theta_prev_p)/dt;
    v_theta_p*dt;
    theta_p];

%Jacobian wrt to state
G_x = jacobian(g, state_prev);
%Jacobian wrt to controls
G_u = jacobian(g, control);

%% Measurment Model
a = (X_L - x_p)*cos(theta_p + pi/2) -(Y_L- y_p)*sin(theta_p + pi/2);
b = (X_L - x_p)*sin(theta_p + pi/2) +(Y_L- y_p)*cos(theta_p + pi/2);
h = [a;
     b;
     theta_p];
 
H_x = jacobian(h, state_prev);


