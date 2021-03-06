%% Lab3 EKF
% pjohnson@g.hmc.edu pking@g.hmc.edu
% This script calculates our jacobians for motion and mesurement models
% The expressions were then copy pasted into our python functions
%% Motion Model/Prediction
syms xd x yd y thetad theta theta_prev  a_xi a_yi yaw dt
syms X_L Y_L;
state_prev=[xd;
            x;
            yd;
            y; 
            thetad
            theta;
            theta_prev];
control = [a_xi;
           a_yi];
           

g= [xd + (-a_yi*sin(theta)+a_xi*cos(theta))*dt;
    x + xd*dt ;
    yd + (a_yi*cos(theta)+ a_xi*sin(theta))*dt;
    y + yd*dt;
    (theta - theta_prev)/dt;
    yaw;
    theta];

%Jacobian wrt to state
G_x = jacobian(g, state_prev);
%Jacobian wrt to controls
G_u = jacobian(g, control);

%% Measurment Model
a = (X_L - x)*cos(-theta + pi/2) -(Y_L- y)*sin(-theta + pi/2);
b = (X_L - x)*sin(-theta + pi/2) +(Y_L- y)*cos(-theta + pi/2);
h = [a;
     b;
     theta];
 
H_x = jacobian(h, state_prev);


