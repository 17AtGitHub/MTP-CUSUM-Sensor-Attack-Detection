%% sim parameters
num_timesteps = 1000;

%% system matrices
% x is modelled with 2 states: so x(k) is 2x1
% x(k+1) = F*x(k) + G*u(k) + v(k)
% y is modelled as a single dimensional output
% y(k) = C*x(k) + n(k)
% u(k) = K*x(k): 1x1 vector.
F = [0.84 0.23;
     -0.47 0.12]; 
G = [0.07;
     0.23]; 
C = [1 0]; 
K = [-1.85 -0.96]; 

% covariance matrices of noise
R1 = [0.45 -0.11;
      -0.11 0.20]; % Process noise covariance
R2 = [1];         % Measurement noise covariance

% Kalman gain: constant for all timesteps
L = [0.31;
     -0.21];

%% declarations
x = zeros(2, num_timesteps); % state x(k): 2x1
x_hat = zeros(2, num_timesteps); % est. state x_hat(k): 2x1
r = zeros(1, num_timesteps); % residual r(k): 1x1
u = zeros(1, num_timesteps); % input u(k): 1x1

%% initializations
x(:, 1) = rand(2, 1); % choosing random values between 0 and 1
u(:, 2) = K*x(:, 1); % control input for timestep 2
x_hat(:, 1) = rand(2, 1); % initializing the initial estimate to a random value

%% simulation
for k = 2:num_timesteps-1
    % process noise: from multivariate normal dist.
    v = mvnrnd([0; 0], R1)';
    % sensor noise: from univariate normal dist.
    n = normrnd(0, sqrt(R2));
    
    % x(k) from x(k-1):
    x(:, k) = F*x(:, k-1) + G*u(k) + v;
    % x_hat(k) from x_hat(k-1):
    x_hat(:, k) = F*x_hat(:, k-1) + G*u(k);
    
    % measured/observed quantities: y_bar
    y_bar = C*x(:, k) + n;

    % residual: using the estimate based on x_hat(k|k-1)
    r(k) = y_bar - C*x_hat(:, k);

    % state estimate update: getting x_hat(k|k)
    x_hat(:, k) = x_hat(:, k) + L*r(k);
    
    % control input for the next state update
    u(k+1) = K*x(:, k);
end

% average absolute residual over all timesteps
avg_residual = mean(abs(r));
disp(['Average absolute residual: ', num2str(avg_residual)]);

% residual plot over all timesteps
figure;
plot(1:num_timesteps-1, abs(r(1:num_timesteps-1)));
title('Absolute Residual Over Time');
xlabel('Timestep');
ylabel('Absolute Residual |r(k)|');
grid on;
