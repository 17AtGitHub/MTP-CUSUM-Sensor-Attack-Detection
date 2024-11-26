%% sim parameters
num_timesteps = 1000; 
% desired false alarm rate
false_alarm_rate = 0.10; 
% no of outputs
p = 1;

% constant error covariance matrix (same as in CUSUM model)
F = [0.84 0.23;
     -0.47 0.12]; 
G = [0.07;
     0.23]; 
C = [1 0]; 
K = [-1.85 -0.96]; 
% covariance matrices of noise
% R1: (2x2) is covariance matrix of process noise v(k): (2x1)
% R2: (1x1) is covariance matrix of measurement noise n(k): (1x1)
R1 = [0.45 -0.11;
      -0.11 0.20];
R2 = [1];
% also the known covariance of residual in the absence of attack
sigma = [1.70];
% kalman gain: constant for all timesteps
L = [0.31;
     -0.21];
%% tau calculation
dof = p;

% using inverse gamma function for chi-squared distribution
tau = chi2inv(1 - false_alarm_rate, dof);
%% declaration/init.
x = zeros(2, num_timesteps); % state x(k): 2x1
x_hat = zeros(2, num_timesteps); % est. state x_hat(k): 2x1
u = zeros(1, num_timesteps); % input u(k): 1x1
x(:, 1) = rand(2, 1); % choosing random values bw 0 and 1
u(:, 2) = K*x(:, 1); % control input for timestep 2
x_hat(:, 1) = rand(2, 1); % initializing the initial est = actual value
r = zeros(1, num_timesteps); % residual
zt = zeros(1, num_timesteps); % distance measure
alarms = zeros(1, num_timesteps); % alarms timesteps

%% sim loop
for k = 2:num_timesteps-1
    v = mvnrnd([0; 0], R1)';
    n = normrnd(0, sqrt(R2));
    
    % x(k) from x(k-1):
    x(:, k) = F*x(:, k-1) + G*u(k) + v;
    % x_hat(k) from x_hat(k-1):
    % x_hat(k|k-1) = F*x_hat(k-1|k-1) + G*u(k)
    x_hat(:, k) = F*x_hat(:, k-1) + G*u(k);
    
    % measured/observed quantities: y_bar
    y_bar = C*x(:, k) + n;

    % residual: using the estimate based on x_hat(k|k-1)
    r(k) = y_bar - C*x_hat(:, k);

    % state estimate update: getting x_hat(k|k)
    x_hat(:, k) = x_hat(:, k) + L*r(k);
    % control input for the next state update
    u(k+1) = K*x(:, k);
    
    % distance measure
    zt(k) = r(k)' * inv(sigma) * r(k);
    
    % alarm detection
    if zt(k) > tau
        alarms(k) = 1;
    end
    
end

%% false alarm rate
false_alarm_rate_observed = sum(alarms) / num_timesteps;
disp(['Observed False Alarm Rate: ', num2str(false_alarm_rate_observed)]);
