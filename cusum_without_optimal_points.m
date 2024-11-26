%% sim-parameters
num_timesteps = 1000;
b = 3;
b_bar = 1.0372; % base value of bias
bias_values = [0.70*b_bar, 1*b_bar, 1.30*b_bar];
tau = 20; % threshold

%% system matrices
F = [0.84 0.23; -0.47 0.12];
G = [0.07; 0.23];
C = [0 1];
K = [-1.85 -0.96];
R1 = [0.45 -0.11; -0.11 0.20];
R2 = [1];
L = [0.31; -0.21];
Sigma_r = [1.70];

%% state, estimation, and residual vectors
x = zeros(2, num_timesteps); % (2x1)
x_hat = zeros(2, num_timesteps); % (2x1)
r = zeros(1, num_timesteps); % residual signal
S = zeros(1, num_timesteps); % cusum vec
attack = zeros(1, num_timesteps); % attack signal

%% initial conditions
x(:, 1) = rand(2, 1); % random initialization of x

%% attack parameters
attack_interval = 100; % introduce attack every 100 timesteps
attack_duration = 10; % duration of each attack
attack_magnitude = 8; % constant attack signal magnitude

%% empty containers to store alarms and attacks
alarms = [];
attack_windows = [];

%% looping over timesteps to simulate
for k = 1:num_timesteps-1
    % noise params
    v = mvnrnd([0; 0], R1)'; % process noise
    n = normrnd(0, sqrt(R2)); % sensor noise
    
    % attack signal
    if mod(k, attack_interval) == 0
        attack_start = k;
        attack_end = min(k + attack_duration - 1, num_timesteps);
        attack_windows = [attack_windows, attack_start:attack_end];
    end
    
    if ismember(k, attack_windows)
        do_k = attack_magnitude;
    else
        do_k = 0; % no attack case
    end
    
    % system dynamics
    u = K * x(:, k); % modelling input from state
    
    % state update from system matrices
    x(:, k+1) = F * x(:, k) + G * u + v;
    
    % "measured" output considering the attack sig
    y_bar = C * x(:, k) + n + do_k;
    
    % next state estimation using kalman filter
    % L known: coming from min. error covariance.
    x_hat(:, k+1) = F * x_hat(:, k) + G * u + L * (y_bar - C * x_hat(:, k));
    
    % residual signal from measured output affected by attack
    r(k) = y_bar - C * x_hat(:, k);
    
    % attack detection from CUSUM scheme
    z_k = abs(r(k)); % distance from the abs value of residual signal
    if S(k) <= tau
        S(k+1) = max(0, S(k) + z_k - b);
    else
        S(k+1) = 0;
        % alarm goes off at this timestep
        alarms(end+1) = k;
        disp(['Alarm at timestep ', num2str(k)]);
    end
end

%% cusum scheme performance metrics
% false alarm rate
false_alarms = 0;
for i = 1:length(alarms)
    if ~ismember(alarms(i), attack_windows)
        false_alarms = false_alarms + 1;
    end
end
total_alarms = length(alarms);
false_alarm_rate = false_alarms / total_alarms;
disp(['Total Alarms: ', num2str(total_alarms)]);
disp(['False Alarms: ', num2str(false_alarms)]);
disp(['False Alarm Rate: ', num2str(false_alarm_rate)]);

%% variation of state x: [x1 x2] and CUSUM value over the timespan
figure;
subplot(3,1,1); plot(1:num_timesteps, x(1,:)); title('State x1');
subplot(3,1,2); plot(1:num_timesteps, x(2,:)); title('State x2');
subplot(3,1,3);
plot(1:num_timesteps, S);
title('Cumulative Sum S(k)');
hold on;

% Plot attack windows
for k = 1:num_timesteps
    if mod(k, attack_interval) == 0
        attack_start = k;
        attack_end = min(k + attack_duration - 1, num_timesteps);
        plot([attack_start, attack_start], [0, max(S)], 'r--');
        plot([attack_end, attack_end], [0, max(S)], 'r--');
    end
end

% Plot threshold line
plot([1, num_timesteps], [tau, tau], 'k--', 'LineWidth', 1.5);

legend('CUSUM', 'Attack Windows', 'Threshold');
xlabel('Time Step');
ylabel('CUSUM Value');