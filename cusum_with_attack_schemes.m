%% sim parameters
num_timesteps = 1000;
bias = 1.2;
tau = 6.8;

%% system matrices
% x is modelled with 2 states: so x(k) is 2x1
% x(k+1) = F*x(k) + G*u(k) + v(k)
% y is modelled as a single dimensional output
% y(k) = C*x(k) + n(k) + do(k)
% u(k) = K*x(k): 1x1 vector.
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

%% declarations
x = zeros(2, num_timesteps); % state x(k): 2x1
x_hat = zeros(2, num_timesteps); % est. state x_hat(k): 2x1
r = zeros(1, num_timesteps); % residual r(k): 1x1
SUM = zeros(1, num_timesteps); % cusum SUM(k): 1x1
u = zeros(1, num_timesteps); % input u(k): 1x1

attack_windows = zeros(1, num_timesteps); 
alarms = zeros(1, num_timesteps); 
cusum = 0;

%% initializations
x(:, 1) = rand(2, 1); % choosing random values bw 0 and 1
u(:, 2) = K*x(:, 1); % control input for timestep 2
x_hat(:, 1) = rand(2, 1); % initializing the initial est = actual value

%% attack parameters: interval, duration, magnitude
attack_interval = 100;

%% five attack patterns: a, b, c, d, e
pattern = 'e';

%% simulation
for k=2:num_timesteps-1
    % process noise: from multivariate normal dist.
    v = mvnrnd([0; 0], R1)';
    % sensor noise: from univariate normal dist.
    n = normrnd(0, sqrt(R2));
    
    if mod(k, attack_interval) == 0
        % switch case on pattern
        switch pattern
            case 'a'
                % regular attack timestep, fixed duration and magnitude
                attack_duration = 10;
                attack_magnitude = 3;
                attack_start = k;
                attack_end = min(attack_start + attack_duration-1, num_timesteps);
            case 'b'
                % random start time
                attack_duration = 10;
                attack_magnitude = 3;
                attack_start = k + randi([0, attack_interval-attack_duration-1]);
                attack_end = min(attack_start+attack_duration-1, num_timesteps);
            case 'c'
                % random magnitude and start time. fixed duration
                attack_duration = 10;
                min_mag = 2;
                max_mag = 3.5;
                attack_magnitude = min_mag+(max_mag-min_mag).*rand;
                attack_start = k + randi([0, attack_interval-attack_duration-1]);
                attack_end = min(attack_start+attack_duration-1, num_timesteps);
            case 'd'
                % random duration, start time. fixed magnitude
                min_dur = 10;
                max_dur = 25;
                attack_duration = randi([min_dur, max_dur]);
                attack_magnitude = 3;
                attack_start = k + randi([0, attack_interval-attack_duration-1]);
                attack_end = min(attack_start+attack_duration-1, num_timesteps);
            case 'e'
                min_dur = 10;
                max_dur = 25;
                attack_duration = randi([min_dur, max_dur]);
                min_mag = 2;
                max_mag = 3.5;
                attack_magnitude = min_mag+(max_mag-min_mag).*rand;
                attack_start = k + randi([0, attack_interval-attack_duration-1]);
                attack_end = min(attack_start+attack_duration-1, num_timesteps);
        end
        attack_windows(attack_start:attack_end) = 1;
    end

    do_k = attack_windows(k)*attack_magnitude;
    
    % -------------- kalman filter --------------
    
    % x(k) from x(k-1):
    x(:, k) = F*x(:, k-1) + G*u(k) + v;
    % x_hat(k) from x_hat(k-1):
    % x_hat(k|k-1) = F*x_hat(k-1|k-1) + G*u(k)
    x_hat(:, k) = F*x_hat(:, k-1) + G*u(k);
    
    % measured/observed quantities: y_bar
    y_bar = C*x(:, k) + n + do_k;

    % residual: using the estimate based on x_hat(k|k-1)
    r(k) = y_bar - C*x_hat(:, k);

    % state estimate update: getting x_hat(k|k)
    x_hat(:, k) = x_hat(:, k) + L*r(k);
    % control input for the next state update
    u(k+1) = K*x(:, k);

    % --------------------------------------------

    % --------------- CUSUM ----------------------
    z_k = abs(r(k));
    cusum = max(0, cusum + z_k - bias);
    SUM(k) = cusum;
    if cusum > tau
        % alarm at this timestep
        alarms(k) = 1;
        disp(['Alarm at timestep ', num2str(k)]);
        % reset SUM to 0
        cusum = 0;
    end
    % --------------------------------------------
end

%% performance metrics
false_alarms = 0;
for i = 1:num_timesteps-1
    if alarms(i)==1 && attack_windows(i)~=1
        false_alarms = false_alarms + 1;
    end
end

alarm_indices = find(alarms == 1);
total_alarms = length(alarm_indices);
true_alarms = total_alarms - false_alarms;
detection_rate = true_alarms/total_alarms;
false_alarm_rate = false_alarms/total_alarms;
disp(['Total Alarms: ', num2str(total_alarms)]);
disp(['True Alarms: ', num2str(true_alarms)]);
disp(['False Alarms: ', num2str(false_alarms)]);
disp(['Detection Rate: ', num2str(detection_rate)]);
disp(['False Alarm Rate: ', num2str(false_alarm_rate)]);

%% variation of state x1, x2
figure(1);
subplot(2,1,1); plot(1:num_timesteps, x(1,:)); title('State x1');
subplot(2,1,2); plot(1:num_timesteps, x(2,:)); title('State x2');
%% plot cummulative sum, and attack windows, and alarm tips
figure(2);
% cusum over time
plot(1:num_timesteps, SUM, 'b', 'LineWidth', 0.8);
hold on;

% horizontal line showing the threshold tau
yline(tau, 'r--', 'LineWidth', 1);

% markers for vertical dashed lines at the boundaries of attack windows
attack_start = find(diff([0 attack_windows]) == 1);
attack_end = find(diff([attack_windows 0]) == -1);

for i = 1:length(attack_start)
    xline(attack_start(i), 'k--', 'LineWidth', 0.7); % start of attack window
    xline(attack_end(i) + 1, 'k--', 'LineWidth', 0.7); % end of attack window
end

% Mark the timesteps with alarms using red filled circles
plot(alarm_indices, SUM(alarm_indices), 'ro', 'MarkerFaceColor', 'r');

% Adding labels and title
xlabel('Timestep');
ylabel('CUSUM');
title('CUSUM over Timesteps with Attack Windows and Alarms');

% Finalize the plot
hold off;
