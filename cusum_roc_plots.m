%% sim parameters
num_timesteps = 1000;
bias = 1.2;
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

%% range of tau values
tau_values = linspace(2, 3.5, 50); % Example range of tau values
Pd = zeros(size(tau_values));
Pfa = zeros(size(tau_values));

%% attack parameters: interval, duration, magnitude
attack_interval = 100;

pattern = 'a';
%% Step 2: Run the simulation for each tau
for t_idx = 1:length(tau_values)
    tau = tau_values(t_idx);
    cusum = 0;
    attack_windows = zeros(1, num_timesteps); 
    alarms = zeros(1, num_timesteps); 

    %% declarations
    x = zeros(2, num_timesteps); % state x(k): 2x1
    x_hat = zeros(2, num_timesteps); % est. state x_hat(k): 2x1
    r = zeros(1, num_timesteps); % residual r(k): 1x1
    SUM = zeros(1, num_timesteps); % cusum SUM(k): 1x1
    u = zeros(1, num_timesteps); % input u(k): 1x1

    %% initializations
    x(:, 1) = rand(2, 1); % choosing random values bw 0 and 1
    u(:, 2) = K*x(:, 1); % control input for timestep 2
    x_hat(:, 1) = rand(2, 1); % initializing the initial est = actual value


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
            % reset SUM to 0
            cusum = 0;
        end
        % --------------------------------------------
    end


    %% Compute Pd and Pfa
    false_alarms = sum(alarms & ~attack_windows); % alarms when no attack
    total_alarms = sum(alarms); % total alarms raised
    true_alarms = sum(alarms & attack_windows); % alarms during attacks
    
    Pfa(t_idx) = false_alarms / total_alarms;
    Pd(t_idx) = true_alarms / total_alarms;
end

%% Step 4: Plot ROC curve
figure;
plot(Pfa, Pd, 'b-o', 'LineWidth', 1.5);
xlabel('Probability of False Alarm (Pfa)');
ylabel('Probability of Detection (Pd)');
title('ROC Curve');
grid on;

% Add markers for each tau value
for t_idx = 1:length(tau_values)
    text(Pfa(t_idx), Pd(t_idx), sprintf('%.2f', tau_values(t_idx)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
        'FontSize', 8, 'Color', 'red');
end
