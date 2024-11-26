% Simulation parameters
num_simulations = 100;
num_timesteps = 100;
% Range of tau values as needed
tau_values = linspace(7, 3, 70); 
bias = 1.5; % Fixed bias
% System matrices
F = [0.84 0.23; -0.47 0.12];
G = [0.07; 0.23];
C = [1 0];
K = [-1.85 -0.96];
R1 = [0.45 -0.11; -0.11 0.20];
R2 = 1;
L = [0.31; -0.21];

% Probability matrices initialization
Pfa_case1 = zeros(size(tau_values));
Pd_case1 = zeros(size(tau_values));
Pfa_case2 = zeros(size(tau_values));
Pd_case2 = zeros(size(tau_values));
Pfa_case3 = zeros(size(tau_values));
Pd_case3 = zeros(size(tau_values));

% Case 1: Variable start time, fixed magnitude
attack_magnitude = 3;
for tau_idx = 1:length(tau_values)
    tau = tau_values(tau_idx);
    false_alarms = 0;
    detections = 0;
    
    % Simulations without attacks (for false alarms)
    for sim = 1:num_simulations
        [false_alarm, ~] = run_simulation(tau, false, num_timesteps, ...
            randi([60, 85]), attack_magnitude, bias, F, G, C, K, R1, R2, L);
        false_alarms = false_alarms + false_alarm;
    end
    
    % Simulations with attacks (for detections)
    for sim = 1:num_simulations
        attack_start = randi([60, 85]);
        [~, detection] = run_simulation(tau, true, num_timesteps, ...
            attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L);
        detections = detections + detection;
    end
    
    % Getting probabilities
    Pfa_case1(tau_idx) = false_alarms / num_simulations;
    Pd_case1(tau_idx) = detections / num_simulations;
end

% Case 2: Fixed start time, variable magnitude
attack_start = 80;
for tau_idx = 1:length(tau_values)
    tau = tau_values(tau_idx);
    false_alarms = 0;
    detections = 0;
    
    % Simulations without attacks (for false alarms)
    for sim = 1:num_simulations
        [false_alarm, ~] = run_simulation(tau, false, num_timesteps, ...
            attack_start, 0, bias, F, G, C, K, R1, R2, L); % No attack
        false_alarms = false_alarms + false_alarm;
    end
    
    % Simulations with attacks (for detections)
    for sim = 1:num_simulations
        attack_magnitude = rand() * (3.5 - 2.8) + 2.8; 
        [~, detection] = run_simulation(tau, true, num_timesteps, ...
            attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L);
        detections = detections + detection;
    end
    
    % Getting probabilities
    Pfa_case2(tau_idx) = false_alarms / num_simulations;
    Pd_case2(tau_idx) = detections / num_simulations;
end

% Case 3: Variable start time and magnitude
for tau_idx = 1:length(tau_values)
    tau = tau_values(tau_idx);
    false_alarms = 0;
    detections = 0;
    
    % Simulations without attacks (for false alarms)
    for sim = 1:num_simulations
        [false_alarm, ~] = run_simulation(tau, false, num_timesteps, ...
            randi([60, 85]), 0, bias, F, G, C, K, R1, R2, L); % No attack
        false_alarms = false_alarms + false_alarm;
    end
    
    % Simulations with attacks (for detections)
    for sim = 1:num_simulations
        attack_start = randi([60, 85]); 
        attack_magnitude = rand() * (3.5 - 2.8) + 2.8;
        [~, detection] = run_simulation(tau, true, num_timesteps, ...
            attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L);
        detections = detections + detection;
    end
    
    % Getting probabilities
    Pfa_case3(tau_idx) = false_alarms / num_simulations;
    Pd_case3(tau_idx) = detections / num_simulations;
end

% ROC curves
figure;
hold on;
plot(Pfa_case1, Pd_case1, 'b-o', 'LineWidth', 1.5);
plot(Pfa_case2, Pd_case2, 'r-s', 'LineWidth', 1.5);
plot(Pfa_case3, Pd_case3, 'g-^', 'LineWidth', 1.5);
xlabel('Probability of False Alarm (Pfa)');
ylabel('Probability of Detection (Pd)');
title('ROC Curves for Different Attack Schemes');
legend('Variable Start Time, Fixed Magnitude', ...
       'Fixed Start Time, Variable Magnitude', ...
       'Variable Start Time and Magnitude');
grid on;

% AUC for each case
auc_case1 = trapz(Pfa_case1, Pd_case1);
auc_case2 = trapz(Pfa_case2, Pd_case2);
auc_case3 = trapz(Pfa_case3, Pd_case3);
annotation('textbox', [0.7, 0.05, 0.25, 0.1], ...
    'String', sprintf('AUC Case 1: %.4f\nAUC Case 2: %.4f\nAUC Case 3: %.4f', ...
                      auc_case1, auc_case2, auc_case3), ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white', 'EdgeColor', 'black');
hold off;

% Function to run a single simulation
function [false_alarm, detection] = run_simulation(tau, with_attack, num_timesteps, attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L)
    x = zeros(2, num_timesteps);
    x_hat = zeros(2, num_timesteps);
    u = zeros(1, num_timesteps);
    r = zeros(1, num_timesteps);
    cusum = 0;
    false_alarm = 0;
    detection = 0;
    
    % State Initialization
    x(:, 1) = randn(2, 1);
    x_hat(:, 1) = x(:, 1);
    
    % Simulation loop
    for k = 2:num_timesteps
        % State update
        v = mvnrnd([0; 0], R1)';
        x(:, k) = F * x(:, k-1) + G * u(k-1) + v;
        
        % Measurement
        n = sqrt(R2) * randn();
        y = C * x(:, k) + n;
        
        % Add attack if in attack window and with_attack is true
        if with_attack && k >= attack_start
            y = y + attack_magnitude;
        end
        
        % State estimate
        x_hat(:, k) = F * x_hat(:, k-1) + G * u(k-1);
        
        % Residual
        r(k) = y - C * x_hat(:, k);
        
        % Update state estimate
        x_hat(:, k) = x_hat(:, k) + L * r(k);
        
        % Control input for next step
        u(k) = K * x_hat(:, k);
        
        % CUSUM
        z_k = abs(r(k));
        cusum = max(0, cusum + z_k - bias);
        
        % Check for alarm
        if cusum > tau
            if ~with_attack
                false_alarm = 1;
                return;
            elseif k >= attack_start
                detection = 1;
                return;
            end
        end
    end
end
