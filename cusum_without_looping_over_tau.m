% Simulation parameters
num_simulations = 1000;
num_timesteps = 150;
attack_start = 100;
attack_magnitude = 2;
tau_values = linspace(10, 1, 70);
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
Pfa = zeros(size(tau_values));
Pd = zeros(size(tau_values));

% Initialize counters for false alarms and detections
false_alarms = zeros(size(tau_values));
detections = zeros(size(tau_values));

% Simulations without attacks (for false alarms)
for sim = 1:num_simulations
    [fa_counts] = run_simulation(tau_values, false, num_timesteps, ...
        attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L);
    false_alarms = false_alarms + fa_counts;
end

% Simulations with attacks (for detections)
for sim = 1:num_simulations
    [det_counts] = run_simulation(tau_values, true, num_timesteps, ...
        attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L);
    detections = detections + det_counts;
end

% Calculate probabilities
Pfa = false_alarms / num_simulations;
Pd = detections / num_simulations;

% ROC curve
figure;
plot(Pfa, Pd, 'b-o', 'LineWidth', 1.5);
xlabel('Probability of False Alarm (Pfa)');
ylabel('Probability of Detection (Pd)');
title('ROC Curve for Attack Detection Model');
grid on;

% xlim([0 1]);
% ylim([0 1]);


% Markers for tau values
for i = 1:2:length(tau_values)
    text(Pfa(i), Pd(i), sprintf('%.2f', tau_values(i)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
        'FontSize', 8, 'Color', 'red');
end

% AUC
auc = trapz(Pfa, Pd); % Integrate using trapezoidal method
annotation('textbox', [0.7, 0.05, 0.25, 0.1], 'String', sprintf('AUC: %.4f', auc), ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white', 'EdgeColor', 'black');
hold off;

% Function to run a single simulation
function [alarm_counts] = run_simulation(tau_values, with_attack, num_timesteps, attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L)
    x = zeros(2, num_timesteps);
    x_hat = zeros(2, num_timesteps);
    u = zeros(1, num_timesteps);
    r = zeros(1, num_timesteps);
    cusum = zeros(size(tau_values));
    alarm_counts = zeros(size(tau_values));

    % Initialize state
    x(:, 1) = randn(2, 1);
    x_hat(:, 1) = x(:, 1);

    % Simulation loop
    for k = 2:num_timesteps
        % State update
        v = mvnrnd([0; 0], R1)';
        x(:, k) = F*x(:, k-1) + G*u(k-1) + v;

        % Measurement
        n = sqrt(R2) * randn();
        y = C*x(:, k) + n;

        % Add attack if in attack window and with_attack is true
        if with_attack && k >= attack_start
            y = y + attack_magnitude;
        end

        % State estimate
        x_hat(:, k) = F*x_hat(:, k-1) + G*u(k-1);

        % Residual
        r(k) = y - C*x_hat(:, k);

        % Update state estimate
        x_hat(:, k) = x_hat(:, k) + L*r(k);

        % Control input for next step
        u(k) = K*x_hat(:, k);

        % CUSUM
        z_k = abs(r(k));
        cusum = max(0, cusum + z_k - bias);

        % Check for alarm for each tau value
        for tau_idx = 1:length(tau_values)
            if cusum(tau_idx) > tau_values(tau_idx)
                if ~with_attack || (with_attack && k >= attack_start)
                    alarm_counts(tau_idx) = 1;
                end
            end
        end
    end
end